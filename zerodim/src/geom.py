"""Geometry utilities for plane fitting, gradients, and mask moments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass(frozen=True)
class PlaneFit:
    ax: float
    ay: float
    bias: float
    inlier_ratio: float
    iterations: int
    threshold: float
    mode_used: str

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "ax": float(self.ax),
            "ay": float(self.ay),
            "bias": float(self.bias),
            "inlier_ratio": float(self.inlier_ratio),
            "iterations": int(self.iterations),
            "threshold": float(self.threshold),
            "mode_used": self.mode_used,
        }


def _collect_points(depth: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        raise ValueError("Not enough pixels inside mask for plane fitting")
    z = depth[ys, xs].astype(np.float64)
    finite = np.isfinite(z)
    xs = xs[finite]
    ys = ys[finite]
    z = z[finite]
    if xs.size < 3:
        raise ValueError("Insufficient finite pixels for plane fitting")
    return xs, ys, z


def _least_squares_plane(xs: np.ndarray, ys: np.ndarray, z: np.ndarray) -> np.ndarray:
    coords = np.column_stack([xs, ys, np.ones_like(xs)])
    coeffs, *_ = np.linalg.lstsq(coords, z, rcond=None)
    return coeffs.astype(np.float64)


def fit_plane_ransac(
    depth: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    max_iters: int = 2048,
    seed: int = 42,
    min_inlier_ratio: float = 0.6,
) -> PlaneFit:
    """Fit a plane z = ax + by + c using RANSAC with optional fallback to LS."""

    xs, ys, z = _collect_points(depth, mask)
    coords = np.column_stack([xs, ys, np.ones_like(xs)])
    rng = np.random.default_rng(seed)

    best_ratio = -1.0
    best_coeffs: np.ndarray | None = None
    best_inliers: np.ndarray | None = None
    iterations = 0

    for _ in range(max_iters):
        iterations += 1
        sample_idx = rng.choice(xs.size, size=3, replace=False)
        sample = coords[sample_idx]
        if np.linalg.matrix_rank(sample) < 3:
            continue
        coeffs = np.linalg.solve(sample, z[sample_idx])
        residuals = np.abs(coords @ coeffs - z)
        inliers = residuals < threshold
        ratio = float(inliers.mean())
        if ratio > best_ratio:
            best_ratio = ratio
            best_coeffs = coeffs
            best_inliers = inliers

    if best_coeffs is not None and best_inliers is not None and np.count_nonzero(best_inliers) >= 3:
        refined_coords = coords[best_inliers]
        refined_z = z[best_inliers]
        coeffs, *_ = np.linalg.lstsq(refined_coords, refined_z, rcond=None)
        residuals = np.abs(coords @ coeffs - z)
        tolerance = max(threshold * 3.0, 5.0)
        combined = residuals < tolerance
        if np.count_nonzero(combined) >= 3:
            coeffs, *_ = np.linalg.lstsq(coords[combined], z[combined], rcond=None)
            best_ratio = float(combined.mean())
        else:
            coeffs = best_coeffs
    else:
        coeffs = _least_squares_plane(xs, ys, z)
        best_ratio = 0.0

    mode_used = "ransac" if best_ratio >= min_inlier_ratio else "least_squares"
    if mode_used == "least_squares" and best_ratio < min_inlier_ratio:
        coeffs = _least_squares_plane(xs, ys, z)
        best_ratio = float(np.nanmean(np.abs(coords @ coeffs - z) < threshold))

    plane = PlaneFit(
        ax=float(coeffs[0]),
        ay=float(coeffs[1]),
        bias=float(coeffs[2]),
        inlier_ratio=best_ratio,
        iterations=iterations,
        threshold=threshold,
        mode_used=mode_used,
    )
    return plane


def apply_plane(depth: np.ndarray, plane: PlaneFit) -> np.ndarray:
    ys, xs = np.indices(depth.shape, dtype=np.float64)
    return plane.ax * xs + plane.ay * ys + plane.bias


def flatten_with_plane(depth: np.ndarray, mask: np.ndarray, plane: PlaneFit, fill_value: float) -> np.ndarray:
    result = depth.copy()
    plane_map = apply_plane(depth, plane)
    result[mask] = depth[mask] - plane_map[mask] + fill_value
    return result


def residual_map(depth: np.ndarray, plane: PlaneFit) -> np.ndarray:
    return depth - apply_plane(depth, plane)


def erode_mask(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 0:
        return mask
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    eroded = cv2.erode(mask.astype(np.uint8), structure, iterations=1)
    if np.count_nonzero(eroded) == 0:
        return mask
    return eroded.astype(bool)


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None or len(points) < 3:
        return mask
    hull = cv2.convexHull(points)
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    return filled.astype(bool)


def fit_plane_huber(
    depth: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    max_iters: int = 32,
) -> PlaneFit:
    xs, ys, z = _collect_points(depth, mask)
    coords = np.column_stack([xs, ys, np.ones_like(xs)])
    coeffs, *_ = np.linalg.lstsq(coords, z, rcond=None)
    delta = max(threshold, 1e-3)

    for _ in range(max_iters):
        residuals = coords @ coeffs - z
        scale = np.median(np.abs(residuals)) / 0.6745
        scale = float(scale if np.isfinite(scale) and scale > 1e-6 else 1.0)
        normalized = residuals / scale
        weights = np.ones_like(normalized)
        mask_large = np.abs(normalized) > delta
        weights[mask_large] = delta / np.abs(normalized[mask_large])
        lhs = coords * weights[:, None]
        rhs = z * weights
        new_coeffs, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        if np.linalg.norm(new_coeffs - coeffs) < 1e-9:
            coeffs = new_coeffs
            break
        coeffs = new_coeffs

    residuals = coords @ coeffs - z
    inlier_ratio = float(np.mean(np.abs(residuals) < threshold)) if threshold > 0 else 0.0
    plane = PlaneFit(
        ax=float(coeffs[0]),
        ay=float(coeffs[1]),
        bias=float(coeffs[2]),
        inlier_ratio=inlier_ratio,
        iterations=max_iters,
        threshold=threshold,
        mode_used="huber",
    )
    return plane


def compute_grad_reports(depth: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    mask_valid = mask & np.isfinite(depth)
    grad_x = depth[:, 1:] - depth[:, :-1]
    grad_y = depth[1:, :] - depth[:-1, :]
    mask_x = mask_valid[:, 1:] & mask_valid[:, :-1]
    mask_y = mask_valid[1:, :] & mask_valid[:-1, :]

    def _stats(values: np.ndarray, mask_vals: np.ndarray) -> Dict[str, float]:
        if not np.any(mask_vals):
            return {"mean": 0.0, "abs_mean": 0.0}
        data = values[mask_vals]
        return {
            "mean": float(np.nanmean(data)),
            "abs_mean": float(np.nanmean(np.abs(data))),
        }

    stats_x = _stats(grad_x, mask_x)
    stats_y = _stats(grad_y, mask_y)
    return {
        "mean_dx": stats_x["mean"],
        "mean_dy": stats_y["mean"],
        "abs_mean_dx": stats_x["abs_mean"],
        "abs_mean_dy": stats_y["abs_mean"],
    }


def shape_moments(mask: np.ndarray) -> Dict[str, float]:
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        raise ValueError("Mask does not contain enough pixels for moments")

    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    center = coords.mean(axis=0)
    centered = coords - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major_vec = eigvecs[:, 0]
    theta = float(-np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    lambda_ratio = float(np.sqrt(eigvals[0] / max(eigvals[1], 1e-9)))
    return {
        "theta_deg": theta,
        "lambda_ratio": lambda_ratio,
    }


def morphological_closing(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 0:
        return mask
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, structure, iterations=1)
    return closed.astype(bool)


__all__ = [
    "PlaneFit",
    "fit_plane_ransac",
    "fit_plane_huber",
    "flatten_with_plane",
    "residual_map",
    "compute_grad_reports",
    "shape_moments",
    "morphological_closing",
    "erode_mask",
    "convex_hull_mask",
]
