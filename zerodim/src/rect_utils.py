"""Image utilities for mask refinement and affine rectification."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RefinementParams:
    closing_kernel: int
    erosion_kernel: int


def load_image_rgba(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    if image.getbands() != ("R", "G", "B", "A"):
        image = image.convert("RGBA")
    return image


def mask_overlay_image(raw_rgb: np.ndarray, mask_bool: np.ndarray) -> Image.Image:
    overlay = raw_rgb.copy()
    edges = cv2.Canny((mask_bool.astype(np.uint8) * 255), 32, 96)
    overlay[edges > 0] = [255, 0, 0]
    return Image.fromarray(overlay, mode="RGB")


def refine_mask(mask_bool: np.ndarray, erosion_px: int, closing_px: int) -> Tuple[np.ndarray, RefinementParams]:
    kernel_close = max(1, closing_px)
    kernel_erode = max(1, erosion_px)

    if kernel_close > 0:
        structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close))
        mask_closed = cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_CLOSE, structure, iterations=1)
    else:
        mask_closed = mask_bool.astype(np.uint8)

    if kernel_erode > 0:
        structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_erode, kernel_erode))
        mask_clean = cv2.erode(mask_closed, structure, iterations=1)
        if np.count_nonzero(mask_clean) == 0:
            mask_clean = mask_closed
    else:
        mask_clean = mask_closed

    return mask_clean.astype(bool), RefinementParams(closing_kernel=kernel_close, erosion_kernel=kernel_erode)


def _wrap_angle(theta: float) -> float:
    wrapped = (theta + 180.0) % 360.0 - 180.0
    if wrapped > 90.0:
        wrapped -= 180.0
    if wrapped < -90.0:
        wrapped += 180.0
    return wrapped

def _rotation_to_front_normal(normal: np.ndarray) -> np.ndarray:
    n = normal.astype(np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        raise ValueError("Plane normal is degenerate")
    n /= norm
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(n, target), -1.0, 1.0))
    if np.isclose(dot, 1.0, atol=1e-6):
        return np.eye(3, dtype=np.float64)
    if np.isclose(dot, -1.0, atol=1e-6):
        rvec = np.array([np.pi, 0.0, 0.0], dtype=np.float64)
        rotation, _ = cv2.Rodrigues(rvec)
        return rotation
    axis = np.cross(n, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    axis /= axis_norm
    angle = float(np.arccos(dot))
    rvec = axis * angle
    rotation, _ = cv2.Rodrigues(rvec.astype(np.float64))
    return rotation


def _mask_corners(mask_bool: np.ndarray) -> np.ndarray:
    points = cv2.findNonZero(mask_bool.astype(np.uint8))
    if points is None or len(points) < 3:
        raise ValueError("Mask does not contain enough points for rectification")
    rect = cv2.minAreaRect(points)
    corners = cv2.boxPoints(rect)
    return corners.astype(np.float64)


def _moments_from_mask(mask_bool: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask_bool)
    if xs.size < 3:
        raise ValueError("Rectified mask is empty or degenerate")
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    center = coords.mean(axis=0)
    centered = coords - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major = eigvecs[:, 0]
    theta = _wrap_angle(float(-np.degrees(np.arctan2(major[1], major[0]))))
    lambda_ratio = float(np.sqrt(eigvals[0] / max(eigvals[1], 1e-9)))
    return theta, lambda_ratio


def _matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _matrix_to_euler(rotation: np.ndarray) -> Tuple[float, float, float]:
    r = rotation
    if np.isclose(r[2, 0], 1.0) or np.isclose(r[2, 0], -1.0):
        pitch = -np.pi / 2.0 if r[2, 0] < 0 else np.pi / 2.0
        yaw = 0.0
        roll = np.arctan2(-r[0, 1], -r[0, 2])
    else:
        pitch = -np.arcsin(r[2, 0])
        cp = np.cos(pitch)
        roll = np.arctan2(r[2, 1] / cp, r[2, 2] / cp)
        yaw = np.arctan2(r[1, 0] / cp, r[0, 0] / cp)
    return roll, pitch, yaw


def rectify_plane_homography(
    image_rgb: np.ndarray,
    mask_bool: np.ndarray,
    plane,
    mask_area_raw: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    ys, xs = np.nonzero(mask_bool)
    if xs.size < 4:
        raise ValueError("Mask is empty after refinement; cannot rectify")

    height, width = mask_bool.shape
    focal = 1.2 * float(max(width, height))
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    K_inv = np.linalg.inv(K)

    normal = np.array([float(plane.ax), float(plane.ay), -1.0], dtype=np.float64)
    normal_norm = max(np.linalg.norm(normal), 1e-9)
    normal_unit = normal / normal_norm

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(normal_unit, z_axis), -1.0, 1.0))
    if np.isclose(dot, 1.0, atol=1e-8):
        rotation = np.eye(3, dtype=np.float64)
    elif np.isclose(dot, -1.0, atol=1e-8):
        rotation, _ = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0], dtype=np.float64))
    else:
        axis = np.cross(normal_unit, z_axis)
        axis /= max(np.linalg.norm(axis), 1e-9)
        angle = float(np.arccos(dot))
        rvec = axis * angle
        rotation, _ = cv2.Rodrigues(rvec.astype(np.float64))

    homography = K @ rotation @ K_inv

    mask_points = cv2.findNonZero(mask_bool.astype(np.uint8))
    if mask_points is None:
        raise ValueError("Mask is empty; cannot rectify")
    mask_points = mask_points.astype(np.float32)
    warped_points = cv2.perspectiveTransform(mask_points, homography).reshape(-1, 2)
    min_xy = warped_points.min(axis=0)
    max_xy = warped_points.max(axis=0)
    out_w = int(np.ceil(max_xy[0] - min_xy[0])) + 1
    out_h = int(np.ceil(max_xy[1] - min_xy[1])) + 1
    out_w = max(out_w, 1)
    out_h = max(out_h, 1)

    translate = np.array(
        [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_final = translate @ homography

    rectified_rgb = cv2.warpPerspective(
        image_rgb,
        H_final,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rectified_mask = cv2.warpPerspective(
        (mask_bool.astype(np.uint8) * 255),
        H_final,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rectified_mask_bool = rectified_mask > 0

    mask_area_raw = max(int(mask_area_raw), 1)
    mask_area_px = int(rectified_mask_bool.sum())
    area_ratio = mask_area_px / float(mask_area_raw)

    if mask_area_px == 0:
        theta_after, lambda_after = 90.0, float("inf")
    else:
        theta_after, lambda_after = _moments_from_mask(rectified_mask_bool)

    roll, pitch, yaw = _matrix_to_euler(rotation)

    params: Dict[str, object] = {
        "type": "homography_plane",
        "K_assumed": K.tolist(),
        "plane_normal_unit": normal_unit.tolist(),
        "rotation_matrix": rotation.tolist(),
        "rotation_euler_deg": {
            "roll": float(np.degrees(roll)),
            "pitch": float(np.degrees(pitch)),
            "yaw": float(np.degrees(yaw)),
        },
        "homography": H_final.tolist(),
        "output_width": out_w,
        "output_height": out_h,
        "theta_after_deg": float(theta_after),
        "lambda_ratio_after": float(lambda_after),
        "mask_area_px": mask_area_px,
        "area_ratio": float(area_ratio),
        "single_warp": True,
    }
    return rectified_rgb, rectified_mask_bool, params



def save_rectified_image(rectified_rgb: np.ndarray, rectified_mask: np.ndarray, output_path: Path) -> None:
    alpha = (rectified_mask.astype(np.uint8) * 255)
    rgba = np.dstack([rectified_rgb, alpha])
    Image.fromarray(rgba, mode="RGBA").save(output_path)


affine_rectify = rectify_plane_homography


__all__ = [
    "load_image_rgba",
    "mask_overlay_image",
    "refine_mask",
    "rectify_plane_homography",
    "affine_rectify",
    "save_rectified_image",
    "RefinementParams",
]
