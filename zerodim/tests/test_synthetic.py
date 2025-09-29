"""Deterministic synthetic regression test for the pipeline."""
from __future__ import annotations

import numpy as np

from geom import compute_grad_reports, fit_plane_ransac, flatten_with_plane


def _make_synthetic_scene(width: int = 1024, height: int = 768) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    plane = 2.0 * xs + 0.5 * ys + 10.0
    noise = np.random.default_rng(0).normal(scale=0.05, size=plane.shape)
    depth = (plane + noise).astype(np.float32)

    cx, cy = width / 2.0, height / 2.0
    rx, ry = width * 0.35, height * 0.3
    ellipse = (((xs - cx) ** 2) / (rx**2)) + (((ys - cy) ** 2) / (ry**2)) <= 1.0
    mask = ellipse.astype(bool)
    return depth, mask


def run_synthetic_test() -> tuple[bool, str]:
    depth, mask = _make_synthetic_scene()
    thresh = 0.015
    try:
        plane = fit_plane_ransac(depth, mask, threshold=thresh, max_iters=256, seed=123, min_inlier_ratio=0.5)
    except ValueError as exc:  # pragma: no cover - should never happen
        return False, f"Plane fit failed on synthetic scene: {exc}"

    median_value = float(np.nanmedian(depth[mask]))
    flattened = flatten_with_plane(depth, mask, plane, median_value)

    std_before = float(np.nanstd(depth[mask]))
    std_after = float(np.nanstd(flattened[mask]))
    std_ratio = std_after / max(std_before, 1e-6)

    grads_before = compute_grad_reports(depth, mask)
    grads_after = compute_grad_reports(flattened, mask)
    grad_ratio_dx = grads_after["abs_mean_dx"] / max(grads_before["abs_mean_dx"], 1e-6)
    grad_ratio_dy = grads_after["abs_mean_dy"] / max(grads_before["abs_mean_dy"], 1e-6)

    ok = std_ratio <= 0.2 and grad_ratio_dx <= 0.3 and grad_ratio_dy <= 0.3
    report = (
        f"std_ratio={std_ratio:.4f}, grad_ratio_dx={grad_ratio_dx:.4f}, grad_ratio_dy={grad_ratio_dy:.4f}, "
        f"inlier_ratio={plane.inlier_ratio:.4f}, mode={plane.mode_used}"
    )
    return ok, report


if __name__ == "__main__":  # pragma: no cover
    success, details = run_synthetic_test()
    status = "PASS" if success else "FAIL"
    print(f"Synthetic test {status}: {details}")
    raise SystemExit(0 if success else 2)
