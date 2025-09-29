"""ZeroDim rug depth flattening pipeline with robust instrumentation."""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import secrets
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
from PIL import Image

from dpt_runner import (
    inverse_consistency_error,
    letterbox_image,
    letterbox_mask,
    load_model,
    preprocess_letterbox,
    resolve_model_path,
    run_depth,
    undo_letterbox_depth,
    undo_letterbox_mask,
)
from geom import (
    compute_grad_reports,
    fit_plane_huber,
    fit_plane_ransac,
    flatten_with_plane,
    erode_mask,
    morphological_closing,
    residual_map,
    shape_moments,
    convex_hull_mask,
)
from rect_utils import (
    RefinementParams,
    load_image_rgba,
    mask_overlay_image,
    rectify_plane_homography,
    refine_mask,
    save_rectified_image,
)

PIPELINE_VERSION = "D1.0"
DOF_STD_RATIO_THRESHOLD = 0.2
DOF_GRAD_RATIO_THRESHOLD = 0.3
DOF_THETA_THRESHOLD = 2.0
DOF_LAMBDA_THRESHOLD = 1.1
DOF_RESIDUAL_STD_RATIO_THRESHOLD = 0.2
RECTIFIED_MASK_RATIO_MIN = 0.85
RECTIFIED_MASK_RATIO_MAX = 1.15
MIN_INLIER_RATIO = 0.6
EXPECTED_OUTPUT_FILES = {
    "depth_raw.npy",
    "depth_flat.npy",
    "depth_flat_vis.png",
    "rectified.png",
}
EXPECTED_DEBUG_FILES = {
    "depth_vis_raw.png",
    "mask_overlay.png",
    "letterbox.json",
    "depth_lbx.npy",
    "mask_raw.npy",
    "mask_fit.npy",
    "residual.npy",
    "residual_vis.png",
    "montage.png",
}

def ensure_synthetic_test_passes() -> None:
    try:
        from tests.test_synthetic import run_synthetic_test
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("Synthetic test module not found; ensure tests/ is present") from exc

    ok, report = run_synthetic_test()
    if not ok:
        print("[zerodim] Synthetic regression test failed:")
        print(report)
        sys.exit(2)
    print("[zerodim] Synthetic regression test passed")


def compute_sha1(path: Path) -> str:
    sha1 = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def align_mask_to_raw(raw_rgb: np.ndarray, mask_image: Image.Image) -> Tuple[np.ndarray, Dict[str, object]]:
    mask_rgba = mask_image.convert("RGBA")
    alpha = np.array(mask_rgba.split()[3], dtype=np.uint8)
    if not np.any(alpha):
        raise ValueError("Mask image does not contain opaque pixels")

    ys, xs = np.nonzero(alpha)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    crop_alpha = alpha[y0 : y1 + 1, x0 : x1 + 1]
    crop_rgb = np.array(mask_rgba)[y0 : y1 + 1, x0 : x1 + 1, :3]

    raw_gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    result = cv2.matchTemplate(raw_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    offset_x = max_loc[0] - x0
    offset_y = max_loc[1] - y0

    mask_full = np.zeros(raw_gray.shape, dtype=bool)
    h, w = mask_full.shape
    trim_h, trim_w = crop_alpha.shape

    y_start = max(0, offset_y)
    x_start = max(0, offset_x)
    y_end = min(h, offset_y + trim_h)
    x_end = min(w, offset_x + trim_w)
    if y_start >= y_end or x_start >= x_end:
        raise ValueError("Aligned mask falls outside raw image bounds")

    src_y0 = y_start - offset_y
    src_x0 = x_start - offset_x
    src_y1 = src_y0 + (y_end - y_start)
    src_x1 = src_x0 + (x_end - x_start)

    mask_full[y_start:y_end, x_start:x_end] = crop_alpha[src_y0:src_y1, src_x0:src_x1] > 0

    meta = {
        "match_score": float(max_val),
        "offset": {"x": int(offset_x), "y": int(offset_y)},
        "trim_bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "mask_area_px": int(mask_full.sum()),
    }
    return mask_full, meta


def refine_mask_offset(letterbox_rgb: np.ndarray, mask_bool: np.ndarray, max_shift: int = 4) -> Tuple[np.ndarray, Dict[str, object]]:
    if max_shift <= 0:
        return mask_bool, {"dx": 0, "dy": 0, "score": 1.0}

    image_gray = cv2.cvtColor(letterbox_rgb, cv2.COLOR_RGB2GRAY)
    edge_image = cv2.Canny(image_gray, 60, 180)
    edge_mask = cv2.Canny((mask_bool.astype(np.uint8) * 255), 60, 180)

    best_score = -np.inf
    best_dx = 0
    best_dy = 0

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = np.zeros_like(edge_mask)
            if dy >= 0:
                src_y_slice = slice(0, edge_mask.shape[0] - dy)
                dst_y_slice = slice(dy, edge_mask.shape[0])
            else:
                src_y_slice = slice(-dy, edge_mask.shape[0])
                dst_y_slice = slice(0, edge_mask.shape[0] + dy)
            if dx >= 0:
                src_x_slice = slice(0, edge_mask.shape[1] - dx)
                dst_x_slice = slice(dx, edge_mask.shape[1])
            else:
                src_x_slice = slice(-dx, edge_mask.shape[1])
                dst_x_slice = slice(0, edge_mask.shape[1] + dx)
            shifted[dst_y_slice, dst_x_slice] = edge_mask[src_y_slice, src_x_slice]
            score = float((edge_image * shifted).sum())
            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy

    aligned_mask = np.zeros_like(mask_bool)
    if best_dy >= 0:
        src_y_slice = slice(0, mask_bool.shape[0] - best_dy)
        dst_y_slice = slice(best_dy, mask_bool.shape[0])
    else:
        src_y_slice = slice(-best_dy, mask_bool.shape[0])
        dst_y_slice = slice(0, mask_bool.shape[0] + best_dy)
    if best_dx >= 0:
        src_x_slice = slice(0, mask_bool.shape[1] - best_dx)
        dst_x_slice = slice(best_dx, mask_bool.shape[1])
    else:
        src_x_slice = slice(-best_dx, mask_bool.shape[1])
        dst_x_slice = slice(0, mask_bool.shape[1] + best_dx)
    aligned_mask[dst_y_slice, dst_x_slice] = mask_bool[src_y_slice, src_x_slice]

    return aligned_mask, {"dx": int(best_dx), "dy": int(best_dy), "score": float(best_score)}


def depth_to_visual(depth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is not None:
        valid = mask & np.isfinite(depth)
    else:
        valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros_like(depth, dtype=np.uint8)
    values = depth[valid]
    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 98))
    if vmax - vmin < 1e-6:
        vis = np.zeros_like(depth, dtype=np.uint8)
        vis[valid] = 128
        return vis
    clipped = np.clip(depth, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    vis = (1.0 - normalized) * 255.0
    vis[~valid] = 0
    return vis.astype(np.uint8)


def residual_to_visual(residual: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_valid = mask & np.isfinite(residual)
    if not np.any(mask_valid):
        return np.zeros_like(residual, dtype=np.uint8)
    values = residual[mask_valid]
    max_abs = float(np.nanpercentile(np.abs(values), 99))
    max_abs = max(max_abs, 1e-6)
    scaled = np.clip((residual / (2.0 * max_abs)) + 0.5, 0.0, 1.0)
    vis = (scaled * 255.0).astype(np.uint8)
    vis[~mask] = 0
    return vis


def create_montage(
    overlay: Image.Image,
    depth_raw_vis: np.ndarray,
    depth_flat_vis: np.ndarray,
    rectified_rgba: np.ndarray,
    run_id: str,
    metrics_summary,
    dest_path: Path,
) -> None:
    def _compose_checkerboard(rgba: np.ndarray, tile: int = 32) -> Image.Image:
        h, w, _ = rgba.shape
        tile = max(8, min(tile, max(8, min(h, w) // 4)))
        yy, xx = np.indices((h, w))
        pattern = ((xx // tile + yy // tile) % 2).astype(np.uint8)
        bg = np.where(pattern[..., None] == 0, 200, 150).astype(np.uint8)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        composed = (alpha * rgba[:, :, :3] + (1.0 - alpha) * bg).astype(np.uint8)
        return Image.fromarray(composed, mode="RGB")

    def to_rgb(image: np.ndarray) -> Image.Image:
        if image.ndim == 2:
            rgb = np.stack([image] * 3, axis=-1)
            return Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        if image.shape[2] == 4:
            return _compose_checkerboard(image)
        return Image.fromarray(image.astype(np.uint8))

    panels = [
        ("Input + Mask", overlay.convert("RGB")),
        ("Depth Raw", to_rgb(depth_raw_vis)),
        ("Depth Flat", to_rgb(depth_flat_vis)),
        ("Rectified", to_rgb(rectified_rgba)),
    ]

    panel_w = max(p.width for _, p in panels)
    panel_h = max(p.height for _, p in panels)
    cols = 2
    rows = 2
    canvas_w = cols * panel_w
    canvas_h = rows * panel_h + 48
    montage = Image.new("RGB", (canvas_w, canvas_h), color=(24, 24, 24))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(montage)

    for idx, (title, panel) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = col * panel_w
        y = row * panel_h
        x_off = x + (panel_w - panel.width) // 2
        y_off = y + (panel_h - panel.height) // 2
        montage.paste(panel, (x_off, y_off))
        draw.rectangle([
            x,
            y,
            x + panel_w - 1,
            y + panel_h - 1,
        ], outline=(200, 200, 200), width=1)
        draw.text((x + 10, y + 10), title, fill=(255, 255, 255))

    footer_y = rows * panel_h + 12
    draw.text((10, footer_y), f"run_id: {run_id}", fill=(255, 255, 0))
    if isinstance(metrics_summary, str):
        summary_lines = metrics_summary.splitlines()
    else:
        summary_lines = list(metrics_summary)
    for idx, line in enumerate(summary_lines):
        draw.text((10, footer_y + 18 * (idx + 1)), line, fill=(255, 255, 0))

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(dest_path)


def enforce_file_contract(output_dir: Path, debug_dir: Path) -> None:
    output_files = {p.name for p in output_dir.iterdir() if p.is_file()}
    debug_files = {p.name for p in debug_dir.iterdir() if p.is_file()}

    extra_outputs = sorted(output_files - EXPECTED_OUTPUT_FILES)
    missing_outputs = sorted(EXPECTED_OUTPUT_FILES - output_files)
    extra_debug = sorted(debug_files - EXPECTED_DEBUG_FILES)
    missing_debug = sorted(EXPECTED_DEBUG_FILES - debug_files)

    if missing_outputs:
        raise RuntimeError(f"Missing expected output files: {missing_outputs}")
    if missing_debug:
        raise RuntimeError(f"Missing expected debug files: {missing_debug}")
    if extra_outputs:
        print(f"[zerodim] Warning: unexpected files in output/: {extra_outputs}")
    if extra_debug:
        print(f"[zerodim] Warning: unexpected files in debug/: {extra_debug}")


def compute_hashes(raw_path: Path, mask_path: Path) -> Dict[str, str]:
    return {
        "raw_image_sha1": compute_sha1(raw_path),
        "mask_image_sha1": compute_sha1(mask_path),
    }


def prepare_directories(base_output: Path, base_debug: Path) -> None:
    for directory in (base_output, base_debug):
        if directory.exists():
            for child in directory.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
                else:
                    for sub in child.rglob("*"):
                        if sub.is_file() or sub.is_symlink():
                            sub.unlink()
                    child.rmdir()
        directory.mkdir(parents=True, exist_ok=True)


def run_pipeline(args: argparse.Namespace) -> None:
    print("[zerodim] DPT_RUNNER_OK", flush=True)
    ensure_synthetic_test_passes()

    run_id = f"{_dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"

    if args.runs_dir:
        base_dir = Path(args.runs_dir) / run_id
        output_dir = base_dir / "output"
        debug_dir = base_dir / "debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("output")
        debug_dir = Path("debug")
        prepare_directories(output_dir, debug_dir)

    raw_path = args.input_raw
    mask_path = args.input_mask

    raw_image = Image.open(raw_path).convert("RGB")
    raw_rgb = np.array(raw_image, dtype=np.uint8)

    mask_image = load_image_rgba(mask_path)
    mask_raw_aligned, alignment_meta = align_mask_to_raw(raw_rgb, mask_image)
    overlay_img = mask_overlay_image(raw_rgb, mask_raw_aligned)
    overlay_img.save(debug_dir / "mask_overlay.png")

    letterbox_rgb, lb_meta = letterbox_image(raw_image, args.target_short)
    mask_letterbox = letterbox_mask(mask_raw_aligned, lb_meta)

    mask_offset_meta = {"dx": 0, "dy": 0, "score": 1.0}
    if args.refine_mask:
        mask_letterbox, mask_offset_meta = refine_mask_offset(letterbox_rgb, mask_letterbox)

    closing_kernel_lbx = max(3, args.erosion_px * 2 + 1) if args.refine_mask else 0
    if closing_kernel_lbx > 0:
        mask_letterbox_closed = morphological_closing(mask_letterbox, closing_kernel_lbx)
    else:
        mask_letterbox_closed = mask_letterbox

    erosion_lbx = args.erosion_px if args.refine_mask else 0
    mask_letterbox_refined, refinement_params = refine_mask(
        mask_letterbox_closed,
        erosion_lbx,
        closing_kernel_lbx,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preferred = Path("models/dpt_large.pt") if args.model == "dpt_large" else Path("models/dpt_hybrid.pt")
    fallback = Path("models/dpt_hybrid.pt") if args.model == "dpt_large" else Path("models/dpt_large.pt")
    model_path, model_name = resolve_model_path(preferred, fallback)
    model = load_model(model_path, model_name, device)

    depth_letterbox = run_depth(
        model,
        preprocess_letterbox(letterbox_rgb),
        device,
        (lb_meta.target, lb_meta.target),
    )
    np.save(debug_dir / "depth_lbx.npy", depth_letterbox.astype(np.float32))

    depth_raw = undo_letterbox_depth(depth_letterbox, lb_meta)
    mask_raw = undo_letterbox_mask(mask_letterbox_refined, lb_meta)

    np.save(output_dir / "depth_raw.npy", depth_raw.astype(np.float32))
    np.save(debug_dir / "mask_raw.npy", mask_raw.astype(np.uint8))

    if depth_raw.shape != mask_raw.shape:
        raise RuntimeError("Depth and mask shapes do not match after unletterbox")

    norm_mse = inverse_consistency_error(depth_letterbox, depth_raw, lb_meta)
    if not np.isfinite(norm_mse) or norm_mse > 1e-3:
        print(
            (
                "[zerodim] Letterbox inverse consistency failed "
                f"(norm_mse={norm_mse:.6f})."
            ),
            flush=True,
        )
        sys.exit(2)

    if not np.any(mask_raw):
        raise RuntimeError("Refined mask is empty after unletterbox")

    closing_kernel_raw = 11
    mask_closed_raw = morphological_closing(mask_raw, closing_kernel_raw)
    erosion_fit = max(5, min(args.erosion_px, 7))
    mask_eroded = erode_mask(mask_closed_raw, erosion_fit)
    if np.any(mask_eroded):
        mask_plane = convex_hull_mask(mask_eroded)
        mask_used_label = f"eroded{erosion_fit}_convex"
    else:
        mask_plane = convex_hull_mask(mask_closed_raw)
        mask_used_label = "closed_convex"
    if not np.any(mask_plane):
        raise RuntimeError("Plane fit mask is empty after morphological operations")

    mask_fit = mask_plane
    mask_metrics = mask_fit
    np.save(debug_dir / "mask_fit.npy", mask_fit.astype(np.uint8))

    mask_area_raw = int(mask_raw.sum())
    mask_area_fit = int(mask_fit.sum())
    mask_area_metrics = mask_area_fit

    depth_vis_raw = depth_to_visual(depth_raw, mask_metrics)
    Image.fromarray(depth_vis_raw).save(debug_dir / "depth_vis_raw.png")

    depth_for_plane = depth_raw.copy()
    finite_mask = np.isfinite(depth_for_plane)
    if not np.any(finite_mask & mask_plane):
        raise RuntimeError("No finite depth values within plane mask")
    median_plane = float(np.nanmedian(depth_for_plane[mask_plane]))
    depth_for_plane[~finite_mask] = median_plane

    depth_values_plane = depth_for_plane[mask_plane]
    p5 = float(np.nanpercentile(depth_values_plane, 5))
    p95 = float(np.nanpercentile(depth_values_plane, 95))
    percentile_span = max(p95 - p5, 1e-3)
    ransac_k = float(np.clip(args.ransac_k, 0.015, 0.02))
    plane_threshold = max(ransac_k * percentile_span, 1e-4)

    plane_fit = fit_plane_ransac(
        depth_for_plane,
        mask_plane,
        plane_threshold,
        max_iters=2048,
        seed=args.seed,
        min_inlier_ratio=MIN_INLIER_RATIO,
    )

    if plane_fit.inlier_ratio < MIN_INLIER_RATIO:
        plane_fit = fit_plane_huber(depth_for_plane, mask_plane, plane_threshold)

    if plane_fit.inlier_ratio < MIN_INLIER_RATIO:
        print(
            (
                "[zerodim] Plane fit inlier ratio below threshold after fallback."
            ),
            flush=True,
        )
        sys.exit(2)

    if args.flatten_mode == "constant":
        depth_flat = depth_raw.copy()
        depth_flat[mask_metrics] = median_plane
    else:
        depth_flat = flatten_with_plane(depth_raw, mask_metrics, plane_fit, median_plane)

    def _mask_stats(array: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        values = array[mask]
        return {
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "std": float(np.nanstd(values)),
        }

    mask_stats_before = _mask_stats(depth_raw, mask_metrics)
    mask_stats_after = _mask_stats(depth_flat, mask_metrics)

    stats_before = compute_grad_reports(depth_raw, mask_metrics)
    stats_after = compute_grad_reports(depth_flat, mask_metrics)

    std_ratio_plane = mask_stats_after["std"] / max(mask_stats_before["std"], 1e-6)
    grad_ratio_dx_plane = stats_after["abs_mean_dx"] / max(stats_before["abs_mean_dx"], 1e-6)
    grad_ratio_dy_plane = stats_after["abs_mean_dy"] / max(stats_before["abs_mean_dy"], 1e-6)

    residual_raw = residual_map(depth_raw, plane_fit)
    residual_std = float(np.nanstd(residual_raw[mask_metrics]))
    residual_std_ratio_plane = residual_std / max(mask_stats_before["std"], 1e-6)
    np.save(debug_dir / "residual.npy", residual_raw.astype(np.float32))
    residual_vis = residual_to_visual(residual_raw, mask_metrics)
    Image.fromarray(residual_vis).save(debug_dir / "residual_vis.png")

    std_pass = std_ratio_plane <= DOF_STD_RATIO_THRESHOLD
    grad_dx_pass = grad_ratio_dx_plane <= DOF_GRAD_RATIO_THRESHOLD
    grad_dy_pass = grad_ratio_dy_plane <= DOF_GRAD_RATIO_THRESHOLD
    residual_pass = residual_std_ratio_plane <= DOF_RESIDUAL_STD_RATIO_THRESHOLD
    plane_only_pass = std_pass and grad_dx_pass and grad_dy_pass and residual_pass

    if not plane_only_pass:
        print("[zerodim] PLANE_ONLY_FAIL", flush=True)
        print(
            (
                "[zerodim] Plane-only DoD failed "
                "(adj. std/grad/residual thresholds unmet)."
            ),
            flush=True,
        )
        sys.exit(2)

    print("[zerodim] PLANE_ONLY_PASS", flush=True)

    np.save(output_dir / "depth_flat.npy", depth_flat.astype(np.float32))
    depth_flat_vis = depth_to_visual(depth_flat, mask_metrics)
    Image.fromarray(depth_flat_vis).save(output_dir / "depth_flat_vis.png")

    if args.rectify != "projective":
        raise ValueError("Only projective rectification is supported")

    rectified_rgb, rectified_mask, rectify_params = rectify_plane_homography(
        raw_rgb,
        mask_metrics,
        plane_fit,
        mask_area_raw,
        refine=False,
    )

    theta_after = abs(rectify_params.get("theta_after_deg", 0.0))
    lambda_after = rectify_params.get("lambda_ratio_after", float("inf"))
    rectified_mask_area = int(rectify_params.get("mask_area_px", int(rectified_mask.sum())))
    mask_rect_ratio = rectified_mask_area / max(mask_area_raw, 1)

    projective_pass = (
        theta_after <= DOF_THETA_THRESHOLD
        and lambda_after <= DOF_LAMBDA_THRESHOLD
        and RECTIFIED_MASK_RATIO_MIN <= mask_rect_ratio <= RECTIFIED_MASK_RATIO_MAX
    )

    if not projective_pass:
        print(
            "[zerodim] Projective DoD failed (rectification tolerances unmet).",
            flush=True,
        )
        sys.exit(2)

    save_rectified_image(rectified_rgb, rectified_mask, output_dir / "rectified.png")
    rectified_rgba = np.dstack([rectified_rgb, rectified_mask.astype(np.uint8) * 255])
    print("[zerodim] RECTIFY_PROJECTIVE_OK", flush=True)

    rectified_size = (
        f"{rectify_params.get('output_width', rectified_rgb.shape[1])}x"
        f"{rectify_params.get('output_height', rectified_rgb.shape[0])}"
    )

    metrics_summary = [
        f"std={std_ratio_plane:.3f} grad_dx={grad_ratio_dx_plane:.3f} grad_dy={grad_ratio_dy_plane:.3f} residual={residual_std_ratio_plane:.3f}",
        f"theta={theta_after:.2f} lambda={lambda_after:.3f} area_ratio={mask_rect_ratio:.3f} size={rectified_size}",
    ]

    create_montage(
        overlay_img,
        depth_vis_raw,
        depth_flat_vis,
        rectified_rgba,
        run_id,
        metrics_summary,
        debug_dir / "montage.png",
    )

    hashes = compute_hashes(raw_path, mask_path)

    letterbox_info = lb_meta.to_dict()
    letterbox_info["inverse_mse"] = float(norm_mse)

    mask_area_meta = alignment_meta.get("mask_area_px", 0)
    mask_area_diff = abs(mask_area_meta - mask_area_raw)
    if mask_area_meta > 0 and mask_area_diff > 0.02 * mask_area_meta:
        raise RuntimeError(
            f"Mask area mismatch (meta={mask_area_meta}, raw={mask_area_raw})."
        )

    moments_before = shape_moments(mask_raw)

    plane_inlier_pass = plane_fit.inlier_ratio >= MIN_INLIER_RATIO
    theta_pass = theta_after <= DOF_THETA_THRESHOLD
    lambda_pass = lambda_after <= DOF_LAMBDA_THRESHOLD
    mask_ratio_pass = RECTIFIED_MASK_RATIO_MIN <= mask_rect_ratio <= RECTIFIED_MASK_RATIO_MAX
    projective_pass = theta_pass and lambda_pass and mask_ratio_pass
    single_warp_pass = bool(rectify_params.get("single_warp", False))
    dod_pass = plane_inlier_pass and plane_only_pass and projective_pass and single_warp_pass

    metrics_summary.append(
        "plane="
        f"{plane_only_pass} inlier={plane_inlier_pass} "
        f"proj={projective_pass} warp={single_warp_pass}"
    )

    metadata: Dict[str, object] = {
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "mask_raw_area": int(mask_area_raw),
        "mask_fit_area": int(mask_area_fit),
        "mask_metrics_area": int(mask_area_metrics),
        "inputs": {
            "raw_image": str(raw_path),
            "mask_image": str(mask_path),
            "model": model_name,
            "hashes": hashes,
        },
        "letterbox": letterbox_info,
        "alignment": alignment_meta,
        "mask_offset": mask_offset_meta,
        "refinement": {
            "enabled": bool(args.refine_mask),
            "closing_kernel": refinement_params.closing_kernel,
            "erosion_kernel": refinement_params.erosion_kernel,
        },
        "interpolation_report": {
            "image_resize": "bilinear",
            "mask_resize": "nearest",
        },
        "fit": {
            "inlier_ratio": plane_fit.inlier_ratio,
            "threshold": plane_threshold,
            "mode_used": plane_fit.mode_used,
            "iterations": plane_fit.iterations,
            "fallback_applied": plane_fit.mode_used == "huber",
        },
        "plane": {
            **plane_fit.to_dict(),
            "threshold": plane_threshold,
            "mask_used": mask_used_label,
            "mask_area_px_metrics": mask_area_metrics,
            "mask_area_px_raw": mask_area_raw,
            "median": median_plane,
            "percentile_span": percentile_span,
            "residual_std": residual_std,
            "residual_std_ratio": residual_std_ratio_plane,
            "flatten_mode_requested": args.flatten_mode,
            "normal_unit": rectify_params.get("plane_normal_unit"),
            "std_ratio_plane": std_ratio_plane,
            "grad_ratio_dx_plane": grad_ratio_dx_plane,
            "grad_ratio_dy_plane": grad_ratio_dy_plane,
        },
        "mask_stats": {
            "before": mask_stats_before,
            "after": mask_stats_after,
        },
        "gradients": {
            "before": stats_before,
            "after": stats_after,
            "ratio_dx_plane": grad_ratio_dx_plane,
            "ratio_dy_plane": grad_ratio_dy_plane,
        },
        "moments": {
            "before": {**moments_before, "space": "raw"},
            "after": {
                "theta_deg": rectify_params.get("theta_after_deg", 0.0),
                "lambda_ratio": rectify_params.get("lambda_ratio_after", float("inf")),
                "space": "rectified",
                "type": rectify_params.get("type", "homography_plane"),
            },
        },
        "rectify": rectify_params,
        "masks": {
            "mask_raw": {"area_px": mask_area_raw, "file": "debug/mask_raw.npy"},
            "mask_fit": {"area_px": mask_area_fit, "label": mask_used_label, "file": "debug/mask_fit.npy"},
            "mask_metrics": {"area_px": mask_area_metrics, "source": "mask_fit"},
        },
        "metrics": {
            "sources": {
                "raw": "output/depth_raw.npy",
                "flat": "output/depth_flat.npy",
            },
            "mask_used": mask_used_label,
            "mask_area_px_raw": mask_area_raw,
            "mask_area_px_metrics": mask_area_metrics,
            "mask_area_px_fit": mask_area_fit,
            "mask_area_px_rectified": rectified_mask_area,
            "mask_area_ratio_rectified": mask_rect_ratio,
            "mask_area_diff_vs_alignment": mask_area_diff,
            "residual_std": residual_std,
            "residual_std_ratio": residual_std_ratio_plane,
        },
        "definition_of_done": {
            "plane_inlier_ratio": plane_fit.inlier_ratio,
            "plane_inlier_pass": plane_inlier_pass,
            "plane_only_pass": plane_only_pass,
            "std_ratio_plane": std_ratio_plane,
            "std_pass": std_pass,
            "grad_ratio_dx_plane": grad_ratio_dx_plane,
            "grad_dx_pass": grad_dx_pass,
            "grad_ratio_dy_plane": grad_ratio_dy_plane,
            "grad_dy_pass": grad_dy_pass,
            "residual_std_ratio_plane": residual_std_ratio_plane,
            "residual_pass": residual_pass,
            "theta_after_abs": theta_after,
            "theta_pass": theta_pass,
            "lambda_ratio_after": lambda_after,
            "lambda_pass": lambda_pass,
            "mask_area_ratio_rectified": mask_rect_ratio,
            "mask_ratio_pass": mask_ratio_pass,
            "projective_pass": projective_pass,
            "single_warp_pass": single_warp_pass,
            "dod_pass": dod_pass,
        },
        "montage_summary": metrics_summary,
    }

    (debug_dir / "letterbox.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    enforce_file_contract(output_dir, debug_dir)

    if not dod_pass:
        print("[zerodim] Definition of Done checks failed. See debug/letterbox.json for details.")
        sys.exit(2)

    print("[zerodim] Pipeline completed successfully")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ZeroDim rug depth flattening pipeline")
    parser.add_argument("--model", choices=["dpt_large", "dpt_hybrid"], default="dpt_large")
    parser.add_argument("--target-short", type=int, default=384)
    parser.add_argument("--input-raw", type=Path, default=Path("input/raw_input.jpg"))
    parser.add_argument("--input-mask", type=Path, default=Path("input/input_data.png"))
    parser.add_argument("--flatten-mode", choices=["remove_plane", "constant"], default="remove_plane")
    parser.add_argument("--rectify", choices=["projective"], default="projective")
    parser.add_argument("--erosion-px", type=int, default=5)
    parser.add_argument("--ransac-k", type=float, default=0.018)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refine-mask", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.flatten_mode == "constant":
        print("[zerodim] flatten_mode=constant forces median depth; overriding user to median.")

    run_pipeline(args)


if __name__ == "__main__":
    main()
