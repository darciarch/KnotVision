"""Segmentation stage leveraging SAM to extract a cleaned rug mask."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.segmentation import (
    morphological_geodesic_active_contour,
    inverse_gaussian_gradient,
)
from segment_anything import SamPredictor, sam_model_registry

from src.input_normalize import NormalizedInput


MASK_DIR = Path("mask")
DEBUG_DIR = Path("debug")
OUTPUT_DIR = Path("output")
MASK_RAW_IMAGE = MASK_DIR / "mask_raw.png"
MASK_SAFE_IMAGE = MASK_DIR / "mask.png"
MASK_INNER_IMAGE = MASK_DIR / "mask_inner.png"
MASK_OUTLINE = DEBUG_DIR / "input_mask_outline.jpg"
MASK_INNER_DEBUG_IMAGE = DEBUG_DIR / "input_inner_mask.png"
MASK_METADATA = DEBUG_DIR / "mask_meta.json"
OUTPUT_IMAGE = OUTPUT_DIR / "output.png"
DEFAULT_MODEL_KEY = "vit_l"
MODEL_OPTIONS = {
    "vit_l": {
        "checkpoint": Path("models/sam_vit_l_0b3195.pth"),
        "model_type": "vit_l",
        "label": "ViT-L (default)",
    },
    "vit_h": {
        "checkpoint": Path("models/sam_vit_h_4b8939.pth"),
        "model_type": "vit_h",
        "label": "ViT-H",
    },
}
PROMPT_MARGIN_FRACTION = 0.05
FRINGE_NEGATIVE_POINTS = 5  # per edge
POSITIVE_VERTICAL_FRACTIONS = (0.5, 0.3, 0.7, 0.85)
MIN_SHRINK_PX = 14
MAX_SHRINK_PX = 22
DEFAULT_SHRINK_RATIO = 0.015
ELLIPSE_SHRINK_RATIO = 0.02
SMALL_HOLE_AREA = 2000
SMALL_OBJECT_AREA = 1500
CLOSING_RADIUS = 5
OPENING_RADIUS = 3
EDGE_SMOOTH_SIGMA = 2.0
GUIDED_RADIUS = 8
GUIDED_EPS = 1e-4
LEAK_RATIO_THRESHOLD = 0.15
ELLIPSE_DELTA_THRESHOLD = 0.05
AREA_LOSS_THRESHOLD = 0.4
UNDERCUT_BAND_WIDTH = 30
UNDERCUT_RATIO_THRESHOLD = 0.05
UNDERCUT_STD_THRESHOLD = 2.0
SMOOTHNESS_THRESHOLD = 0.6
OFFSET_STD_THRESHOLD = 2.0
BORDER_MARGIN_ENERGY_THRESHOLD = 20.0
MIN_SAFETY_MARGIN = 28.0
MAX_SAFETY_MARGIN = 36.0
OUTLINE_STROKE_WIDTH = 2
OUTLINE_INSET_RADIUS = 3
OUTLINE_INSIDE_THRESHOLD = 0.995
INNER_MARGIN_FACTOR = 0.90
INNER_MARGIN_MIN = 18
INNER_MARGIN_MAX = 48
BORDER_STRIP_RATIO = 0.5
BORDER_MIN_AREA_RATIO = 0.003
BORDER_MAX_AREA_RATIO = 0.35
BORDER_COLOR_DIFF_THRESHOLD = 4.5
BORDER_KMEANS_CLUSTERS = 3
BORDER_SAMPLE_LIMIT = 60000
RNG = np.random.default_rng(42)
GEODESIC_GUARD_PX = 0
OUTPUT_MARGIN_PX = 10


@dataclass(frozen=True)
class MaskResult:
    """Metadata describing the generated mask."""

    mask_raw_path: Path
    mask_safe_path: Path
    mask_inner_path: Path
    outline_path: Path
    inner_debug_path: Path
    model_key: str
    model_type: str
    checkpoint_path: Path
    bounding_box: Tuple[int, int, int, int]
    area_px: int
    raw_area_px: int
    aspect_ratio: float
    shape_hint: str
    shrink_pixels: int
    ellipse_ratio: float
    ellipse_ratio_raw: float
    ellipse_delta: float
    leak_ratio: float
    undercut_ratio: float
    undercut_std: float
    offset_mean: float
    offset_std: float
    smoothness: float
    outline_inside: float
    review_needed: bool
    raw_mask_path: Optional[Path]
    border_removed: bool
    safety_margin: float
    object_type: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "mask_raw_path": str(self.mask_raw_path),
            "mask_safe_path": str(self.mask_safe_path),
            "mask_inner_path": str(self.mask_inner_path),
            "outline_path": str(self.outline_path),
            "inner_debug_path": str(self.inner_debug_path),
            "model_key": self.model_key,
            "model_type": self.model_type,
            "checkpoint_path": str(self.checkpoint_path),
            "bounding_box": list(self.bounding_box),
            "area_px": self.area_px,
            "raw_area_px": self.raw_area_px,
            "aspect_ratio": self.aspect_ratio,
            "shape_hint": self.shape_hint,
            "shrink_pixels": self.shrink_pixels,
            "ellipse_ratio": self.ellipse_ratio,
            "ellipse_ratio_raw": self.ellipse_ratio_raw,
            "ellipse_delta": self.ellipse_delta,
            "leak_ratio": self.leak_ratio,
            "undercut_ratio": self.undercut_ratio,
            "undercut_std": self.undercut_std,
            "offset_mean": self.offset_mean,
            "offset_std": self.offset_std,
            "smoothness": self.smoothness,
            "outline_inside": self.outline_inside,
            "review_needed": self.review_needed,
            "raw_mask_path": str(self.raw_mask_path) if self.raw_mask_path else None,
            "border_removed": self.border_removed,
            "safety_margin": self.safety_margin,
            "object_type": self.object_type,
        }


class MaskSegmentationError(RuntimeError):
    """Raised when mask generation fails."""


def prompt_model_selection(default_key: str = DEFAULT_MODEL_KEY) -> str:
    env_choice = os.getenv("DRC_SAM_MODEL")
    if env_choice:
        key = env_choice.strip().lower()
        if key in MODEL_OPTIONS:
            return key
        raise MaskSegmentationError(
            f"Environment specifies DRC_SAM_MODEL={env_choice!r}, but it is not a valid option"
        )

    prompt_lines = ["Select SAM model for segmentation:"]
    for key, info in MODEL_OPTIONS.items():
        suffix = " (default)" if key == default_key else ""
        prompt_lines.append(f"  - {key}: {info['label']}{suffix}")
    prompt_lines.append(f"Enter choice [{'/'.join(MODEL_OPTIONS.keys())}] (default={default_key}): ")
    prompt = "\n".join(prompt_lines)

    try:
        choice = input(prompt)
    except EOFError:
        choice = ""

    choice = choice.strip().lower()
    if not choice:
        choice = default_key

    if choice not in MODEL_OPTIONS:
        raise MaskSegmentationError(f"Invalid SAM model choice: {choice}")
    return choice


def load_predictor(model_key: str) -> Tuple[SamPredictor, str, Path]:
    option = MODEL_OPTIONS[model_key]
    checkpoint = option["checkpoint"]
    if not checkpoint.exists():
        raise MaskSegmentationError(f"SAM checkpoint not found: {checkpoint}")
    model_type = option["model_type"]
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor, model_type, checkpoint


def _build_prompts(image_width: int, image_height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin = int(round(min(image_width, image_height) * PROMPT_MARGIN_FRACTION))
    margin = max(margin, 10)
    x0, y0 = margin, margin
    x1, y1 = image_width - margin - 1, image_height - margin - 1
    box = np.array([x0, y0, x1, y1], dtype=np.float32)

    positives = []
    cx = image_width / 2.0
    for frac in POSITIVE_VERTICAL_FRACTIONS:
        y = min(image_height - 4.0, max(3.0, image_height * frac))
        positives.append([cx, y])
        positives.append([min(image_width - 5.0, cx + image_width * 0.12), y])
        positives.append([max(5.0, cx - image_width * 0.12), y])
    positive_points = np.array(positives, dtype=np.float32)

    xs_top = np.linspace(x0, x1, num=max(2, FRINGE_NEGATIVE_POINTS), dtype=np.float32)
    xs_bottom = np.linspace(x0, x1, num=max(2, FRINGE_NEGATIVE_POINTS), dtype=np.float32)
    ys_left = np.linspace(y0, y1, num=max(2, FRINGE_NEGATIVE_POINTS), dtype=np.float32)
    ys_right = np.linspace(y0, y1, num=max(2, FRINGE_NEGATIVE_POINTS), dtype=np.float32)

    top_y = max(2.0, y0 * 0.6)
    bottom_y = min(image_height - 3.0, y1 + margin * 0.4)
    left_x = max(2.0, x0 * 0.6)
    right_x = min(image_width - 3.0, x1 + margin * 0.4)

    negatives = (
        [[x, top_y] for x in xs_top]
        + [[x, bottom_y] for x in xs_bottom]
        + [[left_x, y] for y in ys_left]
        + [[right_x, y] for y in ys_right]
        + [[cx, top_y + (bottom_y - top_y) * 0.25]]
        + [[cx, bottom_y - (bottom_y - top_y) * 0.25]]
        + [[left_x + (cx - left_x) * 0.35, top_y]]
        + [[right_x - (right_x - cx) * 0.35, bottom_y]]
    )
    negative_points = np.array(negatives, dtype=np.float32)

    point_coords = np.vstack([positive_points, negative_points])
    point_labels = np.concatenate(
        [np.ones(len(positive_points)), np.zeros(len(negative_points))]
    ).astype(np.int32)
    return box, point_coords, point_labels


def _select_best_mask(masks: np.ndarray) -> np.ndarray:
    if masks.ndim != 3:
        raise MaskSegmentationError("Unexpected mask tensor shape")

    best_mask = None
    best_score: Tuple[float, float, float] | None = None

    for mask in masks:
        mask_bool = mask.astype(bool)
        area = int(mask_bool.sum())
        if area == 0:
            continue
        perimeter = measure.perimeter(mask_bool, neighborhood=8)
        smoothness = 0.0 if perimeter == 0 else 4.0 * np.pi * area / (perimeter**2)

        height, width = mask_bool.shape
        if height == 0 or width == 0:
            continue
        border_margin = int(round(min(height, width) * 0.08))
        border_margin = max(border_margin, 6)
        border_region = np.zeros_like(mask_bool, dtype=bool)
        border_region[:border_margin, :] = True
        border_region[-border_margin:, :] = True
        border_region[:, :border_margin] = True
        border_region[:, -border_margin:] = True
        border_overlap = (mask_bool & border_region).sum() / max(1, area)

        area_ratio = area / float(height * width)
        score = (smoothness, -border_overlap, area_ratio)
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask_bool

    if best_mask is None:
        raise MaskSegmentationError("SAM did not produce a usable mask")

    return best_mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled = measure.label(mask, connectivity=1)
    if labeled.max() == 0:
        return mask
    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    return labeled == largest.label


def _apply_shrink(mask: np.ndarray, shrink_pixels: int) -> np.ndarray:
    if shrink_pixels <= 0:
        return mask
    distance_map = ndi.distance_transform_edt(mask)
    shrunk = distance_map >= shrink_pixels
    if shrunk.sum() == 0:
        return mask
    return shrunk


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    return morphology.remove_small_holes(mask, area_threshold=max_area)


def _smooth_edges(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    blurred = cv2.GaussianBlur(mask_uint8, ksize=(0, 0), sigmaX=EDGE_SMOOTH_SIGMA)
    smoothed = blurred > 160
    return smoothed & mask


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    cleaned = morphology.remove_small_objects(mask, min_size=min_area)
    return cleaned.astype(bool)


def _adaptive_inner_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    distance_map = ndi.distance_transform_edt(mask)
    smoothed = cv2.GaussianBlur(distance_map.astype(np.float32), (0, 0), sigmaX=2.0)
    dynamic_threshold = np.clip(
        smoothed * 0.12 + MIN_SHRINK_PX,
        MIN_SHRINK_PX,
        MAX_SHRINK_PX,
    )
    adaptive_inner = distance_map >= dynamic_threshold
    adaptive_inner = morphology.binary_closing(adaptive_inner, morphology.disk(2))
    adaptive_inner = adaptive_inner.astype(bool)
    return adaptive_inner, dynamic_threshold


def _geodesic_dilate(mask: np.ndarray, limit: np.ndarray, radius: int, iterations: int) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    structure = morphology.disk(radius)
    result = mask.astype(bool)
    for _ in range(iterations):
        dilated = morphology.binary_dilation(result, structure)
        result = np.logical_and(dilated, limit)
    return result


def _strip_border(
    mask: np.ndarray, image_rgb: np.ndarray, shrink_pixels: int
) -> Tuple[np.ndarray, bool, np.ndarray]:
    if mask.sum() == 0:
        border_mask = np.zeros_like(mask, dtype=bool)
        return mask, False, border_mask

    distance_map = ndi.distance_transform_edt(mask)
    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    cluster_inner, cluster_removed, cluster_border = _strip_border_via_clusters(
        mask, lab_image, distance_map, shrink_pixels
    )
    if cluster_removed and cluster_inner.sum() > 0:
        border_mask = cluster_border
        return cluster_inner.astype(bool), True, border_mask

    reduce_px = max(3, int(round(shrink_pixels * BORDER_STRIP_RATIO)))
    adaptive_threshold = shrink_pixels + max(2, int(round(reduce_px * 0.75)))
    adaptive_inner = distance_map >= adaptive_threshold
    if adaptive_inner.sum() == 0:
        adaptive_inner = distance_map >= shrink_pixels
    if adaptive_inner.sum() == 0:
        border_mask = np.zeros_like(mask, dtype=bool)
        return mask, False, border_mask

    removed_ratio = (mask.sum() - adaptive_inner.sum()) / mask.sum()
    if removed_ratio < BORDER_MIN_AREA_RATIO:
        border_mask = np.zeros_like(mask, dtype=bool)
        return mask, False, border_mask

    border_mask = mask & ~adaptive_inner
    return adaptive_inner.astype(bool), True, border_mask


def _strip_border_via_clusters(
    mask: np.ndarray,
    lab_image: np.ndarray,
    distance_map: np.ndarray,
    shrink_pixels: int,
) -> Tuple[np.ndarray, bool, np.ndarray]:
    count = int(mask.sum())
    if count < 500:
        border_mask = np.zeros_like(mask, dtype=bool)
        return mask, False, border_mask

    lab_pixels = lab_image[mask]
    dist_vals = distance_map[mask]
    features = np.concatenate(
        [
            (lab_pixels[:, :3].astype(np.float32) / 10.0),
            (dist_vals[:, None].astype(np.float32) * 2.0),
        ],
        axis=1,
    )

    total = features.shape[0]
    sample_limit = min(BORDER_SAMPLE_LIMIT, total)
    if total > sample_limit:
        sample_idx = RNG.choice(total, sample_limit, replace=False)
        sample_features = features[sample_idx]
    else:
        sample_idx = None
        sample_features = features

    k = min(BORDER_KMEANS_CLUSTERS, sample_features.shape[0])
    if k < 2:
        border_mask = np.zeros_like(mask, dtype=bool)
        return mask, False, border_mask

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        60,
        0.1,
    )
    _, sample_labels, centers = cv2.kmeans(
        sample_features.astype(np.float32),
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    if sample_idx is None:
        labels_all = sample_labels.flatten()
    else:
        labels_all = np.zeros(total, dtype=np.int32)
        labels_all[sample_idx] = sample_labels.flatten()
        remaining_idx = np.setdiff1d(np.arange(total), sample_idx, assume_unique=True)
        if remaining_idx.size:
            remaining_features = features[remaining_idx].astype(np.float32)
            dists = np.linalg.norm(
                remaining_features[:, None, :] - centers[None, :, :], axis=2
            )
            labels_all[remaining_idx] = np.argmin(dists, axis=1)

    border_threshold = max(1.5, shrink_pixels * 0.25)
    cluster_info = {}
    for idx in range(k):
        mask_idx = labels_all == idx
        if not np.any(mask_idx):
            continue
        area = int(mask_idx.sum())
        area_ratio = area / count
        mean_dist = float(dist_vals[mask_idx].mean())
        mean_color = lab_pixels[mask_idx].mean(axis=0)
        cluster_info[idx] = {
            "area_ratio": area_ratio,
            "mean_dist": mean_dist,
            "mean_color": mean_color,
        }

    if not cluster_info:
        border_mask = np.zeros_like(mask, dtype=bool)
        return mask, False, border_mask

    inner_idx = max(cluster_info, key=lambda cid: cluster_info[cid]["mean_dist"])
    inner_color = cluster_info[inner_idx]["mean_color"]

    border_indices = [
        cid
        for cid, info in cluster_info.items()
        if cid != inner_idx
        and info["mean_dist"] <= border_threshold
        and BORDER_MIN_AREA_RATIO <= info["area_ratio"] <= BORDER_MAX_AREA_RATIO
    ]

    if not border_indices:
        border_mask = np.zeros_like(mask, dtype=bool)
        return mask, False, border_mask

    border_mask_flat = np.isin(labels_all, border_indices)
    if not np.any(border_mask_flat):
        border_mask = np.zeros_like(mask, dtype=bool)
        return mask, False, border_mask

    border_mask = np.zeros_like(mask, dtype=bool)
    border_mask[mask] = border_mask_flat

    border_pixels = lab_pixels[border_mask_flat]
    border_color = border_pixels.mean(axis=0)
    color_diff = float(np.linalg.norm(border_color - inner_color))
    if color_diff < BORDER_COLOR_DIFF_THRESHOLD:
        border_mask = np.zeros_like(mask, dtype=bool)
        return mask, False, border_mask

    inner_mask = mask.copy()
    inner_mask[border_mask] = False
    inner_mask = _largest_component(inner_mask)
    if inner_mask.sum() == 0:
        border_mask = np.zeros_like(mask, dtype=bool)
        return mask, False, border_mask

    border_mask = mask & ~inner_mask
    return inner_mask.astype(bool), True, border_mask


def _guided_filter_refine(mask: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask

    guidance = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask_float = mask.astype(np.float32)

    radius = GUIDED_RADIUS * 2 + 1
    eps = GUIDED_EPS

    mean_I = cv2.boxFilter(guidance, cv2.CV_32F, (radius, radius))
    mean_p = cv2.boxFilter(mask_float, cv2.CV_32F, (radius, radius))
    corr_I = cv2.boxFilter(guidance * guidance, cv2.CV_32F, (radius, radius))
    corr_Ip = cv2.boxFilter(guidance * mask_float, cv2.CV_32F, (radius, radius))

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))

    q = mean_a * guidance + mean_b
    refined = (q > 0.5).astype(bool)
    return refined


def _hires_edge_refine(
    mask: np.ndarray, image_rgb: np.ndarray, limit: np.ndarray
) -> np.ndarray:
    if mask.sum() == 0:
        return mask

    ys, xs = np.nonzero(limit)
    y_min = max(int(ys.min()) - 8, 0)
    y_max = min(int(ys.max()) + 8, image_rgb.shape[0] - 1)
    x_min = max(int(xs.min()) - 8, 0)
    x_max = min(int(xs.max()) + 8, image_rgb.shape[1] - 1)

    roi_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
    roi_limit = limit[y_min : y_max + 1, x_min : x_max + 1]
    roi_image = image_rgb[y_min : y_max + 1, x_min : x_max + 1]

    upscale_shape = (roi_mask.shape[1] * 2, roi_mask.shape[0] * 2)
    roi_mask_up = cv2.resize(
        roi_mask.astype(np.uint8) * 255, upscale_shape, interpolation=cv2.INTER_NEAREST
    )
    roi_limit_up = cv2.resize(
        roi_limit.astype(np.uint8) * 255, upscale_shape, interpolation=cv2.INTER_NEAREST
    )
    roi_image_up = cv2.resize(
        roi_image, upscale_shape, interpolation=cv2.INTER_LINEAR
    )

    gray = cv2.cvtColor(roi_image_up, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edges = inverse_gaussian_gradient(gray, alpha=100.0, sigma=2.0)

    init_ls = roi_mask_up.astype(bool)
    mgac = morphological_geodesic_active_contour(
        edges,
        60,
        init_level_set=init_ls,
        smoothing=4,
        balloon=0,
        threshold=0.4,
    )

    mgac = mgac & (roi_limit_up.astype(bool))
    mgac_down = cv2.resize(mgac.astype(np.uint8) * 255, (roi_mask.shape[1], roi_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    refined = mask.copy()
    refined[y_min : y_max + 1, x_min : x_max + 1] = mgac_down & roi_limit
    return refined


def _shape_hint(area: int, bbox: Tuple[int, int, int, int], ellipse_ratio: float) -> str:
    x_min, y_min, x_max, y_max = bbox
    width = max(1, x_max - x_min + 1)
    height = max(1, y_max - y_min + 1)
    bbox_area = width * height
    fill_ratio = area / bbox_area if bbox_area else 0
    if ellipse_ratio <= 1.5:
        return "oval"
    if fill_ratio >= 0.85:
        return "rectangular"
    return "unknown"


def _compute_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise MaskSegmentationError("Mask is empty after post-processing")
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    return x_min, y_min, x_max, y_max


def _compute_region_ellipse_ratio(mask: np.ndarray) -> float:
    labeled = measure.label(mask, connectivity=1)
    if labeled.max() == 0:
        return 1.0
    region = measure.regionprops(labeled)[0]
    major = float(region.major_axis_length or 1.0)
    minor = float(region.minor_axis_length or 1.0)
    if minor == 0.0:
        return 1.0
    return major / minor


def _compute_shrink_pixels(mask: np.ndarray) -> Tuple[int, float]:
    height, width = mask.shape
    short_edge = min(height, width)
    ellipse_ratio = _compute_region_ellipse_ratio(mask)
    ratio = DEFAULT_SHRINK_RATIO
    if ellipse_ratio >= 1.2:
        ratio = max(DEFAULT_SHRINK_RATIO, ELLIPSE_SHRINK_RATIO)
    shrink = int(round(short_edge * ratio + MIN_SHRINK_PX))
    shrink = max(MIN_SHRINK_PX, min(MAX_SHRINK_PX, shrink))
    return shrink, ellipse_ratio


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask

    mask = _largest_component(mask)

    selem_close = morphology.disk(CLOSING_RADIUS)
    mask = morphology.binary_closing(mask, selem_close)

    mask = _fill_small_holes(mask, SMALL_HOLE_AREA)

    selem_open = morphology.disk(OPENING_RADIUS)
    mask = morphology.binary_opening(mask, selem_open)

    mask = _largest_component(mask)
    mask = _remove_small_components(mask, SMALL_OBJECT_AREA)
    mask = _smooth_edges(mask)
    mask = _fill_small_holes(mask, SMALL_HOLE_AREA)
    mask = _remove_small_components(mask, SMALL_OBJECT_AREA)
    mask = _largest_component(mask)
    return mask.astype(bool)


def _write_mask_image(mask: np.ndarray, path: Path) -> None:
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _outline_mask(mask: np.ndarray, inset: int) -> np.ndarray:
    eroded = morphology.binary_erosion(mask, morphology.disk(inset))
    if eroded.sum() == 0:
        eroded = mask
    border = eroded & ~morphology.binary_erosion(eroded, morphology.disk(1))
    return border & mask


def _load_normalized_canvas(normalized: NormalizedInput) -> Image.Image:
    with Image.open(normalized.debug_image) as orig_img:
        orig_img = ImageOps.exif_transpose(orig_img)
        canvas = orig_img.convert("RGB")
    return canvas


def _compute_outline_inside(
    raw_mask: np.ndarray, safe_mask: np.ndarray, inset: int
) -> float:
    outline = _outline_mask(safe_mask, inset)
    total = int(outline.sum())
    if total == 0:
        return 1.0
    inside = int((outline & raw_mask).sum())
    return float(inside / total)


def _write_outline(
    normalized: NormalizedInput,
    raw_mask: np.ndarray,
    safe_mask: np.ndarray,
    inner_mask: np.ndarray,
    outline_path: Path,
) -> None:
    canvas = _load_normalized_canvas(normalized)
    canvas_arr = np.array(canvas)
    if canvas_arr.shape[0] != safe_mask.shape[0] or canvas_arr.shape[1] != safe_mask.shape[1]:
        raise MaskSegmentationError("Mask dimensions do not match normalized image")
    safe_bool = safe_mask.astype(bool)
    canvas_arr = canvas_arr.copy()
    canvas_arr[~safe_bool] = 0

    canvas_bgr = cv2.cvtColor(canvas_arr, cv2.COLOR_RGB2BGR)

    if inner_mask.shape != safe_mask.shape:
        raise MaskSegmentationError("Inner mask dimensions do not match safe mask")
    inner_bool = inner_mask.astype(bool)
    if inner_bool.any():
        overlay = canvas_bgr.copy()
        fill_color = np.array([30, 90, 255], dtype=np.uint8)
        alpha = 0.45
        overlay[inner_bool] = (
            alpha * fill_color.astype(np.float32)
            + (1.0 - alpha) * overlay[inner_bool].astype(np.float32)
        ).astype(np.uint8)
        canvas_bgr[inner_bool] = overlay[inner_bool]

    raw_uint8 = (raw_mask.astype(np.uint8) * 255)
    raw_contours, _ = cv2.findContours(raw_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if raw_contours:
        cv2.drawContours(canvas_bgr, raw_contours, -1, (255, 255, 0), thickness=2)

    outline = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    outline_path.parent.mkdir(parents=True, exist_ok=True)
    outline.save(outline_path, format="JPEG")


def _write_inner_debug(
    normalized: NormalizedInput, inner_mask: np.ndarray, output_path: Path
) -> None:
    canvas = _load_normalized_canvas(normalized)
    canvas_arr = np.array(canvas)
    if canvas_arr.shape[0] != inner_mask.shape[0] or canvas_arr.shape[1] != inner_mask.shape[1]:
        raise MaskSegmentationError("Mask dimensions do not match normalized image")
    mask_bool = inner_mask.astype(bool)
    overlay = canvas_arr.copy()
    tint = np.array([220, 40, 40], dtype=np.uint8)
    alpha = 0.45
    overlay[mask_bool] = (
        (1.0 - alpha) * overlay[mask_bool].astype(np.float32)
        + alpha * tint.astype(np.float32)
    ).astype(np.uint8)
    result = Image.fromarray(overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG")


def _write_final_output(
    normalized: NormalizedInput,
    mask: np.ndarray,
    output_path: Path,
    margin: int = OUTPUT_MARGIN_PX,
) -> None:
    canvas = _load_normalized_canvas(normalized)
    if canvas.size != (mask.shape[1], mask.shape[0]):
        raise MaskSegmentationError("Mask dimensions do not match normalized image")
    image = canvas.convert("RGBA")
    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image.putalpha(alpha)
    bbox = alpha.getbbox()
    if bbox:
        left = max(0, bbox[0] - margin)
        upper = max(0, bbox[1] - margin)
        right = min(image.width, bbox[2] + margin)
        lower = min(image.height, bbox[3] + margin)
        image = image.crop((left, upper, right, lower))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def _write_metadata(result: MaskResult) -> None:
    payload = json.dumps(result.to_dict(), indent=2)
    MASK_METADATA.write_text(payload, encoding="utf-8")


def _compute_leak_ratio(raw_mask: np.ndarray, final_mask: np.ndarray) -> float:
    leak_region = final_mask & ~raw_mask
    if not leak_region.any():
        return 0.0
    distance_map = ndi.distance_transform_edt(~raw_mask)
    return float(distance_map[leak_region].mean())


def _compute_undercut_metrics(
    distance_map: np.ndarray,
    raw_mask: np.ndarray,
    final_mask: np.ndarray,
    margin: float,
) -> Tuple[float, float]:
    trimmed = raw_mask & ~final_mask
    threshold = margin + 2.0
    viable = raw_mask & (distance_map >= threshold)
    if viable.sum() == 0:
        return 0.0, 0.0
    excess = trimmed & (distance_map >= threshold)
    if excess.sum() == 0:
        return 0.0, 0.0
    ratio = float(excess.sum() / viable.sum())
    std = float((distance_map[excess] - threshold).std()) if excess.sum() else 0.0
    return ratio, std


def _compute_smoothness(mask: np.ndarray) -> float:
    perimeter = measure.perimeter(mask.astype(bool), neighborhood=8)
    area = mask.sum()
    if perimeter == 0 or area == 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter**2))


def _smooth_with_signed_distance(mask: np.ndarray, sigma: float = 5.5) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    mask_uint8 = mask.astype(np.uint8)
    inside_dist = ndi.distance_transform_edt(mask_uint8)
    outside_dist = ndi.distance_transform_edt(1 - mask_uint8)
    signed = inside_dist - outside_dist
    signed_smooth = cv2.GaussianBlur(signed.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    refined = signed_smooth >= 0.0
    return refined


def _ellipse_fit_mask(mask: np.ndarray, scale: float = 0.97) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    labeled = measure.label(mask, connectivity=1)
    region = measure.regionprops(labeled)[0]
    cy, cx = region.centroid
    major = region.major_axis_length * 0.5 * scale
    minor = region.minor_axis_length * 0.5 * scale
    angle = np.deg2rad(region.orientation * -180.0 / np.pi)

    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_shift = xx - cx
    y_shift = yy - cy
    x_rot = x_shift * cos_angle + y_shift * sin_angle
    y_rot = -x_shift * sin_angle + y_shift * cos_angle
    ellipse_mask = (x_rot**2) / (major**2 + 1e-6) + (y_rot**2) / (minor**2 + 1e-6) <= 1.0
    return ellipse_mask


def _compute_offset_stats(
    distance_map: np.ndarray,
    final_mask: np.ndarray,
    margin: float,
    edge_thickness: int = 2,
) -> Tuple[float, float]:
    shell = final_mask & ~morphology.binary_erosion(final_mask, morphology.disk(edge_thickness))
    values = distance_map[shell]
    if values.size == 0:
        values = distance_map[final_mask]
    if values.size == 0:
        return float(margin), 0.0
    deltas = values - margin
    return float(values.mean()), float(np.abs(deltas).std())


def _compute_safety_margin(
    distance_map: np.ndarray,
    percentile: float = 10.0,
    padding: float = 3.0,
) -> float:
    distances = distance_map[distance_map > 0]
    if distances.size == 0:
        return float(MIN_SHRINK_PX)
    base = np.percentile(distances, percentile)
    margin = base + padding
    return float(np.clip(margin, MIN_SAFETY_MARGIN, MAX_SAFETY_MARGIN))


def _adjust_margin_for_fringes(
    image_rgb: np.ndarray,
    distance_map: np.ndarray,
    raw_mask: np.ndarray,
    margin: float,
) -> float:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    L = lab[:, :, 0].astype(np.float32)
    grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    inner_band = raw_mask & (
        (distance_map <= margin + 6)
        & (distance_map >= max(1.0, margin - 6))
    )
    if inner_band.any():
        avg = float(grad_mag[inner_band].mean())
        if avg > BORDER_MARGIN_ENERGY_THRESHOLD:
            margin = min(margin + 2.0, MAX_SAFETY_MARGIN)
    return margin


def _build_safe_inner_shape(
    raw_mask: np.ndarray,
    distance_map: np.ndarray,
    margin: float,
) -> np.ndarray:
    safe = distance_map >= margin
    safe = morphology.binary_closing(safe, morphology.disk(1))
    return safe & raw_mask


def segment_mask(normalized: NormalizedInput, *, force_model_key: Optional[str] = None) -> MaskResult:
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    model_key = force_model_key or prompt_model_selection()
    predictor, model_type, checkpoint = load_predictor(model_key)

    image_bgr = cv2.imread(str(normalized.debug_image))
    if image_bgr is None:
        raise MaskSegmentationError(f"Failed to read normalized image: {normalized.debug_image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    height, width = image_rgb.shape[:2]
    box, point_coords, point_labels = _build_prompts(width, height)

    masks, _, _ = predictor.predict(
        box=box,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    raw_mask = _select_best_mask(masks)
    raw_mask = _largest_component(raw_mask)
    raw_mask = _fill_small_holes(raw_mask, SMALL_HOLE_AREA)
    raw_mask = _remove_small_components(raw_mask, SMALL_OBJECT_AREA)
    raw_mask = _largest_component(raw_mask)
    raw_area = int(raw_mask.sum())

    adaptive_inner, shrink_map = _adaptive_inner_mask(raw_mask)

    shrink_pixels, raw_ellipse_ratio = _compute_shrink_pixels(raw_mask)
    if adaptive_inner.any():
        shrink_pixels = int(
            round(
                np.clip(
                    float(shrink_map[adaptive_inner].mean()),
                    MIN_SHRINK_PX,
                    MAX_SHRINK_PX,
                )
            )
        )

    distance_map = ndi.distance_transform_edt(raw_mask)
    raw_bbox = _compute_bounding_box(raw_mask)
    raw_shape_hint = _shape_hint(raw_area, raw_bbox, raw_ellipse_ratio)
    safety_margin = _compute_safety_margin(distance_map)
    safety_margin = _adjust_margin_for_fringes(image_rgb, distance_map, raw_mask, safety_margin)

    def build_refined_mask(margin: float) -> np.ndarray:
        safe = _build_safe_inner_shape(raw_mask, distance_map, margin)
        if safe.sum() == 0:
            return safe
        mask_local = safe.copy()
        mask_local = morphology.binary_closing(mask_local, morphology.disk(3))
        mask_local = morphology.binary_opening(mask_local, morphology.disk(1))
        mask_local = _fill_small_holes(mask_local, SMALL_HOLE_AREA)
        mask_local = _remove_small_components(mask_local, SMALL_OBJECT_AREA)
        mask_local = _largest_component(mask_local)
        mask_local = mask_local & raw_mask
        mask_local = _smooth_with_signed_distance(mask_local)
        mask_local = mask_local & raw_mask
        return mask_local

    final_mask = None
    metrics = {}
    for attempt in range(2):
        candidate = build_refined_mask(safety_margin)
        if candidate.sum() == 0:
            safety_margin = max(MIN_SHRINK_PX, safety_margin - 2.0)
            continue
        leak_ratio = _compute_leak_ratio(raw_mask, candidate)
        undercut_ratio, undercut_std = _compute_undercut_metrics(
            distance_map, raw_mask, candidate, safety_margin
        )
        smoothness = _compute_smoothness(candidate)
        offset_mean, offset_std = _compute_offset_stats(distance_map, candidate, safety_margin)
        if (
            leak_ratio <= LEAK_RATIO_THRESHOLD
            and undercut_ratio <= UNDERCUT_RATIO_THRESHOLD
            and offset_std <= OFFSET_STD_THRESHOLD
            and smoothness >= SMOOTHNESS_THRESHOLD
        ):
            final_mask = candidate
            metrics = {
                "leak_ratio": leak_ratio,
                "undercut_ratio": undercut_ratio,
                "undercut_std": undercut_std,
                "smoothness": smoothness,
                "offset_mean": offset_mean,
                "offset_std": offset_std,
            }
            break
        safety_margin = min(safety_margin + 2.0, MAX_SHRINK_PX)
        final_mask = candidate
        metrics = {
            "leak_ratio": leak_ratio,
            "undercut_ratio": undercut_ratio,
            "undercut_std": undercut_std,
            "smoothness": smoothness,
            "offset_mean": offset_mean,
            "offset_std": offset_std,
        }

    if final_mask is None or final_mask.sum() == 0:
        raise MaskSegmentationError("Safe inner mask could not be generated")

    final_mask = final_mask & raw_mask

    _write_mask_image(raw_mask, MASK_RAW_IMAGE)
    _write_mask_image(final_mask, MASK_SAFE_IMAGE)

    inner_radius = int(round(max(1.0, safety_margin * INNER_MARGIN_FACTOR)))
    inner_radius = max(INNER_MARGIN_MIN, min(INNER_MARGIN_MAX, inner_radius, int(safety_margin) - 1))
    inner_mask = morphology.binary_erosion(final_mask, morphology.disk(inner_radius))
    if inner_mask.sum() == 0:
        inner_mask = final_mask.copy()
    _write_mask_image(inner_mask, MASK_INNER_IMAGE)

    _write_outline(normalized, raw_mask, final_mask, inner_mask, MASK_OUTLINE)
    _write_inner_debug(normalized, inner_mask, MASK_INNER_DEBUG_IMAGE)
    _write_final_output(normalized, inner_mask, OUTPUT_IMAGE)

    border_mask = raw_mask & ~final_mask

    bbox = _compute_bounding_box(final_mask)
    area = int(final_mask.sum())
    aspect_ratio = (bbox[2] - bbox[0] + 1) / max(1, (bbox[3] - bbox[1] + 1))
    ellipse_ratio = _compute_region_ellipse_ratio(final_mask)
    shape = _shape_hint(area, bbox, ellipse_ratio)
    leak_ratio = metrics.get("leak_ratio", 0.0)
    undercut_ratio = metrics.get("undercut_ratio", 0.0)
    undercut_std = metrics.get("undercut_std", 0.0)
    smoothness = metrics.get("smoothness", 0.0)
    offset_mean = metrics.get("offset_mean", float(safety_margin))
    offset_std = metrics.get("offset_std", 0.0)
    inset = max(1, int(np.ceil(OUTLINE_STROKE_WIDTH / 2) + OUTLINE_INSET_RADIUS))
    outline_inside = _compute_outline_inside(raw_mask, final_mask, inset)

    ellipse_delta = 0.0
    if raw_ellipse_ratio > 0:
        ellipse_delta = abs(ellipse_ratio - raw_ellipse_ratio) / raw_ellipse_ratio
    area_loss = 1.0 - (area / max(1, raw_area))
    border_removed = True
    object_type = raw_shape_hint
    shrink_value = int(round(safety_margin))

    review_needed = bool(
        leak_ratio >= LEAK_RATIO_THRESHOLD
        or ellipse_delta > ELLIPSE_DELTA_THRESHOLD
        or area_loss > AREA_LOSS_THRESHOLD
        or not border_removed
        or undercut_ratio > UNDERCUT_RATIO_THRESHOLD
        or undercut_std > UNDERCUT_STD_THRESHOLD
        or offset_std > OFFSET_STD_THRESHOLD
        or smoothness < SMOOTHNESS_THRESHOLD
        or outline_inside < OUTLINE_INSIDE_THRESHOLD
    )

    result = MaskResult(
        mask_raw_path=MASK_RAW_IMAGE,
        mask_safe_path=MASK_SAFE_IMAGE,
        mask_inner_path=MASK_INNER_IMAGE,
        outline_path=MASK_OUTLINE,
        inner_debug_path=MASK_INNER_DEBUG_IMAGE,
        model_key=model_key,
        model_type=model_type,
        checkpoint_path=checkpoint,
        bounding_box=bbox,
        area_px=area,
        raw_area_px=raw_area,
        aspect_ratio=float(aspect_ratio),
        shape_hint=shape,
        shrink_pixels=shrink_value,
        ellipse_ratio=float(ellipse_ratio),
        ellipse_ratio_raw=float(raw_ellipse_ratio),
        ellipse_delta=float(ellipse_delta),
        leak_ratio=float(leak_ratio),
        undercut_ratio=float(undercut_ratio),
        undercut_std=float(undercut_std),
        offset_mean=float(offset_mean),
        offset_std=float(offset_std),
        smoothness=float(smoothness),
        outline_inside=float(outline_inside),
        review_needed=review_needed,
        raw_mask_path=MASK_RAW_IMAGE,
        border_removed=border_removed,
        safety_margin=float(safety_margin),
        object_type=object_type,
    )
    _write_metadata(result)
    return result


__all__ = ["segment_mask", "MaskResult", "MaskSegmentationError"]
