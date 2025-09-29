from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import cv2
import numpy as np
from PIL import Image


QuantizerName = Literal["kmeans", "mediancut"]


@dataclass
class ColorQuantizationResult:
    labels: np.ndarray  # shape (H, W), ints
    centers_lab: np.ndarray  # shape (K, 3), cv2 Lab space
    centers_rgb: np.ndarray  # shape (K, 3), uint8
    method: QuantizerName


def quantize_colors(
    image_rgb: np.ndarray,
    k: int,
    method: QuantizerName = "kmeans",
    attempts: int = 3,
    max_samples: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ColorQuantizationResult:
    if method == "kmeans":
        return _quantize_kmeans(image_rgb, k, attempts, max_samples=max_samples, progress=progress)
    if method == "mediancut":
        return _quantize_mediancut(image_rgb, k)
    raise ValueError(f"Unsupported quantizer '{method}'")


def _quantize_kmeans(
    image_rgb: np.ndarray,
    k: int,
    attempts: int,
    max_samples: Optional[int],
    progress: Optional[Callable[[int, int], None]],
) -> ColorQuantizationResult:
    lab_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    pixels = lab_img.reshape((-1, 3)).astype(np.float32)
    total_pixels = pixels.shape[0]

    sample_pixels = pixels
    if max_samples is not None and total_pixels > max(max_samples, k):
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(total_pixels, size=max(max_samples, k), replace=False)
        sample_pixels = pixels[sample_indices]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, _, centers = cv2.kmeans(
        sample_pixels,
        k,
        None,
        criteria,
        max(attempts, 1),
        cv2.KMEANS_PP_CENTERS,
    )
    centers_lab = np.clip(centers.astype(np.float32), 0, 255)
    centers_rgb = _lab_centers_to_rgb(centers_lab)

    labels_flat = _assign_pixels_to_centers(pixels, centers_lab, progress)
    labels_2d = labels_flat.reshape(image_rgb.shape[:2]).astype(np.int32)
    return ColorQuantizationResult(labels=labels_2d, centers_lab=centers_lab, centers_rgb=centers_rgb, method="kmeans")


def _quantize_mediancut(image_rgb: np.ndarray, k: int) -> ColorQuantizationResult:
    method = getattr(getattr(Image, "Quantize", Image), "MEDIANCUT", getattr(Image, "MEDIANCUT", 0))
    pil_img = Image.fromarray(image_rgb, mode="RGB")
    quantized = pil_img.quantize(colors=k, method=method, dither=Image.Dither.NONE)
    labels = np.array(quantized, dtype=np.int32)

    palette = quantized.getpalette()
    if not palette:
        raise ValueError("Median-cut quantization produced no palette")
    palette_array = np.array(palette, dtype=np.uint8).reshape(-1, 3)
    used_indices = np.unique(labels)
    centers_rgb = palette_array[used_indices]
    index_map = {original: new for new, original in enumerate(used_indices)}

    remapped_labels = np.vectorize(index_map.get)(labels).astype(np.int32)
    centers_lab = _rgb_colors_to_lab(centers_rgb)
    return ColorQuantizationResult(labels=remapped_labels, centers_lab=centers_lab, centers_rgb=centers_rgb, method="mediancut")


def _lab_centers_to_rgb(centers_lab: np.ndarray) -> np.ndarray:
    lab_reshaped = centers_lab.reshape((-1, 1, 3)).astype(np.uint8)
    rgb = cv2.cvtColor(lab_reshaped, cv2.COLOR_Lab2RGB).reshape((-1, 3))
    return rgb.astype(np.uint8)


def _rgb_colors_to_lab(colors_rgb: np.ndarray) -> np.ndarray:
    rgb = colors_rgb.reshape((-1, 1, 3)).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape((-1, 3)).astype(np.float32)
    return lab


def _assign_pixels_to_centers(
    pixels: np.ndarray,
    centers_lab: np.ndarray,
    progress: Optional[Callable[[int, int], None]],
    chunk_size: int = 750_000,
) -> np.ndarray:
    total = pixels.shape[0]
    labels = np.empty(total, dtype=np.int32)
    chunk_size = max(10_000, min(chunk_size, total))
    processed = 0
    while processed < total:
        end = min(processed + chunk_size, total)
        chunk = pixels[processed:end]
        diff = chunk[:, None, :] - centers_lab[None, :, :]
        dist = np.sum(diff * diff, axis=2)
        labels[processed:end] = np.argmin(dist, axis=1)
        processed = end
        if progress is not None:
            progress(processed, total)
    return labels
