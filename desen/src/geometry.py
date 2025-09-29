from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TargetDimensions:
    width_px: int
    height_px: int


def compute_target_dimensions(width_cm: float, height_cm: float, ppi: int) -> TargetDimensions:
    if ppi <= 0:
        raise ValueError("ppi must be positive")
    width_px = int(round(width_cm / 2.54 * ppi))
    height_px = int(round(height_cm / 2.54 * ppi))
    if width_px <= 0 or height_px <= 0:
        raise ValueError("Target dimensions must be positive")
    return TargetDimensions(width_px=width_px, height_px=height_px)


def resize_to_dimensions(image: np.ndarray, dims: TargetDimensions) -> np.ndarray:
    if image.shape[1] == dims.width_px and image.shape[0] == dims.height_px:
        return image
    return cv2.resize(image, (dims.width_px, dims.height_px), interpolation=cv2.INTER_CUBIC)


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)
    means = image_f.reshape(-1, 3).mean(axis=0)
    scale = means.mean() / (means + 1e-6)
    balanced = np.clip(image_f * scale, 0.0, 255.0)
    return balanced.astype(np.uint8)


def apply_photometric_normalization(image: np.ndarray, clahe_clip: float = 2.0, clahe_grid: int = 8) -> np.ndarray:
    balanced = gray_world_white_balance(image)
    denoised = cv2.bilateralFilter(balanced, d=7, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    normalized = cv2.cvtColor(merged, cv2.COLOR_Lab2RGB)
    return normalized
