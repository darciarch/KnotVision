"""Utilities for loading DPT models and managing letterbox transforms."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


@dataclass(frozen=True)
class LetterboxMeta:
    target: int
    scale: float
    pad_top: int
    pad_left: int
    new_width: int
    new_height: int
    orig_width: int
    orig_height: int

    def to_dict(self) -> Dict[str, int | float]:
        return {
            "target": self.target,
            "scale": float(self.scale),
            "pad_top": int(self.pad_top),
            "pad_left": int(self.pad_left),
            "new_width": int(self.new_width),
            "new_height": int(self.new_height),
            "orig_width": int(self.orig_width),
            "orig_height": int(self.orig_height),
        }


def _compute_letterbox_meta(width: int, height: int, target_size: int) -> LetterboxMeta:
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    scale = min(target_size / width, target_size / height)
    scaled_width = int(round(width * scale))
    scaled_height = int(round(height * scale))
    pad_left = (target_size - scaled_width) // 2
    pad_top = (target_size - scaled_height) // 2
    return LetterboxMeta(
        target=target_size,
        scale=scale,
        pad_top=pad_top,
        pad_left=pad_left,
        new_width=scaled_width,
        new_height=scaled_height,
        orig_width=width,
        orig_height=height,
    )


def resolve_model_path(preferred: Path, fallback: Optional[Path]) -> Tuple[Path, str]:
    """Pick first valid model between preferred and fallback."""

    def _is_valid(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 1_000_000

    if _is_valid(preferred):
        return preferred, "DPT_Large"
    if fallback and _is_valid(fallback):
        return fallback, "DPT_Hybrid"
    raise FileNotFoundError("No valid DPT model checkpoint found.")


def _load_architecture(model_tag: str) -> torch.nn.Module:
    repo = "intel-isl/MiDaS"
    if model_tag == "DPT_Large":
        model = torch.hub.load(repo, "DPT_Large", pretrained=False, trust_repo=True)
    elif model_tag == "DPT_Hybrid":
        model = torch.hub.load(repo, "DPT_Hybrid", pretrained=False, trust_repo=True)
    else:
        raise ValueError(f"Unsupported DPT tag: {model_tag}")
    return model


def load_model(weight_path: Path, model_tag: str, device: torch.device) -> torch.nn.Module:
    """Load DPT architecture and apply weights from checkpoint."""

    model = _load_architecture(model_tag)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def letterbox_image(image: Image.Image, target_size: int, fill_value: int = 128) -> Tuple[np.ndarray, LetterboxMeta]:
    """Letterbox the image to a square target while preserving aspect ratio."""

    width, height = image.size
    meta = _compute_letterbox_meta(width, height, target_size)

    resized = image.resize((meta.new_width, meta.new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (meta.target, meta.target), (fill_value,) * 3)
    canvas.paste(resized.convert("RGB"), (meta.pad_left, meta.pad_top))
    return np.array(canvas, dtype=np.uint8), meta


def letterbox_mask(mask_bool: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Apply letterbox transform to a boolean mask."""

    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    resized = cv2.resize(mask_uint8, (meta.new_width, meta.new_height), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((meta.target, meta.target), dtype=np.uint8)
    canvas[meta.pad_top : meta.pad_top + meta.new_height, meta.pad_left : meta.pad_left + meta.new_width] = resized
    return canvas > 0


def preprocess_letterbox(letterboxed_rgb: np.ndarray) -> torch.Tensor:
    """Convert letterboxed RGB array into normalized tensor."""

    if letterboxed_rgb.dtype != np.float32:
        img = letterboxed_rgb.astype(np.float32) / 255.0
    else:
        img = letterboxed_rgb
    img = (img - _MEAN) / _STD
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor


def run_depth(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Run model inference and resize output back to letterbox size."""

    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
    depth = prediction.squeeze().cpu().numpy().astype(np.float32)
    return depth


def undo_letterbox_depth(depth_map: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Resize depth map back to original image dimensions using stored letterbox metadata."""

    cropped = depth_map[
        meta.pad_top : meta.pad_top + meta.new_height,
        meta.pad_left : meta.pad_left + meta.new_width,
    ]
    if cropped.shape != (meta.new_height, meta.new_width):
        raise ValueError("Depth map dimensions do not match letterbox metadata")

    tensor = torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0)
    restored = F.interpolate(
        tensor,
        size=(meta.orig_height, meta.orig_width),
        mode="bilinear",
        align_corners=False,
    )
    return restored.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def undo_letterbox_mask(mask_letterbox: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Resize a letterboxed mask back to the raw resolution with nearest interpolation."""

    if mask_letterbox.ndim != 2:
        raise ValueError("Mask must be 2D for unletterbox")

    cropped = mask_letterbox[
        meta.pad_top : meta.pad_top + meta.new_height,
        meta.pad_left : meta.pad_left + meta.new_width,
    ]
    if cropped.shape != (meta.new_height, meta.new_width):
        raise ValueError("Mask dimensions do not match letterbox metadata")

    resized = cv2.resize(
        cropped.astype(np.uint8),
        (meta.orig_width, meta.orig_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def letterbox_depth(depth_raw: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Apply letterbox transform to a depth map for validation."""

    resized = cv2.resize(depth_raw, (meta.new_width, meta.new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((meta.target, meta.target), dtype=np.float32)
    canvas[meta.pad_top : meta.pad_top + meta.new_height, meta.pad_left : meta.pad_left + meta.new_width] = resized
    return canvas


def inverse_consistency_error(
    depth_letterbox: np.ndarray,
    depth_raw: np.ndarray,
    meta: LetterboxMeta,
) -> float:
    """Return normalized MSE after a letterbox→raw→letterbox round trip."""

    relb = letterbox_depth(depth_raw, meta)
    valid = np.zeros_like(depth_letterbox, dtype=bool)
    valid[
        meta.pad_top : meta.pad_top + meta.new_height,
        meta.pad_left : meta.pad_left + meta.new_width,
    ] = True
    diff = relb[valid] - depth_letterbox[valid]
    mse = float(np.mean(diff**2))
    span = float(np.nanmax(depth_letterbox[valid]) - np.nanmin(depth_letterbox[valid]))
    norm = max(span * span, 1.0)
    return mse / norm


def assert_inverse_consistency(
    depth_letterbox: np.ndarray,
    depth_raw: np.ndarray,
    meta: LetterboxMeta,
    tol: float = 1e-3,
) -> None:
    """Validate letterbox/unletterbox stability, raising on failure."""

    norm_mse = inverse_consistency_error(depth_letterbox, depth_raw, meta)
    if not np.isfinite(norm_mse) or norm_mse > tol:
        raise AssertionError(
            f"Letterbox/unletterbox inconsistency detected (norm_mse={norm_mse:.6f}, tol={tol})."
        )


__all__ = [
    "LetterboxMeta",
    "resolve_model_path",
    "load_model",
    "letterbox_image",
    "letterbox_mask",
    "preprocess_letterbox",
    "run_depth",
    "undo_letterbox_depth",
    "undo_letterbox_mask",
    "letterbox_depth",
    "assert_inverse_consistency",
    "inverse_consistency_error",
]
