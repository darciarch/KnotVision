"""Input normalization stage for the DRC pipeline."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image, ImageOps


INPUT_DIR = Path("input")
DEBUG_DIR = Path("debug")
DEBUG_IMAGE = DEBUG_DIR / "input_norm.jpg"
DEBUG_METADATA = DEBUG_DIR / "input_metadata.json"
MASK_DIR = Path("mask")
TARGET_LONG_EDGE = 1024
SUPPORTED_FILENAMES = ("input.jpeg", "input.jpg")
RESAMPLING_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


@dataclass(frozen=True)
class NormalizedInput:
    """Metadata describing the normalized working copy of the input image."""

    input_path: Path
    original_size: Tuple[int, int]
    working_size: Tuple[int, int]
    target_long_edge: int
    debug_image: Path
    source_orientation: Optional[int]
    auto_rotation_deg: int
    manual_rotation_deg: int
    applied_rotation_deg: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "input_path": str(self.input_path),
            "original_size": {
                "width": self.original_size[0],
                "height": self.original_size[1],
            },
            "working_size": {
                "width": self.working_size[0],
                "height": self.working_size[1],
            },
            "target_long_edge": self.target_long_edge,
            "debug_image": str(self.debug_image),
            "source_orientation": self.source_orientation,
            "auto_rotation_deg": self.auto_rotation_deg,
            "manual_rotation_deg": self.manual_rotation_deg,
            "applied_rotation_deg": self.applied_rotation_deg,
        }


def _discover_input_image() -> Path:
    for name in SUPPORTED_FILENAMES:
        path = INPUT_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No input image found. Expected one of: {', '.join(SUPPORTED_FILENAMES)}"
    )


def _scale_dimensions(size: Tuple[int, int], target_long_edge: int) -> Tuple[int, int]:
    width, height = size
    long_edge = max(width, height)
    if long_edge == 0:
        raise ValueError("Invalid image with zero dimension")
    if long_edge == target_long_edge:
        return width, height
    scale = target_long_edge / long_edge
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return new_width, new_height


def _resolve_force_rotation_deg() -> Optional[int]:
    raw = os.getenv("DRC_FORCE_ROTATION_DEG")
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "Environment variable DRC_FORCE_ROTATION_DEG must be an integer"
        ) from exc
    value %= 360
    if value % 90 != 0:
        raise ValueError(
            "Environment variable DRC_FORCE_ROTATION_DEG must be a multiple of 90"
        )
    return value


def _reset_output_dirs() -> None:
    for path in (DEBUG_DIR, MASK_DIR):
        if path.exists():
            shutil.rmtree(path)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)


def normalize_input(force_rotation_deg: Optional[int] = None) -> NormalizedInput:
    _reset_output_dirs()

    if force_rotation_deg is None:
        force_rotation_deg = _resolve_force_rotation_deg()

    image_path = _discover_input_image()
    with Image.open(image_path) as img:
        exif_data = img.getexif()
        source_orientation = exif_data.get(274)
        img = ImageOps.exif_transpose(img)
        auto_rotation_deg = 0
        auto_rotate_enabled = os.getenv("DRC_DISABLE_AUTO_LANDSCAPE", "0").lower() not in {
            "1",
            "true",
            "yes",
        }
        width, height = img.size
        if auto_rotate_enabled and height > width:
            auto_rotation_deg = 90
            img = img.rotate(-auto_rotation_deg, expand=True)
            width, height = img.size
        img = img.convert("RGB")
        manual_rotation_deg = 0
        if force_rotation_deg:
            img = img.rotate(-force_rotation_deg, expand=True)
            manual_rotation_deg = force_rotation_deg % 360
            width, height = img.size
        original_size = (width, height)
        applied_rotation_deg = (auto_rotation_deg + manual_rotation_deg) % 360
        resized_size = _scale_dimensions(original_size, TARGET_LONG_EDGE)
        if resized_size != original_size:
            img_resized = img.resize(resized_size, resample=RESAMPLING_LANCZOS)
        else:
            img_resized = img
        img_resized.save(DEBUG_IMAGE, format="JPEG")

    result = NormalizedInput(
        input_path=image_path,
        original_size=original_size,
        working_size=resized_size,
        target_long_edge=TARGET_LONG_EDGE,
        debug_image=DEBUG_IMAGE,
        source_orientation=source_orientation,
        auto_rotation_deg=auto_rotation_deg,
        manual_rotation_deg=manual_rotation_deg,
        applied_rotation_deg=applied_rotation_deg,
    )
    DEBUG_METADATA.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    info = normalize_input()
    print(json.dumps(info.to_dict(), indent=2))
