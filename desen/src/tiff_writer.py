from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import tifffile
from PIL import Image

Color = Tuple[int, int, int]


def write_indexed_image(
    output_path: Path,
    index_map: np.ndarray,
    palette: Sequence[Color],
    ppi: int,
    fmt: str = "tiff",
    compression: str | None = None,
) -> None:
    if index_map.dtype != np.uint8:
        raise ValueError("index_map must be uint8 for indexed outputs")
    if len(palette) == 0:
        raise ValueError("Palette must contain at least one color")
    if len(palette) > 256:
        raise ValueError("Indexed palette cannot exceed 256 colors")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "tiff":
        _write_tiff(output_path, index_map, palette, ppi, compression)
    elif fmt == "bmp":
        _write_bmp(output_path, index_map, palette, ppi)
    else:
        raise ValueError(f"Unsupported output format '{fmt}'")


def _write_tiff(
    output_path: Path,
    index_map: np.ndarray,
    palette: Sequence[Color],
    ppi: int,
    compression: str | None,
) -> None:
    colormap = _build_tiff_colormap(palette)
    tifffile.imwrite(
        output_path,
        index_map,
        photometric="palette",
        colormap=colormap,
        compression=(compression or "none").lower(),
        resolution=(float(ppi), float(ppi)),
        resolutionunit="INCH",
    )


def _write_bmp(output_path: Path, index_map: np.ndarray, palette: Sequence[Color], ppi: int) -> None:
    image = Image.fromarray(index_map, mode="P")
    flat_palette: list[int] = []
    for color in palette:
        flat_palette.extend(color)
    if len(flat_palette) < 256 * 3:
        flat_palette.extend([0] * (256 * 3 - len(flat_palette)))
    image.putpalette(flat_palette)
    image.save(output_path, format="BMP", dpi=(float(ppi), float(ppi)))


def write_color_masks(masks_dir: Path, index_map: np.ndarray, palette_codes: Sequence[int]) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    for idx, code in enumerate(palette_codes):
        mask = (index_map == idx).astype(np.uint8) * 255
        tifffile.imwrite(masks_dir / f"yarn_{code}.tif", mask, photometric="minisblack")


def _build_tiff_colormap(palette: Sequence[Color]) -> np.ndarray:
    colormap = np.zeros((3, 256), dtype=np.uint16)
    for idx, (r, g, b) in enumerate(palette):
        colormap[0, idx] = int(r * 257)
        colormap[1, idx] = int(g * 257)
        colormap[2, idx] = int(b * 257)
    return colormap
