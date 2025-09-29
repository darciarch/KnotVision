from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence
import time

try:
    from . import cleanup, color_reduce, geometry, palette_match, tiff_writer
except ImportError:  # pragma: no cover - executed when run as a loose script
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    import cleanup  # type: ignore
    import color_reduce  # type: ignore
    import geometry  # type: ignore
    import palette_match  # type: ignore
    import tiff_writer  # type: ignore

import cv2
import numpy as np


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carpet photo to indexed yarn map pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Input image path")
    parser.add_argument("--width_cm", required=True, type=float, help="Final carpet width in centimeters")
    parser.add_argument("--height_cm", required=True, type=float, help="Final carpet height in centimeters")
    parser.add_argument("--ppi", required=True, type=int, help="Target pixels per inch")
    parser.add_argument("--k_colors", type=int, default=8, help="Number of clusters for quantization (6..12)")
    parser.add_argument("--quantizer", choices=["kmeans", "mediancut"], default="kmeans", help="Color quantization method")
    parser.add_argument("--palette", type=Path, required=True, help="Palette CSV file")
    parser.add_argument("--palette_order", choices=["csv", "code_asc"], default="code_asc", help="Palette ordering strategy")
    parser.add_argument(
        "--pre_downsample",
        type=float,
        default=1.0,
        help="Scale factor (0<factor<=1) for a faster pre-quantization pass; labels are upscaled back to target size",
    )
    parser.add_argument("--min_region", type=int, default=12, help="Minimum connected region size in pixels")
    parser.add_argument("--morph", nargs="*", choices=["open", "close"], default=[], help="Morphology operations to apply in order")
    parser.add_argument("--deltaE_warn", type=float, default=5.0, help="Warn if ΔE exceeds this threshold")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--format", choices=["tiff", "bmp"], default="tiff", help="Output format")
    parser.add_argument("--compression", choices=["none", "lzw"], default="none", help="TIFF compression (ignored for BMP)")
    parser.add_argument("--skip_normalize", action="store_true", help="Skip photometric normalization stage")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--quant_sample",
        type=int,
        default=1_200_000,
        help="Maximum number of pixels to train K-means on (0 to disable sampling)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    start_time = time.perf_counter()
    estimated_total_holder = {"value": None}  # type: dict[str, Optional[float]]

    if args.k_colors < 6 or args.k_colors > 12:
        print("--k_colors must be between 6 and 12", file=sys.stderr)
        return 1

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        palette_entries = palette_match.load_palette(args.palette)
    except ValueError as exc:
        print(f"Palette error: {exc}", file=sys.stderr)
        return 1

    if not (0.0 < args.pre_downsample <= 1.0):
        print("--pre_downsample must be within (0, 1]", file=sys.stderr)
        return 1

    image_bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"Failed to load image: {args.input}", file=sys.stderr)
        return 1
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if not args.skip_normalize:
        image_rgb = geometry.apply_photometric_normalization(image_rgb)

    dims = geometry.compute_target_dimensions(args.width_cm, args.height_cm, args.ppi)
    image_rgb = geometry.resize_to_dimensions(image_rgb, dims)

    if args.verbose:
        effective_pixels = (dims.width_px * dims.height_px) * (args.pre_downsample ** 2)
        effective_mpix = effective_pixels / 1_000_000.0
        # Empirical baseline ~0.9 seconds per megapixel on recent CPUs.
        estimated_seconds = max(5.0, effective_mpix * 0.9)
        estimated_total_holder["value"] = estimated_seconds
        print(
            "Estimated finishing time: "
            f"~{_format_duration(estimated_seconds)} (effective {effective_mpix:.1f} MP)"
        )

    quant_sample = args.quant_sample if args.quant_sample > 0 else None
    quant_progress_cb: Optional[Callable[[int, int], None]] = None
    if args.verbose:
        quant_progress_cb = _build_quant_progress_callback(start_time, estimated_total_holder)

    quant_image = image_rgb
    if args.pre_downsample < 1.0:
        scale = args.pre_downsample
        quant_width = max(1, int(round(dims.width_px * scale)))
        quant_height = max(1, int(round(dims.height_px * scale)))
        if args.verbose:
            print(
                "Downsampling for quantization to "
                f"{quant_width}x{quant_height} ({scale:.2f}x)"
            )
        quant_image = cv2.resize(image_rgb, (quant_width, quant_height), interpolation=cv2.INTER_AREA)

    quant_result = color_reduce.quantize_colors(
        quant_image,
        args.k_colors,
        method=args.quantizer,
        max_samples=quant_sample,
        progress=quant_progress_cb,
    )
    actual_k = quant_result.centers_lab.shape[0]
    if actual_k != args.k_colors and args.verbose:
        print(f"Quantizer produced {actual_k} clusters (requested {args.k_colors})")
    if estimated_total_holder["value"] is not None:
        estimated_total_holder["value"] = max(
            estimated_total_holder["value"],
            (time.perf_counter() - start_time) * 1.05,
        )
    _log_progress("Quantization", args.verbose, start_time, estimated_total_holder["value"])

    if args.pre_downsample < 1.0:
        labels_upsampled = cv2.resize(
            quant_result.labels.astype(np.int32),
            (dims.width_px, dims.height_px),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
    else:
        labels_upsampled = quant_result.labels

    try:
        match_result = palette_match.match_clusters_to_palette(quant_result.centers_lab, labels_upsampled, palette_entries)
    except ValueError as exc:
        print(f"Palette match error: {exc}", file=sys.stderr)
        return 1
    _log_progress("Palette matching", args.verbose, start_time, estimated_total_holder["value"])

    cleaned_labels = cleanup.apply_cleanup(
        labels_upsampled,
        min_region=args.min_region,
        use_majority_filter=True,
        morph_operations=args.morph,
    )
    _log_progress("Cleanup", args.verbose, start_time, estimated_total_holder["value"])

    cluster_counts = np.bincount(cleaned_labels.reshape(-1), minlength=len(match_result.assignments))
    total_pixels = cleaned_labels.size
    for assignment in match_result.assignments:
        count = int(cluster_counts[assignment.cluster_id])
        assignment.pixel_count = count
        assignment.coverage_pct = (count / total_pixels) * 100.0 if total_pixels else 0.0

    cluster_to_code = np.array([assignment.entry.yarn_code for assignment in match_result.assignments], dtype=np.int32)
    yarn_code_map = cluster_to_code[cleaned_labels]

    palette_sequence = palette_match.build_palette_sequence(match_result.assignments, match_result.palette_entries, order=args.palette_order)
    code_to_index = {entry.yarn_code: idx for idx, entry in enumerate(palette_sequence)}

    try:
        indexed_map = _remap_yarn_codes(yarn_code_map, code_to_index)
    except KeyError as exc:
        print(f"Missing palette entry for yarn code {exc.args[0]}", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / f"{args.input.stem}_indexed.{args.format}"
    compression = args.compression if args.format == "tiff" else None
    palette_rgb = [(entry.r, entry.g, entry.b) for entry in palette_sequence]
    tiff_writer.write_indexed_image(output_path, indexed_map, palette_rgb, args.ppi, fmt=args.format, compression=compression)

    masks_dir = out_dir / "masks"
    tiff_writer.write_color_masks(masks_dir, indexed_map, [entry.yarn_code for entry in palette_sequence])

    usage_path = out_dir / "usage_report.csv"
    _write_usage_report(
        usage_path,
        indexed_map,
        palette_sequence,
        {assignment.entry.yarn_code: assignment.delta_e for assignment in match_result.assignments},
    )
    _log_progress("Output writing", args.verbose, start_time, estimated_total_holder["value"])

    _log_quality_report(indexed_map, match_result, palette_sequence, args.deltaE_warn)

    if args.verbose:
        print(f"Saved indexed image to {output_path}")
        print(f"Saved masks to {masks_dir}")
        print(f"Saved usage report to {usage_path}")
        elapsed = time.perf_counter() - start_time
        print(f"Actual finishing time: {_format_duration(elapsed)}")

    return 0


def _remap_yarn_codes(yarn_code_map: np.ndarray, code_to_index: dict[int, int]) -> np.ndarray:
    codes_sorted = np.array(sorted(code_to_index.keys()), dtype=np.int32)
    indices = np.array([code_to_index[code] for code in codes_sorted], dtype=np.uint8)
    positions = np.searchsorted(codes_sorted, yarn_code_map)
    valid = (positions < codes_sorted.size) & (codes_sorted[positions] == yarn_code_map)
    if not np.all(valid):
        missing_codes = np.unique(yarn_code_map[~valid])
        raise KeyError(int(missing_codes[0]))
    return indices[positions].astype(np.uint8)


def _write_usage_report(
    path: Path,
    index_map: np.ndarray,
    palette_entries: Sequence[palette_match.PaletteEntry],
    delta_by_code: dict[int, float],
) -> None:
    total_pixels = index_map.size
    counts = np.bincount(index_map.reshape(-1), minlength=len(palette_entries))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "yarn_code", "name", "R", "G", "B", "pixels", "percent", "deltaE"])
        for idx, entry in enumerate(palette_entries):
            pixels = int(counts[idx])
            percent = (pixels / total_pixels) * 100.0 if total_pixels else 0.0
            delta = delta_by_code.get(entry.yarn_code, 0.0)
            writer.writerow([
                idx,
                entry.yarn_code,
                entry.name,
                entry.r,
                entry.g,
                entry.b,
                pixels,
                f"{percent:.4f}",
                f"{delta:.4f}",
            ])


def _log_quality_report(
    index_map: np.ndarray,
    match_result: palette_match.PaletteMatchResult,
    palette_entries: Sequence[palette_match.PaletteEntry],
    delta_warn: float,
) -> None:
    h, w = index_map.shape
    stats = match_result.delta_stats
    print(f"Final image size: {w}x{h} px")
    print(f"Quantized clusters: {len(match_result.assignments)}")
    print(f"Palette colors used: {len(palette_entries)}")
    print(f"ΔE2000 min/avg/max: {stats['min']:.3f} / {stats['avg']:.3f} / {stats['max']:.3f}")

    coverage = sorted(
        (
            assignment.entry.yarn_code,
            assignment.entry.name,
            assignment.coverage_pct,
        )
        for assignment in match_result.assignments
    )
    coverage.sort(key=lambda item: item[2], reverse=True)
    print("Top colors:")
    for yarn_code, name, pct in coverage[:3]:
        print(f"  #{yarn_code}: {name} -> {pct:.2f}%")

    exceed = [
        (assignment.entry.yarn_code, assignment.delta_e)
        for assignment in match_result.assignments
        if assignment.delta_e > delta_warn
    ]
    for yarn_code, delta in exceed:
        print(f"Warning: yarn {yarn_code} ΔE2000={delta:.2f} exceeds threshold {delta_warn}")


def _format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def _log_progress(
    phase: str,
    verbose: bool,
    start_time: float,
    estimated_total_seconds: Optional[float],
) -> None:
    if not verbose:
        return
    elapsed = time.perf_counter() - start_time
    if estimated_total_seconds is not None:
        remaining = max(1.0, estimated_total_seconds - elapsed)
        print(
            f"[{phase}] elapsed {_format_duration(elapsed)}, remaining ≈ {_format_duration(remaining)}"
        )
    else:
        print(f"[{phase}] elapsed {_format_duration(elapsed)}")


def _build_quant_progress_callback(
    start_time: float,
    estimated_total_holder: dict[str, Optional[float]],
) -> Callable[[int, int], None]:
    last_fraction = {"value": -0.1}

    def _callback(processed: int, total: int) -> None:
        fraction = processed / total if total else 1.0
        if fraction < 1.0 and fraction - last_fraction["value"] < 0.05:
            return
        last_fraction["value"] = fraction
        elapsed = time.perf_counter() - start_time
        if fraction > 0 and total:
            projected_total = (elapsed / fraction) * 1.05
            current_estimate = estimated_total_holder.get("value")
            if current_estimate is None or projected_total > current_estimate:
                estimated_total_holder["value"] = projected_total
        remaining = None
        current_estimate = estimated_total_holder.get("value")
        if current_estimate is not None:
            remaining = max(0.0, current_estimate - elapsed)
        message = (
            f"[Quantization] {fraction * 100:.1f}% elapsed {_format_duration(elapsed)}"
        )
        if remaining is not None:
            message += f", remaining ≈ {_format_duration(remaining)}"
        print(message)

    return _callback


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
