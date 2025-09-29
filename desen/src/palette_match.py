from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor

# numpy.asscalar was removed in recent NumPy releases while
# older colormath versions still rely on it internally. Provide a
# lightweight compatibility shim so delta_e_cie2000 keeps working.
if not hasattr(np, "asscalar"):
    np.asscalar = lambda array: np.asarray(array).item()

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - SciPy may be absent at runtime
    linear_sum_assignment = None


@dataclass(frozen=True)
class PaletteEntry:
    yarn_code: int
    name: str
    r: int
    g: int
    b: int
    material: Optional[str] = None


@dataclass
class ClusterAssignment:
    cluster_id: int
    palette_index: int
    entry: PaletteEntry
    delta_e: float
    pixel_count: int
    coverage_pct: float


@dataclass
class PaletteMatchResult:
    assignments: List[ClusterAssignment]
    delta_matrix: np.ndarray
    palette_entries: List[PaletteEntry]

    @property
    def delta_stats(self) -> Dict[str, float]:
        values = [assign.delta_e for assign in self.assignments]
        if not values:
            return {"min": 0.0, "avg": 0.0, "max": 0.0}
        return {
            "min": float(np.min(values)),
            "avg": float(np.mean(values)),
            "max": float(np.max(values)),
        }


def load_palette(csv_path: Path) -> List[PaletteEntry]:
    entries: List[PaletteEntry] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"yarn_code", "name", "R", "G", "B"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Palette CSV missing required columns: {sorted(missing)}")
        for row in reader:
            try:
                yarn_code = int(row["yarn_code"].strip())
                r = int(row["R"].strip())
                g = int(row["G"].strip())
                b = int(row["B"].strip())
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid palette row: {row}") from exc
            name = row.get("name", "").strip()
            material = row.get("material")
            material = material.strip() if material else None
            entries.append(PaletteEntry(yarn_code=yarn_code, name=name, r=r, g=g, b=b, material=material))
    if not entries:
        raise ValueError("Palette file is empty")
    return entries


def match_clusters_to_palette(
    centers_lab: np.ndarray,
    labels: np.ndarray,
    palette_entries: Sequence[PaletteEntry],
) -> PaletteMatchResult:
    k = centers_lab.shape[0]
    if len(palette_entries) < k:
        raise ValueError(f"Palette has {len(palette_entries)} entries but requires at least {k} for mapping.")

    total_pixels = labels.size
    counts = np.bincount(labels.reshape(-1), minlength=k)

    cluster_lab = [_cv2lab_to_labcolor(center) for center in centers_lab]
    palette_lab = [_palette_entry_to_lab(entry) for entry in palette_entries]

    delta_matrix = np.zeros((k, len(palette_entries)), dtype=np.float32)
    for i, c_lab in enumerate(cluster_lab):
        delta_matrix[i, :] = [_delta_e(c_lab, p_lab) for p_lab in palette_lab]

    assignment_map = _solve_assignment(delta_matrix)

    assignments: List[ClusterAssignment] = []
    for cluster_id in range(k):
        palette_index = assignment_map[cluster_id]
        entry = palette_entries[palette_index]
        delta = float(delta_matrix[cluster_id, palette_index])
        pixel_count = int(counts[cluster_id])
        coverage = (pixel_count / total_pixels) * 100.0 if total_pixels else 0.0
        assignments.append(
            ClusterAssignment(
                cluster_id=cluster_id,
                palette_index=palette_index,
                entry=entry,
                delta_e=delta,
                pixel_count=pixel_count,
                coverage_pct=coverage,
            )
        )

    return PaletteMatchResult(assignments=assignments, delta_matrix=delta_matrix, palette_entries=list(palette_entries))


def build_palette_sequence(
    assignments: Iterable[ClusterAssignment],
    palette_entries: Sequence[PaletteEntry],
    order: str = "code_asc",
) -> List[PaletteEntry]:
    assignment_list = list(assignments)
    used_codes = {assignment.entry.yarn_code for assignment in assignment_list if assignment.pixel_count > 0}
    if not used_codes and assignment_list:
        used_codes = {assignment_list[0].entry.yarn_code}

    if order == "csv":
        ordered = [entry for entry in palette_entries if entry.yarn_code in used_codes]
    elif order == "code_asc":
        unique_entries = {entry.yarn_code: entry for entry in palette_entries if entry.yarn_code in used_codes}
        ordered = [unique_entries[code] for code in sorted(unique_entries)]
    else:
        raise ValueError(f"Unsupported palette order '{order}'")
    return ordered


def _solve_assignment(cost_matrix: np.ndarray) -> Dict[int, int]:
    k, n = cost_matrix.shape
    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapping = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
        if len(mapping) == k:
            return mapping
    pairs = [
        (int(cluster), int(palette), float(cost_matrix[cluster, palette]))
        for cluster in range(k)
        for palette in range(n)
    ]
    pairs.sort(key=lambda item: item[2])
    assigned_clusters: Dict[int, int] = {}
    used_palette: set[int] = set()
    for cluster, palette, _ in pairs:
        if cluster in assigned_clusters or palette in used_palette:
            continue
        assigned_clusters[cluster] = palette
        used_palette.add(palette)
        if len(assigned_clusters) == k:
            break
    if len(assigned_clusters) != k:
        raise RuntimeError("Failed to assign palette colors uniquely to clusters")
    return assigned_clusters


def _palette_entry_to_lab(entry: PaletteEntry) -> LabColor:
    srgb = sRGBColor(entry.r, entry.g, entry.b, is_upscaled=True)
    return convert_color(srgb, LabColor)


def _cv2lab_to_labcolor(center: np.ndarray) -> LabColor:
    L = float(center[0]) * 100.0 / 255.0
    a = float(center[1]) - 128.0
    b = float(center[2]) - 128.0
    return LabColor(lab_l=L, lab_a=a, lab_b=b)


def _delta_e(c_lab: LabColor, p_lab: LabColor) -> float:
    return float(delta_e_cie2000(c_lab, p_lab))
