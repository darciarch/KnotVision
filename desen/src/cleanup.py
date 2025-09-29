from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np
from skimage.morphology import remove_small_objects


def apply_cleanup(
    label_map: np.ndarray,
    min_region: int = 12,
    use_majority_filter: bool = True,
    morph_operations: Sequence[str] = (),
) -> np.ndarray:
    result = label_map.astype(np.int32, copy=True)

    if min_region > 0:
        result = _remove_small_regions(result, min_region)

    if use_majority_filter:
        result = _apply_majority_filter(result)

    if morph_operations:
        result = _apply_morphology(result, morph_operations)

    return result.astype(label_map.dtype)


def _remove_small_regions(label_map: np.ndarray, min_size: int) -> np.ndarray:
    offset = label_map + 1
    cleaned = remove_small_objects(offset, min_size=min_size, connectivity=1)
    cleaned = cleaned.astype(np.int32) - 1
    removed_mask = cleaned < 0
    if np.any(removed_mask):
        fill = _apply_majority_filter(label_map)
        cleaned[removed_mask] = fill[removed_mask]
    return cleaned


def _apply_majority_filter(label_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    best_scores = np.full(label_map.shape, -1.0, dtype=np.float32)
    best_labels = np.zeros(label_map.shape, dtype=np.int32)

    values, counts = np.unique(label_map, return_counts=True)
    order = values[np.argsort(-counts)]
    for label in order:
        mask = (label_map == label).astype(np.uint8)
        if not mask.any():
            continue
        score = cv2.filter2D(mask.astype(np.float32), ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
        update = score > best_scores
        best_labels[update] = label
        best_scores[update] = score[update]
    unresolved = best_scores < 0
    if np.any(unresolved):
        best_labels[unresolved] = label_map[unresolved]
    return best_labels


def _apply_morphology(label_map: np.ndarray, operations: Iterable[str], kernel_size: int = 3) -> np.ndarray:
    result = label_map.copy()
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    for op in operations:
        op_lower = op.lower()
        if op_lower not in {"open", "close"}:
            raise ValueError(f"Unsupported morphology op '{op}'")
        values, counts = np.unique(result, return_counts=True)
        order = values[np.argsort(-counts)]
        updated = np.full(result.shape, -1, dtype=np.int32)
        scores = np.full(result.shape, -1.0, dtype=np.float32)
        for label in order:
            mask = (result == label).astype(np.uint8)
            if not mask.any():
                continue
            morph_flag = cv2.MORPH_OPEN if op_lower == "open" else cv2.MORPH_CLOSE
            morphed = cv2.morphologyEx(mask, morph_flag, kernel)
            if not morphed.any():
                continue
            score = cv2.filter2D(morphed.astype(np.float32), ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
            update_mask = (morphed > 0) & (score >= scores)
            updated[update_mask] = label
            scores[update_mask] = score[update_mask]
        if op_lower == "close":
            fallback = result
        else:
            fallback = _apply_majority_filter(result)
        unresolved = updated < 0
        updated[unresolved] = fallback[unresolved]
        result = updated
    return result
