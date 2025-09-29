"""DRC application entrypoint and pipeline definition."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from drc.pipeline import Pipeline, PipelineStage, PipelineState
from src.input_normalize import NormalizedInput, normalize_input
from src.mask_segment import MaskResult, segment_mask


def stage_input_normalize(state: PipelineState) -> NormalizedInput:
    """Stage 1: read and normalize the raw input image."""
    normalized = normalize_input()
    return normalized


def stage_mask_segment(state: PipelineState) -> MaskResult:
    """Stage 2: generate rug segmentation mask via SAM."""
    normalized = state.get("input_normalize")
    if not isinstance(normalized, NormalizedInput):
        raise RuntimeError("input_normalize stage did not return NormalizedInput")
    result = segment_mask(normalized)
    return result


def build_pipeline() -> Pipeline:
    stages = [
        PipelineStage(
            name="input_normalize",
            description="Load, sanitize, and cache the working copy of the input image",
            action=stage_input_normalize,
        ),
        PipelineStage(
            name="mask_segment",
            description="Generate rug mask using SAM and post-processing",
            action=stage_mask_segment,
        ),
    ]
    return Pipeline(stages=stages)


def main() -> None:
    pipeline = build_pipeline()
    state = PipelineState(root_dir=Path.cwd())
    pipeline.run(state)
    summary = state.summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
