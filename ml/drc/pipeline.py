"""Core pipeline primitives for orchestrating DRC processing stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


class PipelineError(RuntimeError):
    """Raised when the pipeline encounters an unrecoverable state."""


@dataclass
class PipelineState:
    """Mutable state shared across pipeline stages."""

    root_dir: Path
    stage_results: Dict[str, Any] = field(default_factory=dict)

    def record(self, key: str, value: Any, *, replace: bool = False) -> None:
        if not replace and key in self.stage_results:
            raise PipelineError(f"Stage '{key}' has already produced a result")
        self.stage_results[key] = value

    def get(self, key: str) -> Any:
        if key not in self.stage_results:
            raise PipelineError(f"Stage '{key}' has not been executed yet")
        return self.stage_results[key]

    def summary(self) -> Dict[str, Any]:
        return {name: self._serialize(value) for name, value in self.stage_results.items()}

    def _serialize(self, value: Any) -> Any:
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize(item) for item in value]
        return value


@dataclass
class PipelineStage:
    """A single stage in the pipeline."""

    name: str
    action: Callable[[PipelineState], Any]
    description: Optional[str] = None

    def run(self, state: PipelineState) -> None:
        print(f"[stage] {self.name}")
        result = self.action(state)
        if result is not None:
            state.record(self.name, result)


@dataclass
class Pipeline:
    """Sequential pipeline runner."""

    stages: List[PipelineStage]

    def run(self, state: PipelineState) -> PipelineState:
        for stage in self.stages:
            stage.run(state)
        return state

    def names(self) -> Iterable[str]:
        return (stage.name for stage in self.stages)
