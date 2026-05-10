"""Registry for token-eval mode runners."""

from __future__ import annotations

from .artifact_progressive import ArtifactProgressiveEvalRunner
from .category import CategoryEvalRunner
from .progressive import ProgressiveEvalRunner
from .standard import StandardEvalRunner


_RUNNERS = {
    "standard": StandardEvalRunner(),
    "progressive": ProgressiveEvalRunner(),
    "category": CategoryEvalRunner(),
    "artifact_progressive": ArtifactProgressiveEvalRunner(),
}


def list_eval_modes() -> tuple[str, ...]:
    return tuple(_RUNNERS)


def get_eval_mode_runner(name: str):
    try:
        return _RUNNERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown eval mode: {name!r}. Available: {sorted(_RUNNERS)}") from exc
