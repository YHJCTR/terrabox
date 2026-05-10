"""Token-eval runner for artifact-progressive mode."""

from __future__ import annotations

from ..artifact_progressive_graph import run_artifact_progressive_loop
from .base import EvalModeContext, EvalModeResult


class ArtifactProgressiveEvalRunner:
    name = "artifact_progressive"

    def run(self, context: EvalModeContext) -> EvalModeResult:
        if context.verbose:
            print("\n" + "#" * 70)
            print("  MODE: ARTIFACT-PROGRESSIVE (stateful disclosure + artifact IO)")
            print("#" * 70)
        result = run_artifact_progressive_loop(
            llm=context.llm,
            question=context.question,
            config=context.config,
            user=context.user,
            image_paths=context.image_paths,
            allowed_slugs=context.allowed_slugs,
            verbose=context.verbose,
        )
        return EvalModeResult(
            messages=result.get("messages", []),
            final=result.get("final", ""),
            elapsed=result.get("elapsed", 0.0),
            extra={"artifact_state": result.get("artifact_state", {})},
        )
