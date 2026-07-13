from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from terrabox.evolution.promptevo.interfaces import MetricSpec, TaskMetric
from terrabox.evolution.shared.rollout_report import compare_experiments, status_report


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = "terrabox"
DEFAULT_TOTAL = 1162


@dataclass(frozen=True)
class MetricField:
    key: str
    label: str
    direction: str = "neutral"
    format: str = "number"
    decimals: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "direction": self.direction,
            "format": self.format,
            "decimals": self.decimals,
        }


@dataclass(frozen=True)
class ExperimentRef:
    name: str
    scene: str
    source_path: Path
    kind: str = "experiment"
    prompt_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        suffix = self.metadata.get("snapshot_key")
        return f"{self.source_path.resolve()}#{suffix}" if suffix else str(self.source_path.resolve())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "scene": self.scene,
            "kind": self.kind,
            "source_path": str(self.source_path.resolve()),
            "results_dir": str(self.source_path.resolve()),
            "prompt_path": str(self.prompt_path) if self.prompt_path else None,
        }


class SceneAdapter:
    scene = "generic"
    label = "Generic"
    supports_scope = False

    def discover(self) -> list[ExperimentRef]:
        raise NotImplementedError

    def status(self, ref: ExperimentRef, scope: str = "all") -> dict[str, Any]:
        raise NotImplementedError

    def compare(self, current: ExperimentRef, baseline: ExperimentRef) -> dict[str, Any]:
        raise NotImplementedError

    def metric_schema(self) -> list[MetricField]:
        raise NotImplementedError

    def prompt(self, ref: ExperimentRef) -> dict[str, Any]:
        path = ref.prompt_path
        return {
            "experiment": ref.name,
            "scene": ref.scene,
            "source_path": str(ref.source_path),
            "prompt_path": str(path) if path else None,
            "prompt": path.read_text(encoding="utf-8", errors="replace") if path and path.is_file() else "",
            "found": bool(path and path.is_file()),
        }


TERRABOX_FIELDS = [
    MetricField("success_rate", "success_rate", "higher_better", "percent"),
    MetricField("set_f1", "set-F1", "higher_better", "decimal", 3),
    MetricField("multiset_f1", "multiset-F1", "higher_better", "decimal", 3),
    MetricField("exact_match", "exact_match", "higher_better", "percent"),
    MetricField("ordered_exact", "ordered_exact", "higher_better", "percent"),
    MetricField("any_order", "AnyOrder", "higher_better", "percent"),
    MetricField("same_order", "SameOrder", "higher_better", "percent"),
    MetricField("unique", "Unique", "higher_better", "percent"),
    MetricField("f1_perception", "F1 perception", "higher_better"),
    MetricField("f1_operation", "F1 operation", "higher_better"),
    MetricField("f1_logic", "F1 logic", "higher_better"),
    MetricField("f1_gis", "F1 gis", "higher_better"),
    MetricField("empty_rate", "empty_rate", "lower_better", "percent"),
    MetricField("cap_rate", "cap_rate", "lower_better", "percent"),
    MetricField("repeat4_tasks", "same-tool>=4 tasks", "lower_better", "integer", 0),
    MetricField("errors_per_task", "errors/task", "lower_better"),
    MetricField("tools_per_task", "tools/task", "neutral"),
    MetricField("llm_per_task", "llm/task", "neutral"),
    MetricField("tokens_per_task", "tokens/task", "neutral", "integer", 0),
    MetricField("time_per_task", "time/task", "lower_better", "seconds", 1),
]


class TerraboxSceneAdapter(SceneAdapter):
    scene = "terrabox"
    label = "Terrabox / OEA"
    supports_scope = True

    def discover(self) -> list[ExperimentRef]:
        refs: list[ExperimentRef] = []
        for root in self._candidate_roots():
            if not root.exists():
                continue
            for results_dir in root.rglob("results"):
                if not results_dir.is_dir() or not any(results_dir.glob("*.json")):
                    continue
                name = self._display_name(results_dir)
                refs.append(
                    ExperimentRef(
                        name=name,
                        scene=self.scene,
                        source_path=results_dir.resolve(),
                        kind=self._kind_for(results_dir),
                        prompt_path=self._find_prompt(results_dir, name),
                    )
                )
        return sorted(_dedupe_refs(refs), key=lambda ref: (ref.kind, ref.name))

    def status(self, ref: ExperimentRef, scope: str = "all") -> dict[str, Any]:
        report = status_report(ref.source_path, scope=scope, total=DEFAULT_TOTAL)
        report.update(self._meta(ref))
        report["metric_schema"] = [field.to_dict() for field in self.metric_schema()]
        return report

    def compare(self, current: ExperimentRef, baseline: ExperimentRef) -> dict[str, Any]:
        report = compare_experiments(current.source_path, baseline.source_path)
        report.update(
            {
                "scene": self.scene,
                "current": self._ref_meta(current),
                "baseline": self._ref_meta(baseline),
                "metric_schema": [field.to_dict() for field in self.metric_schema()],
            }
        )
        return report

    def metric_schema(self) -> list[MetricField]:
        return TERRABOX_FIELDS

    def _meta(self, ref: ExperimentRef) -> dict[str, Any]:
        return {
            "experiment": ref.name,
            "scene": self.scene,
            "kind": ref.kind,
            "source_path": str(ref.source_path),
            "results_dir": str(ref.source_path),
            "supports_scope": self.supports_scope,
        }

    @staticmethod
    def _ref_meta(ref: ExperimentRef) -> dict[str, Any]:
        return {"name": ref.name, "scene": ref.scene, "kind": ref.kind, "source_path": str(ref.source_path)}

    @staticmethod
    def _candidate_roots() -> list[Path]:
        return [
            REPO_ROOT / "tmp" / "trajectories",
            REPO_ROOT / "src" / "terrabox" / "evolution" / "ReAct" / "exp",
            REPO_ROOT / "src" / "terrabox" / "evolution" / "reflection" / "exp",
        ]

    @staticmethod
    def _display_name(results_dir: Path) -> str:
        parts = results_dir.relative_to(REPO_ROOT).parts
        if parts[:2] == ("tmp", "trajectories") and len(parts) >= 3:
            return parts[2]
        for marker in ("exp", "experiments"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    tail = list(parts[idx + 1 : -1])
                    return "/".join(tail) if tail else results_dir.parent.name
        return results_dir.parent.name

    @staticmethod
    def _kind_for(results_dir: Path) -> str:
        parts = tuple(part.lower() for part in results_dir.relative_to(REPO_ROOT).parts)
        name = "/".join(parts)
        if "reflection" in parts:
            return "reflection"
        if "react" in parts:
            return "react"
        if "promptevo" in name:
            return "promptevo"
        return "rollout"

    def _find_prompt(self, results_dir: Path, exp_name: str) -> Path | None:
        names = [exp_name.split("/")[0]]
        if names[0].endswith("_rollout"):
            names.append(names[0][: -len("_rollout")])
        candidates: list[Path] = []
        for name in names:
            candidates.extend(
                [
                    REPO_ROOT / "evolution_store" / "promptevo" / "terrabox" / "versions" / f"{name}.txt",
                    REPO_ROOT / "evolution_store" / "promptevo" / "versions" / f"{name}.txt",
                ]
            )
        experiment_dir = results_dir.parents[1]
        candidates.extend(experiment_dir / name for name in _PROMPT_FILENAMES)
        return _first_file(candidates)


class ProviderSceneAdapter(SceneAdapter):
    def provider(self, ref: ExperimentRef):
        raise NotImplementedError

    def metric_schema(self) -> list[MetricField]:
        refs = self.discover()
        if not refs:
            return []
        return _fields_from_specs(self.provider(refs[0]).metric_specs())

    def status(self, ref: ExperimentRef, scope: str = "all") -> dict[str, Any]:
        provider = self.provider(ref)
        summary = _numeric_dict(provider.aggregate(ref.id))
        done = int(summary.get("n_total_evaluated", summary.get("n", 0)) or 0)
        total = int(summary.get("n_expected_api_calls", summary.get("n_expected", done)) or done)
        return {
            "experiment": ref.name,
            "scene": self.scene,
            "kind": ref.kind,
            "source_path": str(ref.source_path),
            "results_dir": str(ref.source_path),
            "done": done,
            "total": total,
            "progress_pct": (100.0 * done / total) if total else 0.0,
            "summary": summary,
            "supports_scope": self.supports_scope,
            "metric_schema": [field.to_dict() for field in _fields_from_specs(provider.metric_specs())],
        }

    def compare(self, current: ExperimentRef, baseline: ExperimentRef) -> dict[str, Any]:
        cur_provider = self.provider(current)
        base_provider = self.provider(baseline)
        cur_tasks = cur_provider.per_task(current.id)
        base_tasks = base_provider.per_task(baseline.id)
        common = sorted(set(cur_tasks) & set(base_tasks))
        cur = _numeric_dict(cur_provider.aggregate(current.id, common))
        base = _numeric_dict(base_provider.aggregate(baseline.id, common))
        keys = set(cur) | set(base)
        delta = {key: float(cur.get(key, 0)) - float(base.get(key, 0)) for key in keys}
        up = sum(not base_tasks[key].success and cur_tasks[key].success for key in common)
        down = sum(base_tasks[key].success and not cur_tasks[key].success for key in common)
        return {
            "scene": self.scene,
            "n_common": len(common),
            "base": base,
            "cur": cur,
            "delta": delta,
            "success_flips_up": up,
            "success_flips_down": down,
            "success_flips_net": up - down,
            "current": current.to_dict(),
            "baseline": baseline.to_dict(),
            "metric_schema": [field.to_dict() for field in _fields_from_specs(cur_provider.metric_specs())],
        }


class APIBankSceneAdapter(ProviderSceneAdapter):
    scene = "api_bank"
    label = "API-Bank"
    root = REPO_ROOT / "src" / "terrabox" / "evolution" / "promptevo" / "adapters" / "api_bank" / "experiments"

    def discover(self) -> list[ExperimentRef]:
        refs: list[ExperimentRef] = []
        if not self.root.exists():
            return refs
        for predictions in self.root.glob("**/predictions.jsonl"):
            run_dir = predictions.parent
            if re.search(r"_shard\d+$", run_dir.name):
                continue
            relative = run_dir.relative_to(self.root)
            refs.append(
                ExperimentRef(
                    name=str(relative),
                    scene=self.scene,
                    source_path=run_dir,
                    kind=_stage_kind(run_dir.name),
                    prompt_path=self._find_prompt(run_dir),
                    metadata=self._metadata(run_dir),
                )
            )
        return sorted(_dedupe_refs(refs), key=lambda ref: (ref.name, ref.kind))

    def provider(self, ref: ExperimentRef):
        from terrabox.evolution.promptevo.adapters.api_bank import APIBankMetricProvider

        meta = ref.metadata or self._metadata(ref.source_path)
        data_dir = str(meta.get("data_dir") or "")
        return APIBankMetricProvider(
            data_dir_fn=lambda _exp: data_dir,
            prediction_path_fn=lambda _exp: str(ref.source_path / "predictions.jsonl"),
            rollout_path_fn=lambda _exp: str(ref.source_path / "rollout.jsonl"),
            include_missing=True,
        )

    def _metadata(self, run_dir: Path) -> dict[str, Any]:
        direct = _read_json(run_dir / "meta.json")
        if direct.get("data_dir"):
            return direct
        for meta_path in sorted(run_dir.parent.glob(f"{run_dir.name}_shard*/meta.json")):
            meta = _read_json(meta_path)
            if meta.get("data_dir"):
                return meta
        level = "level-2-toolsearcher" if "level2" in run_dir.name else "level-1-given-desc"
        return {"data_dir": f"/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank/lv1-lv2-samples/{level}"}

    def _find_prompt(self, run_dir: Path) -> Path | None:
        meta = self._metadata(run_dir)
        version = str(meta.get("prompt_version") or "")
        candidates = []
        if version:
            candidates.append(REPO_ROOT / "evolution_store" / "promptevo" / "api_bank" / "versions" / f"{version}.txt")
        root = run_dir.parent
        prefix = "stage2copy" if "stage2copy" in run_dir.name else "stage2" if "stage2" in run_dir.name else "stage1" if "stage1" in run_dir.name else "base"
        pointer = root / f"{prefix}_prompt_path.txt"
        if pointer.is_file():
            pointed = Path(pointer.read_text(encoding="utf-8").strip())
            candidates.append(pointed)
        candidates.extend(run_dir / name for name in _PROMPT_FILENAMES)
        return _first_file(candidates)


class Tau2SceneAdapter(ProviderSceneAdapter):
    scene = "tau2_bench"
    label = "tau2-bench"
    root = REPO_ROOT / "src" / "terrabox" / "evolution" / "promptevo" / "adapters" / "tau2_bench" / "experiments"

    def discover(self) -> list[ExperimentRef]:
        refs = []
        if not self.root.exists():
            return refs
        for results in self.root.glob("**/tau2_results"):
            if not any(results.rglob("*.json")):
                continue
            run_dir = results.parent
            refs.append(
                ExperimentRef(
                    name=str(run_dir.relative_to(self.root)),
                    scene=self.scene,
                    source_path=results,
                    kind=_stage_kind(run_dir.name),
                    prompt_path=_first_file([run_dir / "active_static_instruction.txt", *(run_dir / name for name in _PROMPT_FILENAMES)]),
                )
            )
        return sorted(_dedupe_refs(refs), key=lambda ref: ref.name)

    def provider(self, ref: ExperimentRef):
        from terrabox.evolution.promptevo.adapters.tau2_bench import Tau2MetricProvider

        return Tau2MetricProvider(results_path_fn=lambda _exp: str(ref.source_path))


class AgentDojoSceneAdapter(ProviderSceneAdapter):
    scene = "agentdojo"
    label = "AgentDojo"
    root = REPO_ROOT / "src" / "terrabox" / "evolution" / "promptevo" / "adapters" / "agentdojo" / "experiments"
    upstream_root = Path("/data1/yuhongjie2/agentdojo/runs")

    def discover(self) -> list[ExperimentRef]:
        refs: list[ExperimentRef] = []
        if self.root.exists():
            for run_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
                if not self._contains_results(run_dir):
                    continue
                refs.append(
                    ExperimentRef(
                        run_dir.name,
                        self.scene,
                        run_dir,
                        _stage_kind(run_dir.name),
                        self._find_prompt(run_dir),
                    )
                )
        if self.upstream_root.exists():
            for run_dir in sorted(path for path in self.upstream_root.iterdir() if path.is_dir()):
                if not self._contains_results(run_dir):
                    continue
                refs.append(
                    ExperimentRef(
                        f"upstream/{run_dir.name}",
                        self.scene,
                        run_dir,
                        "benchmark",
                        self._find_prompt(run_dir),
                    )
                )
        return sorted(_dedupe_refs(refs), key=lambda ref: (ref.kind, ref.name))

    def provider(self, ref: ExperimentRef):
        from terrabox.evolution.promptevo.adapters.agentdojo import AgentDojoMetricProvider

        return AgentDojoMetricProvider(results_path_fn=lambda _exp: str(ref.source_path))

    @staticmethod
    def _find_prompt(run_dir: Path) -> Path | None:
        version = REPO_ROOT / "evolution_store" / "promptevo" / "agentdojo" / "versions" / f"{run_dir.name}.txt"
        return _first_file([run_dir / "active_system_message.txt", version])

    @staticmethod
    def _contains_results(run_dir: Path) -> bool:
        for path in run_dir.rglob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(row, dict) and "messages" in row and "utility" in row:
                return True
        return False


class ToolBenchSceneAdapter(ProviderSceneAdapter):
    scene = "toolbench"
    label = "ToolBench"
    root = Path("/data1/yuhongjie2/ToolBench/data_example/answer")

    def discover(self) -> list[ExperimentRef]:
        refs = []
        if not self.root.exists():
            return refs
        for run_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            if any(run_dir.glob("*.json")):
                refs.append(ExperimentRef(run_dir.name, self.scene, run_dir, "example"))
        return refs

    def provider(self, ref: ExperimentRef):
        from terrabox.evolution.promptevo.adapters.toolbench import ToolBenchMetricProvider

        return ToolBenchMetricProvider(results_dir_fn=lambda _exp: str(ref.source_path))


class MetricViewerRegistry:
    def __init__(self, adapters: Iterable[SceneAdapter]):
        self.adapters = {adapter.scene: adapter for adapter in adapters}

    def scenes(self) -> list[dict[str, Any]]:
        rows = []
        for adapter in self.adapters.values():
            count = len(adapter.discover())
            rows.append(
                {
                    "id": adapter.scene,
                    "label": adapter.label,
                    "count": count,
                    "supports_scope": adapter.supports_scope,
                    "default": adapter.scene == DEFAULT_SCENE,
                }
            )
        return rows

    def discover(self, scene: str = DEFAULT_SCENE) -> list[ExperimentRef]:
        return self.adapter(scene).discover()

    def status(self, scene: str, experiment: str, scope: str = "all") -> dict[str, Any]:
        adapter = self.adapter(scene)
        return adapter.status(self.resolve(scene, experiment), scope=scope)

    def compare(self, scene: str, current: str, baseline: str) -> dict[str, Any]:
        adapter = self.adapter(scene)
        return adapter.compare(self.resolve(scene, current), self.resolve(scene, baseline))

    def prompt(self, scene: str, experiment: str) -> dict[str, Any]:
        adapter = self.adapter(scene)
        return adapter.prompt(self.resolve(scene, experiment))

    def adapter(self, scene: str) -> SceneAdapter:
        try:
            return self.adapters[scene]
        except KeyError as exc:
            raise ValueError(f"unknown scene: {scene}") from exc

    def resolve(self, scene: str, spec: str) -> ExperimentRef:
        for ref in self.discover(scene):
            if spec in {ref.id, ref.name, str(ref.source_path), str(ref.source_path.resolve())}:
                return ref
        raise ValueError(f"experiment not found in scene {scene}: {spec}")


def _fields_from_specs(specs: Iterable[MetricSpec]) -> list[MetricField]:
    fields = []
    for spec in specs:
        key = spec.name
        if key.startswith("n_") or key in {"n", "n_total_evaluated"}:
            fmt, decimals = "integer", 0
        elif key.endswith("_rate") or key.endswith("_accuracy") or key in {"success_rate", "pass_rate"}:
            fmt, decimals = "ratio_percent", 2
        elif key.endswith("_s") or "duration" in key or "latency" in key:
            fmt, decimals = "seconds", 2
        elif "f1" in key:
            fmt, decimals = "decimal", 3
        else:
            fmt, decimals = "number", 3 if "reward" in key else 2
        fields.append(MetricField(key, key.replace("_", " "), spec.direction, fmt, decimals))
    return fields


def _numeric_dict(data: dict[str, Any]) -> dict[str, float | int]:
    return {key: value for key, value in data.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}


def _dedupe_refs(refs: Iterable[ExperimentRef]) -> list[ExperimentRef]:
    out: dict[str, ExperimentRef] = {}
    for ref in refs:
        out[ref.id] = ref
    return list(out.values())


def _first_file(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path and path.is_file():
            return path.resolve()
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _stage_kind(name: str) -> str:
    lower = name.lower()
    if "stage2" in lower:
        return "stage2"
    if "stage1" in lower:
        return "stage1"
    if "base" in lower:
        return "base"
    if "smoke" in lower or "mock" in lower:
        return "smoke"
    return "run"


_PROMPT_FILENAMES = ("prompt.txt", "system_prompt.txt", "react_system_prompt.txt", "stage1_prompt.txt", "stage2_prompt.txt")


REGISTRY = MetricViewerRegistry(
    [
        TerraboxSceneAdapter(),
        APIBankSceneAdapter(),
        Tau2SceneAdapter(),
        AgentDojoSceneAdapter(),
        ToolBenchSceneAdapter(),
    ]
)

# Backward-compatible name for older callers. New code should use REGISTRY.
DEFAULT_ADAPTER = REGISTRY
