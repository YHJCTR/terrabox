"""Data loaders for OpenEarth and EarthBench datasets."""
from __future__ import annotations

import json
import logging
import re
from typing import Iterator, Optional

from .trajectory import Trajectory, Turn

logger = logging.getLogger(__name__)

# Map legacy OpenEarth tool names → current Terrabox slugs
# Slugs must match actual CoreRegistry registrations (verified against registry output)
_OPENEARTH_TOOL_MAP: dict[str, str] = {
    "GetAreaBoundary": "osm_gis.get_area_boundary",
    "AddPoisLayer": "osm_gis.add_pois_layer",
    "ComputeRouteDist": "osm_gis.compute_route_dist",
    "AddIndexLayer": "geo_raster.calculate_index",      # was georaster.* (wrong prefix)
    "ComputeIndexChange": "geo_raster.raster_diff",     # was georaster.* (wrong prefix)
    "ComputeRasterStatistics": "geo_raster.statistics",
    "ImageDescription": "geo_perception.vlm_analyze",
    "TextToBbox": "geo_perception.remotesam",           # was remotesam_segment (not in registry)
    "CountGivenObject": "geo_perception.remotesam",     # was remotesam_segment (not in registry)
    "SAM2Segment": "geo_perception.sam2_segment",
    "RemoteCLIPAnalysis": "geo_perception.remoteclip_analysis",
    "StripRCNNDetect": "geo_perception.strip_rcnn_detect",
    "Calculator": "ipython_code.execute",
    "PythonCode": "ipython_code.execute",
    "ExecutePython": "ipython_code.execute",
    "BingSearch": "bing_search.search",
    "STACSearch": "stac_basic.search",
    "GeoGridify": "geobasic.gridify",
    "GeoBuffer": "geoanalysis.buffer",
    "GeoDissolve": "geoanalysis.dissolve",
    "GeoIntersection": "geoanalysis.intersection",
    "GeoUnion": "geoanalysis.union",
    "GeoCentroid": "geoanalysis.centroid",
    "CalculateArea": "geobasic.area",
    "CalculateDistance": "geobasic.distance",
    "ValidateAOI": "geobasic.aoi_validate",
}

# Tool names that appear in training data but are workflow noise (not real tool calls).
# These are filtered out when building Trajectory.tools_called.
_NOISE_TOOLS: frozenset[str] = frozenset({
    "Terminate",
    "Solver",
    "GoogleSearch",
    "Plot",
    "DrawBox",
    "RegionAttributeDescription",
    "ShowIndexLayer",
    "Finish",
    "Done",
    "FinalAnswer",
})

# EarthBench uses a similar mapping
_EARTHBENCH_TOOL_MAP: dict[str, str] = {
    **_OPENEARTH_TOOL_MAP,
    "ComputeTVDI": "geo_raster.calculate_index",
    "ComputeNDVI": "geo_raster.calculate_index",
    "ComputeNDWI": "geo_raster.calculate_index",
    "ComputeNDBI": "geo_raster.calculate_index",
    "AnalyzeTrend": "geo_statistics.trend_analysis",
    "GenerateHeatmap": "geo_statistics.heatmap",
    "Summarize": "geo_statistics.summarize",
}


def _normalize_slug(raw_name: str, tool_map: dict[str, str]) -> str:
    """Map a legacy tool name to a Terrabox slug, or return lowercased fallback."""
    if raw_name in tool_map:
        return tool_map[raw_name]
    # Try case-insensitive match
    for k, v in tool_map.items():
        if k.lower() == raw_name.lower():
            return v
    # Return as-is if already looks like a slug (contains a dot)
    if "." in raw_name:
        return raw_name
    logger.debug(f"Unknown tool name: {raw_name!r} — using as-is")
    return raw_name


def _parse_gpt_turn(value: str) -> tuple[Optional[str], list[str]]:
    """Parse a GPT conversation turn value.

    Returns (thought_text, list_of_tool_slugs_called).
    """
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value, []

    thought = data.get("thought", "")
    actions = data.get("actions", [])
    tool_names = []
    for action in actions:
        name = action.get("name", "")
        if name:
            tool_names.append(name)
    return thought, tool_names


class OpenEarthLoader:
    """Load trajectories from data/openearth/train.json and eval.jsonl."""

    def __init__(self, train_path: str, eval_path: str):
        self.train_path = train_path
        self.eval_path = eval_path

    def iter_train(self, limit: Optional[int] = None) -> Iterator[Trajectory]:
        """Iterate over train.json records as Trajectory objects (low memory)."""
        count = 0
        with open(self.train_path, encoding="utf-8") as f:
            # train.json is a JSON array — parse incrementally
            raw = json.load(f)

        for record in raw:
            if limit is not None and count >= limit:
                break
            traj = self._parse_train_record(record)
            if traj is not None:
                yield traj
                count += 1

    def load_train_trajectories(self, limit: Optional[int] = None) -> list[Trajectory]:
        return list(self.iter_train(limit=limit))

    def load_eval_cases(self) -> list[dict]:
        """Return raw eval.jsonl dicts (each has question + expected_tools)."""
        cases = []
        with open(self.eval_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        return cases

    def load_combined_eval_cases(self, earthbench_eval_path: Optional[str] = None) -> list[dict]:
        """Return eval cases from openearth AND earthbench combined.

        Args:
            earthbench_eval_path: Path to earthbench eval.jsonl.  If None, inferred
                from eval_path (sibling directory data/earthbench/eval.jsonl).
        """
        import os as _os
        cases = self.load_eval_cases()

        if earthbench_eval_path is None:
            earthbench_eval_path = _os.path.join(
                _os.path.dirname(_os.path.dirname(self.eval_path)),
                "earthbench", "eval.jsonl",
            )

        if _os.path.exists(earthbench_eval_path):
            with open(earthbench_eval_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cases.append(json.loads(line))
            logger.info(f"Combined eval: {len(cases)} cases (openearth + earthbench)")
        else:
            logger.warning(f"EarthBench eval not found at {earthbench_eval_path} — openearth only")

        return cases

    def _parse_train_record(self, record: dict) -> Optional[Trajectory]:
        """Convert a train.json record into a Trajectory."""
        conversation = record.get("conversation", [])
        turns: list[Turn] = []
        tools_called: list[str] = []

        for turn in conversation:
            role = turn.get("from", "")
            value = turn.get("value", "")

            if role == "human":
                # Strip <AGENT_PROMPT> wrapper if present
                clean = re.sub(r"<AGENT_PROMPT>\s*\n?", "", value).strip()
                turns.append(Turn(role="human", content=clean))
            elif role == "gpt":
                thought, raw_tools = _parse_gpt_turn(value)
                content = thought or value
                turns.append(Turn(role="assistant", content=content))
                for raw_name in raw_tools:
                    if raw_name in _NOISE_TOOLS:
                        continue
                    slug = _normalize_slug(raw_name, _OPENEARTH_TOOL_MAP)
                    tools_called.append(slug)
                    turns.append(Turn(
                        role="tool",
                        content="",
                        tool_name=slug,
                        tool_args=None,
                    ))

        question = record.get("question", "")
        label = record.get("label", "")
        task_type = record.get("type", "unknown")

        return Trajectory(
            task_id=f"oe_train_{record.get('idx', 0)}",
            question=question,
            images=record.get("images", []),
            turns=turns,
            tools_called=list(dict.fromkeys(tools_called)),  # dedup, preserve order
            expected_tools=[],   # train split has no expected_tools ground truth
            final_answer=label,
            success=True,        # assume train examples are successful demonstrations
            source="openearth",
            task_type=task_type,
        )


    def load_test_trajectories(
        self,
        limit: Optional[int] = None,
    ) -> list["Trajectory"]:
        """Parse test.json (same format as train.json) and merge expected_tools
        from the paired eval.jsonl so metrics can be computed without running
        the agent.

        Matching: test.json record ``idx`` N  ↔  eval.jsonl record id
        ``oea_test_N``.  If no match is found the trajectory is kept but
        ``expected_tools`` remains empty (evaluator falls back to heuristic).
        """
        # Load expected_tools index from eval.jsonl (keyed by id)
        expected_by_id: dict[str, list[str]] = {}
        try:
            with open(self.eval_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        expected_by_id[rec.get("id", "")] = rec.get("expected_tools", [])
        except FileNotFoundError:
            logger.warning(f"eval.jsonl not found at {self.eval_path} — expected_tools will be empty")

        # Determine test.json path: same directory as train.json, file "test.json"
        import os as _os
        test_path = _os.path.join(_os.path.dirname(self.train_path), "test.json")
        with open(test_path, encoding="utf-8") as f:
            raw = json.load(f)

        trajectories: list[Trajectory] = []
        for record in raw:
            if limit is not None and len(trajectories) >= limit:
                break
            traj = self._parse_train_record(record)
            if traj is None:
                continue
            idx = record.get("idx", 0)
            expected = expected_by_id.get(f"oea_test_{idx}", [])
            # Replace empty expected_tools with ground-truth from eval.jsonl
            trajectories.append(Trajectory(
                task_id=f"oea_test_{idx}",
                question=traj.question,
                images=traj.images,
                turns=traj.turns,
                tools_called=traj.tools_called,
                expected_tools=expected,
                final_answer=traj.final_answer,
                success=traj.success,
                source="openearth",
                task_type=traj.task_type,
            ))
        return trajectories


class EarthBenchLoader:
    """Load trajectories from data/earthbench/question.json and eval.jsonl."""

    def __init__(self, question_path: str, eval_path: str):
        self.question_path = question_path
        self.eval_path = eval_path

    def load_trajectories(self, limit: Optional[int] = None) -> list[Trajectory]:
        with open(self.question_path, encoding="utf-8") as f:
            data = json.load(f)

        trajectories = []
        for key, record in data.items():
            if limit is not None and len(trajectories) >= limit:
                break
            traj = self._parse_record(key, record)
            if traj is not None:
                trajectories.append(traj)
        return trajectories

    def load_eval_cases(self) -> list[dict]:
        cases = []
        with open(self.eval_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        return cases

    def _parse_record(self, key: str, record: dict) -> Optional[Trajectory]:
        tools_called: list[str] = []
        turns: list[Turn] = []

        for dlg in record.get("dialogs", []):
            role = dlg.get("role", "")
            content = dlg.get("content", "")

            if role in ("user", "human"):
                turns.append(Turn(role="human", content=content))
            elif role in ("assistant", "gpt"):
                turns.append(Turn(role="assistant", content=content))
                for tc in dlg.get("tool_calls", []):
                    name = tc.get("function", {}).get("name", "")
                    if name:
                        slug = _normalize_slug(name, _EARTHBENCH_TOOL_MAP)
                        tools_called.append(slug)
                        turns.append(Turn(role="tool", content="", tool_name=slug))

        dialogs = record.get("dialogs", [])
        question = dialogs[0].get("content", key) if dialogs else key
        tools_meta = record.get("tools", [])
        expected = [_normalize_slug(t.get("name", ""), _EARTHBENCH_TOOL_MAP) for t in tools_meta]

        return Trajectory(
            task_id=f"eb_{key}",
            question=question,
            images=record.get("files", []),
            turns=turns,
            tools_called=list(dict.fromkeys(tools_called)),
            expected_tools=expected,
            final_answer="",
            success=True,
            source="earthbench",
            task_type="unknown",
        )
