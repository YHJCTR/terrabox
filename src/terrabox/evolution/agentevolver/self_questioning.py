"""Self-Questioning: autonomous task generation for AgentEvolver.

From: AgentEvolver: Towards Efficient Self-Evolving Agent System (arXiv 2511.10395)

Self-Questioning reduces dependence on hand-crafted training datasets by enabling
agents to probe the environment and generate diverse synthetic tasks automatically.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional

from ..shared.trajectory import Trajectory

logger = logging.getLogger(__name__)

# Geo-task templates for question synthesis
_GEO_TASK_TEMPLATES = {
    "change_detection": {
        "pattern": "Analyze {metric} change in {location} between {t1} and {t2}.",
        "tool_sequence": ["osm_gis.get_area_boundary", "georaster.calculate_index", "georaster.raster_diff"],
        "slots": {"metric": ["NBR", "NDVI", "NDWI", "NDBI"], "t1": ["2018", "2019", "2020", "2021"], "t2": ["2022", "2023", "2024"]},
    },
    "poi_routing": {
        "pattern": "Find the {poi_type} closest to {landmark} in {area} within {buffer}m.",
        "tool_sequence": ["osm_gis.get_area_boundary", "osm_gis.add_pois_layer", "osm_gis.compute_route_dist"],
        "slots": {"poi_type": ["hospital", "school", "fire station", "police station"], "buffer": ["1000", "2000", "3000", "5000"]},
    },
    "segmentation": {
        "pattern": "Detect {object_type} in the image and calculate their combined area (GSD={gsd} m/pixel).",
        "tool_sequence": ["geo_perception.remotesam_segment", "ipython_code.execute"],
        "slots": {"object_type": ["buildings", "vehicles", "water bodies", "roads"], "gsd": ["0.1", "0.3", "0.5", "1.0"]},
    },
    "index_calculation": {
        "pattern": "Calculate {index} for {area} in {year} and summarize the results.",
        "tool_sequence": ["osm_gis.get_area_boundary", "georaster.calculate_index"],
        "slots": {"index": ["NDVI", "NDWI", "NDBI", "NBR"], "year": ["2022", "2023", "2024"]},
    },
    "classification": {
        "pattern": "Analyze the {scene_type} in the provided satellite image and identify key features.",
        "tool_sequence": ["geo_perception.vlm_analyze"],
        "slots": {"scene_type": ["urban area", "agricultural field", "forest", "coastal zone"]},
    },
}

# Known geo locations for template filling
_GEO_LOCATIONS = [
    "New York City, USA",
    "London, UK",
    "Paris, France",
    "Tokyo, Japan",
    "Sydney, Australia",
    "Cape Town, South Africa",
    "Amazon Rainforest, Brazil",
    "Sahara Desert, Algeria",
    "Tibetan Plateau, China",
    "Nile Delta, Egypt",
]


class TaskGenerator:
    """Self-Questioning module: mine templates from data and generate new tasks.

    Two modes:
    1. Data-driven mining: extract task templates from training trajectories
       (question patterns, tool sequences)
    2. LLM-driven synthesis: fill template slots with valid geospatial entities
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client
        self._mined_templates: list[dict] = []

    def mine_templates(self, trajectories: list[Trajectory]) -> list[dict]:
        """Extract task templates from training data.

        Groups trajectories by tool sequence pattern, counts frequency,
        and returns reusable templates with metadata.
        """
        # Group by tool_sequence tuple
        seq_counter: Counter = Counter()
        seq_questions: dict[tuple, list[str]] = {}

        for traj in trajectories:
            if not traj.tools_called:
                continue
            seq_key = tuple(traj.tools_called[:4])   # up to 4 tools per template
            seq_counter[seq_key] += 1
            seq_questions.setdefault(seq_key, []).append(traj.question[:200])

        # Keep top-20 most common sequences
        templates = []
        for seq, count in seq_counter.most_common(20):
            questions = seq_questions.get(seq, [])
            task_type = self._infer_task_type_from_seq(list(seq))
            templates.append({
                "tool_sequence": list(seq),
                "task_type": task_type,
                "frequency": count,
                "example_questions": questions[:3],
                "pattern": self._derive_pattern(questions[:3]) if questions else "",
            })

        self._mined_templates = templates
        logger.info(f"Mined {len(templates)} task templates from {len(trajectories)} trajectories")
        return templates

    def generate_tasks(self, template: dict, n: int = 5) -> list[dict]:
        """Generate N new task instances from a template.

        First tries LLM synthesis; falls back to rule-based slot filling.
        """
        tasks = []

        if self._llm:
            try:
                questions = self._llm.synthesize_task_variants(template, n=n)
                for q in questions[:n]:
                    if len(q) > 20:
                        tasks.append({
                            "question": q,
                            "tool_sequence": template.get("tool_sequence", []),
                            "task_type": template.get("task_type", "unknown"),
                            "source": "generated",
                        })
            except Exception as e:
                logger.warning(f"LLM task synthesis failed: {e}")

        # Fill with rule-based generation if needed
        while len(tasks) < n:
            task_type = template.get("task_type", "general_qa")
            if task_type in _GEO_TASK_TEMPLATES:
                tmpl = _GEO_TASK_TEMPLATES[task_type]
                q = self._fill_template_slots(tmpl, tasks)
                if q:
                    tasks.append({
                        "question": q,
                        "tool_sequence": tmpl["tool_sequence"],
                        "task_type": task_type,
                        "source": "rule_based",
                    })
            else:
                break

        return tasks[:n]

    def get_all_templates(self) -> list[dict]:
        """Return mined templates + built-in templates."""
        builtin = [
            {**tmpl, "task_type": tt, "frequency": 0, "example_questions": [], "pattern": tmpl["pattern"]}
            for tt, tmpl in _GEO_TASK_TEMPLATES.items()
        ]
        return self._mined_templates + builtin

    def _fill_template_slots(self, tmpl: dict, existing_tasks: list[dict]) -> Optional[str]:
        """Fill a template with valid slot values."""
        import random
        pattern = tmpl.get("pattern", "")
        slots = tmpl.get("slots", {})
        location = random.choice(_GEO_LOCATIONS)
        filled = pattern.replace("{location}", location).replace("{area}", location).replace("{landmark}", location)

        for slot_name, values in slots.items():
            if f"{{{slot_name}}}" in filled:
                val = random.choice(values)
                filled = filled.replace(f"{{{slot_name}}}", val, 1)

        # If still has unfilled slots, skip
        if re.search(r"\{[a-z_]+\}", filled):
            return None

        # Avoid duplicates
        existing_qs = {t.get("question", "") for t in existing_tasks}
        if filled in existing_qs:
            return None

        return filled

    def _infer_task_type_from_seq(self, tool_seq: list[str]) -> str:
        seq_str = " ".join(tool_seq)
        if "raster_diff" in seq_str or "calculate_index" in seq_str:
            return "change_detection"
        if "add_pois_layer" in seq_str or "compute_route" in seq_str:
            return "poi_routing"
        if "remotesam" in seq_str or "sam2" in seq_str or "strip_rcnn" in seq_str:
            return "segmentation"
        if "vlm_analyze" in seq_str:
            return "classification"
        if "calculate_index" in seq_str:
            return "index_calculation"
        return "general_qa"

    def _derive_pattern(self, questions: list[str]) -> str:
        if not questions:
            return ""
        # Return shortest question as approximate pattern
        return min(questions, key=len)
