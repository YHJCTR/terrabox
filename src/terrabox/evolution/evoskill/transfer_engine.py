"""CrossDomainTransfer: transfer skills between geospatial tool domains.

From: EvoSkill (arXiv 2603.02766):
Skills learned for one domain can transfer to other domains by adapting
tool slugs and parameter hints while preserving the abstract skill structure.
Zero-shot skill transfer: e.g., skills for spatial tasks can improve
perception tasks with similar logical structure.
"""
from __future__ import annotations

import logging
from typing import Optional

from .skill_module import SkillModule

logger = logging.getLogger(__name__)

# Domain clustering — maps domain name to representative tool slugs
DOMAIN_TOOLS: dict[str, list[str]] = {
    "geo_perception": [
        "geo_perception.vlm_analyze",
        "geo_perception.sam2_segment",
        "geo_perception.remotesam_segment",
        "geo_perception.remoteclip_analysis",
        "geo_perception.strip_rcnn_detect",
    ],
    "spatial": [
        "osm_gis.get_area_boundary",
        "osm_gis.add_pois_layer",
        "osm_gis.compute_route_dist",
        "geobasic.area",
        "geobasic.distance",
        "geoanalysis.buffer",
        "geoanalysis.dissolve",
        "geoanalysis.intersection",
    ],
    "raster": [
        "georaster.calculate_index",
        "georaster.raster_diff",
        "georaster.statistics",
        "stac_basic.search",
    ],
    "code": [
        "ipython_code.execute",
        "bash.execute",
    ],
}

# Analogous tool pairs for cross-domain adaptation
# (source_slug → target_slug for various transfers)
_ANALOGIES: dict[str, dict[str, str]] = {
    # spatial → perception: spatial workflows adapted for image-based queries
    "spatial→geo_perception": {
        "osm_gis.get_area_boundary": "geo_perception.vlm_analyze",
        "osm_gis.add_pois_layer": "geo_perception.remotesam_segment",
        "osm_gis.compute_route_dist": "ipython_code.execute",
    },
    # raster → perception: index calculations → visual features
    "raster→geo_perception": {
        "georaster.calculate_index": "geo_perception.remoteclip_analysis",
        "georaster.raster_diff": "geo_perception.sam2_segment",
        "stac_basic.search": "geo_perception.vlm_analyze",
    },
    # perception → spatial: feature extraction → spatial queries
    "geo_perception→spatial": {
        "geo_perception.vlm_analyze": "osm_gis.get_area_boundary",
        "geo_perception.remotesam_segment": "osm_gis.add_pois_layer",
        "ipython_code.execute": "osm_gis.compute_route_dist",
    },
}


class CrossDomainTransfer:
    """Transfer skills from one geospatial domain to another.

    Transfer preserves:
    - Skill name, description, trigger condition, preconditions
    - Abstract logical structure (sequence length, roles)

    Adapts:
    - tool_sequence: map source slugs to analogous target slugs
    - parameter_hints: clear domain-specific hints, add generic guidance
    - domain label
    """

    def adapt_skill(
        self,
        skill: SkillModule,
        target_domain: str,
    ) -> Optional[SkillModule]:
        """Return an adapted copy of the skill for the target domain.

        Returns None if no adaptation mapping exists.
        """
        import uuid

        source_domain = skill.domain
        if source_domain == target_domain:
            return skill   # no adaptation needed

        transfer_key = f"{source_domain}→{target_domain}"
        analogy_map = _ANALOGIES.get(transfer_key)

        if analogy_map is None:
            logger.debug(f"No analogy map for {transfer_key}")
            return None

        new_seq = []
        for slug in skill.tool_sequence:
            adapted = analogy_map.get(slug)
            if adapted:
                new_seq.append(adapted)
            elif any(slug in tools for tools in DOMAIN_TOOLS.values()):
                # Keep domain-agnostic tools (e.g. ipython_code.execute)
                new_seq.append(slug)

        if not new_seq:
            return None

        # Clear domain-specific parameter hints; add generic guidance
        new_hints = {slug: "adapt parameters for " + target_domain for slug in new_seq[:3]}

        return SkillModule(
            id=str(uuid.uuid4()),
            name=skill.name + f"_transfer_{target_domain}",
            description=skill.description + f" [transferred from {source_domain}]",
            trigger_condition=skill.trigger_condition,
            tool_sequence=new_seq,
            parameter_hints=new_hints,
            preconditions=skill.preconditions,
            domain=target_domain,
            validation_f1_delta=0.0,   # needs re-evaluation
            generality_score=0.0,
        )

    def find_transferable_skills(
        self,
        skills: list[SkillModule],
        target_domain: str,
    ) -> list[SkillModule]:
        """Return transferred versions of skills applicable to target_domain."""
        transferred = []
        for skill in skills:
            if skill.domain == target_domain:
                continue
            adapted = self.adapt_skill(skill, target_domain)
            if adapted is not None:
                transferred.append(adapted)
        return transferred

    @staticmethod
    def classify_domain(question: str, images: list[str]) -> str:
        """Infer the most likely domain from a user query."""
        q = question.lower()
        if images or any(kw in q for kw in ["image", "satellite", "aerial", "detect", "segment"]):
            return "geo_perception"
        if any(kw in q for kw in ["ndvi", "ndwi", "ndbi", "nbr", "index", "raster"]):
            return "raster"
        if any(kw in q for kw in ["boundary", "poi", "route", "area", "distance", "nearest"]):
            return "spatial"
        if any(kw in q for kw in ["calculate", "compute", "python", "statistics"]):
            return "code"
        return "multi"
