"""IntentParser: extract structured intent from a geospatial user query."""
from __future__ import annotations

import re
from typing import Optional


# Keyword rules for task type classification
_TASK_KEYWORDS: dict[str, list[str]] = {
    "change_detection": [
        "change", "difference", "compare", "before", "after", "between",
        "temporal", "trend", "growth", "loss", "increase", "decrease",
        "nbr", "ndvi change", "burn severity",
    ],
    "poi_routing": [
        "nearest", "closest", "route", "distance", "station", "hospital",
        "school", "restaurant", "poi", "point of interest", "facility",
        "navigate", "path",
    ],
    "segmentation": [
        "detect", "segment", "identify", "locate", "find object",
        "count", "measure", "extract", "delineate", "mask",
    ],
    "classification": [
        "classify", "what type", "land use", "land cover", "category",
        "label", "recognize", "identify type",
    ],
    "index_calculation": [
        "calculate", "compute", "ndvi", "ndwi", "ndbi", "nbr",
        "tvdi", "index", "spectral", "radiometric",
    ],
    "area_analysis": [
        "area", "coverage", "percentage", "proportion", "statistics",
        "summarize", "total area", "square",
    ],
    "general_qa": [],
}

# Domain classification by expected toolkit
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "geo_perception": [
        "image", "photo", "satellite", "aerial", "detect", "segment",
        "visual", "pixel", "gsd", "band",
    ],
    "spatial": [
        "boundary", "area", "poi", "route", "distance", "location",
        "geospatial", "region", "park", "city", "country",
    ],
    "raster": [
        "raster", "ndvi", "index", "band", "spectral", "sentinel",
        "landsat", "stac", "tif", "geotiff", "time series",
    ],
    "code": [
        "calculate", "compute", "statistics", "average", "sum",
        "python", "formula", "percentage",
    ],
}


class IntentParser:
    """Parse a user query into a structured Intent dictionary.

    Intent structure (IEU framework):
    {
        task_type: str,          # one of _TASK_KEYWORDS keys
        target_domain: str,      # primary toolkit domain
        entities: dict,          # extracted geo entities
        complexity: str,         # "simple" | "multi_step" | "complex"
        requires_image: bool,
    }
    """

    def parse(self, query: str, images: Optional[list[str]] = None) -> dict:
        q_lower = query.lower()
        task_type = self._classify_task_type(q_lower)
        domain = self._classify_domain(q_lower)
        entities = self._extract_entities(query)
        complexity = self._estimate_complexity(q_lower)
        requires_image = bool(images) or self._mentions_image(q_lower)

        return {
            "task_type": task_type,
            "target_domain": domain,
            "entities": entities,
            "complexity": complexity,
            "requires_image": requires_image,
            "raw_query": query[:200],   # keep for BM25 matching
        }

    def _classify_task_type(self, q: str) -> str:
        best = "general_qa"
        best_score = 0
        for task_type, keywords in _TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in q)
            if score > best_score:
                best_score = score
                best = task_type
        return best

    def _classify_domain(self, q: str) -> str:
        best = "multi"
        best_score = 0
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in q)
            if score > best_score:
                best_score = score
                best = domain
        return best

    def _extract_entities(self, query: str) -> dict:
        entities: dict = {}

        # Extract year patterns
        years = re.findall(r"\b(20\d{2}|19\d{2})\b", query)
        if years:
            entities["years"] = years

        # Extract month names
        months = re.findall(
            r"\b(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\b",
            query, re.IGNORECASE,
        )
        if months:
            entities["months"] = months

        # Extract numeric measurements
        measurements = re.findall(r"\b\d+(?:\.\d+)?\s*(?:m|km|meter|kilometer|pixel)\b", query, re.IGNORECASE)
        if measurements:
            entities["measurements"] = measurements

        return entities

    def _estimate_complexity(self, q: str) -> str:
        multi_indicators = ["and then", "first", "second", "step", "after that",
                            "following", "subsequently", "next"]
        count = sum(1 for ind in multi_indicators if ind in q)
        if count >= 3:
            return "complex"
        if count >= 1:
            return "multi_step"
        return "simple"

    def _mentions_image(self, q: str) -> bool:
        return any(kw in q for kw in ["image", "photo", "satellite", "aerial", "picture"])

    def similarity(self, intent1: dict, intent2: dict) -> float:
        """Compute similarity score between two intents (0.0 – 1.0).

        Weighted: task_type (0.4) + domain (0.3) + requires_image (0.2) + complexity (0.1)
        """
        score = 0.0
        if intent1.get("task_type") == intent2.get("task_type"):
            score += 0.4
        if intent1.get("target_domain") == intent2.get("target_domain"):
            score += 0.3
        if intent1.get("requires_image") == intent2.get("requires_image"):
            score += 0.2
        if intent1.get("complexity") == intent2.get("complexity"):
            score += 0.1
        return score
