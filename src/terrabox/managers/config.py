"""Configuration helpers shared by tool service managers."""
from __future__ import annotations

import os
from typing import Any


_yaml_cache: dict[str, Any] | None = None


def load_raw_yaml() -> dict[str, Any]:
    """Return the raw agent_config.yaml contents when present."""
    global _yaml_cache
    if _yaml_cache is not None:
        return _yaml_cache

    path = os.environ.get("AGENT_CONFIG_PATH", "agent_config.yaml")
    _yaml_cache = {}
    if os.path.exists(path):
        try:
            import yaml

            with open(path) as f:
                _yaml_cache = yaml.safe_load(f) or {}
        except Exception:
            pass
    return _yaml_cache
