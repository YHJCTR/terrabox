"""Schema-aware coercion for tool inputs submitted from GUI forms."""

from __future__ import annotations

import ast
import json
from typing import Any, Mapping


def _parse_structured_text(value: str) -> Any:
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value


def _coerce_scalar(value: Any, schema_type: str | None) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if schema_type == "number":
        try:
            return float(text)
        except ValueError:
            return value
    if schema_type == "integer":
        try:
            return int(text)
        except ValueError:
            return value
    if schema_type == "boolean":
        lowered = text.lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return value


def _coerce_value(value: Any, schema: Mapping[str, Any] | None) -> Any:
    schema = schema or {}
    schema_type = schema.get("type")

    if schema_type == "object":
        if isinstance(value, str):
            parsed = _parse_structured_text(value)
            if parsed is not value:
                value = parsed
        if isinstance(value, Mapping):
            properties = schema.get("properties") or {}
            return {
                key: _coerce_value(item, properties.get(key))
                for key, item in value.items()
            }
        return value

    if schema_type == "array":
        item_schema = schema.get("items") or {}
        if isinstance(value, str):
            parsed = _parse_structured_text(value)
            if isinstance(parsed, (list, tuple)):
                value = list(parsed)
            else:
                value = [line for line in value.splitlines() if line.strip()]

        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
            parsed = _parse_structured_text(value[0])
            if isinstance(parsed, (list, tuple)):
                value = list(parsed)

        if isinstance(value, list):
            return [_coerce_value(item, item_schema) for item in value]
        return value

    return _coerce_scalar(value, schema_type)


def coerce_tool_inputs(inputs: Mapping[str, Any] | None, parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Coerce string form values into the shapes declared by a tool schema."""
    properties = dict((parameters or {}).get("properties") or {})
    return {
        key: _coerce_value(value, properties.get(key))
        for key, value in dict(inputs or {}).items()
    }
