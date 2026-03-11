"""Convert CoreRegistry tools into LangChain StructuredTools for Agent use."""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import create_model

from ..core.registry import registry

logger = logging.getLogger(__name__)

# Mapping from JSON Schema types to Python types
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_schema_to_pydantic(slug: str, schema: dict[str, Any]):
    """Dynamically build a Pydantic model from a JSON Schema properties block."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        py_type = _TYPE_MAP.get(prop.get("type", "string"), str)
        if name in required:
            fields[name] = (py_type, ...)
        else:
            default = prop.get("default", None)
            fields[name] = (Optional[py_type], default)

    if not fields:
        return None

    model_name = "Args_" + slug.replace(".", "_").replace("-", "_")
    return create_model(model_name, **fields)


def _make_tool_func(handler, user):
    """Return a callable that forwards **kwargs as a dict to the handler."""
    def _call(**kwargs):
        try:
            result = handler(kwargs, context=None, account=user)
            return str(result)
        except Exception as e:
            logger.warning(f"Tool execution error: {e}")
            return f"Tool execution error: {e}"
    return _call


def build_langchain_tools(user) -> list[StructuredTool]:
    """
    Wrap every tool registered in CoreRegistry as a LangChain StructuredTool.
    The agent LLM can call any of these tools autonomously.
    """
    tools = []
    for spec in registry.list_tools():
        handler = registry.get_handler(spec.slug)
        if handler is None:
            continue

        args_schema = _json_schema_to_pydantic(spec.slug, spec.parameters)
        lc_tool = StructuredTool.from_function(
            func=_make_tool_func(handler, user),
            # LangChain tool names must not contain dots
            name=spec.slug.replace(".", "__"),
            description=spec.description,
            args_schema=args_schema,
        )
        tools.append(lc_tool)
        logger.debug(f"Registered LangChain tool: {lc_tool.name}")

    logger.info(f"Built {len(tools)} LangChain tools from CoreRegistry")
    return tools
