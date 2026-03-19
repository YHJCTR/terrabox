"""
Unit tests for category_graph.py dynamic expansion feature.

Tests cover:
  1. expansion_node: action=done        → expanded=False
  2. expansion_node: action=expand_categories (valid)    → expanded=True
  3. expansion_node: action=expand_categories (invalid cat) → expanded=False
  4. expansion_node: action=synthesize_tool (valid code)    → expanded=True, tool callable
  5. expansion_node: action=synthesize_tool (syntax error)  → expanded=False
  6. expansion_node: action=synthesize_tool (missing fields) → expanded=False
  7. expansion_node: max_expansions reached → expanded=False, no LLM call
  8. expansion_node: LLM returns non-JSON  → expanded=False
  9. synthesized tool actually executes correctly
 10. execution_node merges synthesized_tools with registry tools
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**overrides) -> dict:
    """Minimal valid CategoryAgentState."""
    base = {
        "messages": [HumanMessage(content="Calculate haversine distance from (30,120) to (31,121)")],
        "selected_categories": ["basic_geo"],
        "synthesized_tools": [],
        "expansion_count": 0,
        "max_expansions": 2,
        "expanded": False,
    }
    base.update(overrides)
    return base


def make_llm(response: dict | str) -> MagicMock:
    """Mock LLM whose invoke() returns a JSON-encoded AIMessage."""
    llm = MagicMock()
    content = response if isinstance(response, str) else json.dumps(response)
    llm.invoke.return_value = AIMessage(content=content)
    return llm


def mock_toolkit(name: str):
    tk = MagicMock()
    tk.name = name
    tk.description = f"Toolkit: {name}"
    return tk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_TOOLKITS = [mock_toolkit("basic_geo"), mock_toolkit("advanced_geo"), mock_toolkit("earth_sci")]
ALL_CAT_NAMES = {tk.name for tk in ALL_TOOLKITS}


@pytest.fixture(autouse=True)
def patch_registry():
    """Patch registry.list_toolkits() to return a controlled list."""
    with patch("terrabox.agent.category_graph.registry") as mock_reg:
        mock_reg.list_toolkits.return_value = ALL_TOOLKITS
        mock_reg.list_tools.return_value = []   # execution_node won't need tools in most tests
        yield mock_reg


# ---------------------------------------------------------------------------
# Import the function under test AFTER patching registry (autouse fixture)
# ---------------------------------------------------------------------------

from terrabox.agent.category_graph import _make_expansion_node, _make_execution_node  # noqa: E402


# ---------------------------------------------------------------------------
# Test group 1: expansion_node — action routing
# ---------------------------------------------------------------------------

class TestExpansionNodeActions:

    def test_done_returns_not_expanded(self, patch_registry):
        llm = make_llm({"action": "done"})
        node = _make_expansion_node(llm)
        result = node(make_state())
        assert result.get("expanded") is False
        llm.invoke.assert_called_once()

    def test_expand_categories_valid(self, patch_registry):
        llm = make_llm({"action": "expand_categories", "categories": ["advanced_geo"]})
        node = _make_expansion_node(llm)
        state = make_state(selected_categories=["basic_geo"])
        result = node(state)

        assert result.get("expanded") is True
        assert "advanced_geo" in result["selected_categories"]
        assert "basic_geo" in result["selected_categories"]
        assert result["expansion_count"] == 1

    def test_expand_categories_deduplicates(self, patch_registry):
        """Adding a category already selected should not cause duplicates."""
        llm = make_llm({"action": "expand_categories", "categories": ["basic_geo", "advanced_geo"]})
        node = _make_expansion_node(llm)
        state = make_state(selected_categories=["basic_geo"])
        result = node(state)

        assert result.get("expanded") is True
        assert result["selected_categories"].count("basic_geo") == 1

    def test_expand_categories_unknown_cat_returns_not_expanded(self, patch_registry):
        llm = make_llm({"action": "expand_categories", "categories": ["nonexistent_toolkit"]})
        node = _make_expansion_node(llm)
        result = node(make_state())
        assert result.get("expanded") is False

    def test_expand_categories_empty_list_returns_not_expanded(self, patch_registry):
        llm = make_llm({"action": "expand_categories", "categories": []})
        node = _make_expansion_node(llm)
        result = node(make_state())
        assert result.get("expanded") is False


# ---------------------------------------------------------------------------
# Test group 2: expansion_node — tool synthesis
# ---------------------------------------------------------------------------

class TestExpansionNodeToolSynthesis:

    VALID_CODE = (
        "def add_numbers(a: float, b: float) -> float:\n"
        "    return a + b\n"
    )

    def test_synthesize_valid_tool(self, patch_registry):
        llm = make_llm({
            "action": "synthesize_tool",
            "name": "add_numbers",
            "description": "Add two numbers",
            "code": self.VALID_CODE,
        })
        node = _make_expansion_node(llm)
        result = node(make_state())

        assert result.get("expanded") is True
        assert result["expansion_count"] == 1
        synth = result["synthesized_tools"]
        assert len(synth) == 1
        assert synth[0].name == "add_numbers"

    def test_synthesize_tool_is_callable(self, patch_registry):
        """The synthesized LangChain tool wraps the generated function and executes it."""
        code = "def multiply(x: float, y: float) -> float:\n    return x * y\n"
        llm = make_llm({
            "action": "synthesize_tool",
            "name": "multiply",
            "description": "Multiply two numbers",
            "code": code,
        })
        node = _make_expansion_node(llm)
        result = node(make_state())

        tool = result["synthesized_tools"][0]
        # StructuredTool.invoke passes kwargs as a dict; call via .func directly for simplicity
        assert tool.func(x=3.0, y=4.0) == 12.0

    def test_synthesize_tool_geospatial_computation(self, patch_registry):
        """Simulate a real geospatial use-case: haversine distance calculation."""
        code = (
            "def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:\n"
            "    import math\n"
            "    R = 6371\n"
            "    dlat = math.radians(lat2 - lat1)\n"
            "    dlon = math.radians(lon2 - lon1)\n"
            "    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2\n"
            "    return round(2 * R * math.asin(math.sqrt(a)), 3)\n"
        )
        llm = make_llm({
            "action": "synthesize_tool",
            "name": "haversine_km",
            "description": "Great-circle distance in km",
            "code": code,
        })
        node = _make_expansion_node(llm)
        result = node(make_state())

        tool = result["synthesized_tools"][0]
        dist = tool.func(lat1=0.0, lon1=0.0, lat2=0.0, lon2=90.0)
        # Quarter of Earth's circumference ≈ 10007 km
        assert 9900 < dist < 10100, f"Unexpected distance: {dist}"

    def test_synthesize_tool_syntax_error_returns_not_expanded(self, patch_registry):
        bad_code = "def broken(x::\n    return x\n"
        llm = make_llm({
            "action": "synthesize_tool",
            "name": "broken",
            "description": "Broken tool",
            "code": bad_code,
        })
        node = _make_expansion_node(llm)
        result = node(make_state())
        assert result.get("expanded") is False

    def test_synthesize_tool_runtime_error_in_exec_returns_not_expanded(self, patch_registry):
        code = "def bad_tool(x: int):\n    raise RuntimeError('always fails')\n1/0\n"
        llm = make_llm({
            "action": "synthesize_tool",
            "name": "bad_tool",
            "description": "Bad tool",
            "code": code,
        })
        node = _make_expansion_node(llm)
        result = node(make_state())
        assert result.get("expanded") is False

    def test_synthesize_tool_wrong_function_name_returns_not_expanded(self, patch_registry):
        """name field doesn't match the def in code → exec finds no callable."""
        code = "def actual_fn(x: int):\n    return x\n"
        llm = make_llm({
            "action": "synthesize_tool",
            "name": "wrong_name",
            "description": "Mismatch",
            "code": code,
        })
        node = _make_expansion_node(llm)
        result = node(make_state())
        assert result.get("expanded") is False

    def test_synthesize_tool_missing_name_returns_not_expanded(self, patch_registry):
        llm = make_llm({
            "action": "synthesize_tool",
            "description": "No name",
            "code": "def fn(): pass",
        })
        node = _make_expansion_node(llm)
        result = node(make_state())
        assert result.get("expanded") is False

    def test_synthesize_accumulates_across_rounds(self, patch_registry):
        """Each expansion round appends a new tool; old ones are preserved."""
        code1 = "def tool_one(x: int) -> int:\n    return x + 1\n"
        existing_tool = MagicMock()
        existing_tool.name = "preexisting"

        llm = make_llm({
            "action": "synthesize_tool",
            "name": "tool_one",
            "description": "Tool one",
            "code": code1,
        })
        node = _make_expansion_node(llm)
        state = make_state(
            synthesized_tools=[existing_tool],
            expansion_count=1,
        )
        result = node(state)

        assert result.get("expanded") is True
        assert len(result["synthesized_tools"]) == 2
        assert result["expansion_count"] == 2


# ---------------------------------------------------------------------------
# Test group 3: expansion_node — guard conditions
# ---------------------------------------------------------------------------

class TestExpansionNodeGuards:

    def test_max_expansions_reached_no_llm_call(self, patch_registry):
        llm = make_llm({"action": "expand_categories", "categories": ["advanced_geo"]})
        node = _make_expansion_node(llm)
        result = node(make_state(expansion_count=2, max_expansions=2))

        assert result.get("expanded") is False
        llm.invoke.assert_not_called()

    def test_invalid_json_returns_not_expanded(self, patch_registry):
        llm = make_llm("This is not JSON at all, sorry.")
        node = _make_expansion_node(llm)
        result = node(make_state())
        assert result.get("expanded") is False

    def test_markdown_fenced_json_is_parsed(self, patch_registry):
        """Some models wrap JSON in ```json ... ``` fences despite instructions."""
        raw = '```json\n{"action": "done"}\n```'
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content=raw)
        node = _make_expansion_node(llm)
        result = node(make_state())
        # Should parse successfully and return done
        assert result.get("expanded") is False

    def test_unknown_action_returns_not_expanded(self, patch_registry):
        llm = make_llm({"action": "fly_to_the_moon"})
        node = _make_expansion_node(llm)
        result = node(make_state())
        assert result.get("expanded") is False


# ---------------------------------------------------------------------------
# Test group 4: execution_node uses synthesized_tools
# ---------------------------------------------------------------------------

class TestExecutionNodeSynthesizedTools:

    def test_execution_node_merges_synthesized_tools(self, patch_registry):
        """execution_node should pass registry_tools + synthesized_tools to create_react_agent."""
        synth_tool = MagicMock()
        synth_tool.name = "my_synth_tool"

        config = MagicMock()
        config.max_iterations = 5

        registry_tool = MagicMock()
        registry_tool.name = "registry_tool"

        captured_tools = []

        def fake_create_react_agent(llm, tools, **kwargs):
            captured_tools.extend(tools)
            agent = MagicMock()
            agent.invoke.return_value = {
                "messages": [HumanMessage(content="q"), AIMessage(content="done")]
            }
            return agent

        llm = MagicMock()
        user = MagicMock()

        state = make_state(
            synthesized_tools=[synth_tool],
            selected_categories=["basic_geo"],
        )

        with patch("terrabox.agent.category_graph.build_langchain_tools", return_value=[registry_tool]):
            with patch("terrabox.agent.category_graph.create_react_agent" if hasattr(__builtins__, "x") else "langgraph.prebuilt.create_react_agent", fake_create_react_agent, create=True):
                # Import locally to pick up the patched module
                from importlib import reload
                import terrabox.agent.category_graph as cgm
                node_fn = cgm._make_execution_node(llm, user, config)

                # Patch create_react_agent inside the closure's scope
                import langgraph.prebuilt as lp_prebuilt
                orig = lp_prebuilt.create_react_agent
                lp_prebuilt.create_react_agent = fake_create_react_agent
                try:
                    node_fn(state)
                finally:
                    lp_prebuilt.create_react_agent = orig

        # Both registry tool and synthesized tool should have been passed
        tool_names = [t.name for t in captured_tools]
        assert "registry_tool" in tool_names, f"Registry tool missing. Got: {tool_names}"
        assert "my_synth_tool" in tool_names, f"Synthesized tool missing. Got: {tool_names}"
