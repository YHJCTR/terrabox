"""Compute toolkit: Calculator / Solver / Plot tools.

These restore the original OpenEarth computation tools that the
``data/newdata`` conversion collapsed into ``ipython.execute``. By exposing
them as distinct, executable Terrabox tools, tool-flow datasets keep faithful
tool identity (so ``ipython`` no longer artificially dominates the
distribution).

Behaviour is ported 1:1 from OpenEarthAgent's tool workers:
  - ``compute.calculator`` : safe ``eval`` over a math namespace (Calculator)
  - ``compute.solver``      : exec Python/SymPy code defining ``solution()`` (Solver)
  - ``compute.plot``        : exec matplotlib code defining ``solution()`` (Plot)
"""
from __future__ import annotations

import math
import os
import uuid
from typing import Any

from ..core.registry import ToolSpec


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
_FORBIDDEN_MODULES = (
    "subprocess", "multiprocessing", "threading", "socket",
    "psutil", "resource", "ctypes",
)
_FORBIDDEN_PATTERNS = (
    "os.system", "os.popen", "os.spawn", "os.fork", "os.exec",
    "os._exit", "os.kill", "shutil.rmtree",
)


def _has_forbidden(code: str) -> bool:
    for module in _FORBIDDEN_MODULES:
        if f"import {module}" in code or f"from {module}" in code:
            return True
    return any(pattern in code for pattern in _FORBIDDEN_PATTERNS)


def _strip_code_fences(code: str) -> str:
    """Mirror the OpenEarth Solver/Plot fence + ``python`` prefix stripping."""
    code = code.strip()
    if code.startswith("```python") and code.endswith("```"):
        code = code.split("```python", 1)[1].rsplit("```", 1)[0]
    elif code.startswith("```") and code.endswith("```"):
        code = code.split("```", 1)[1].rsplit("```", 1)[0]
    stripped = code.lstrip()
    if stripped.startswith("python"):
        code = stripped[len("python"):]
    return code


class _GenericRuntime:
    """Isolated exec/eval namespace (ported from OpenEarth workers)."""

    def __init__(self, headers: tuple[str, ...] = ()):
        self._g: dict[str, Any] = {}
        for header in headers:
            exec(header, self._g, self._g)

    def exec_code(self, code: str) -> None:
        exec(code, self._g, self._g)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._g, self._g)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------
def _safe_eval(expression: str) -> Any:
    math_methods = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed: dict[str, Any] = {
        "max": max, "min": min, "round": round, "sum": sum,
        "abs": abs, "pow": pow, "len": len,
        **math_methods,
    }
    allowed["__builtins__"] = None
    return eval(expression, allowed, allowed)


def calculator_handler(arguments: dict, context: dict | None = None, account=None) -> str:
    """Evaluate a Python math expression and return the result as text."""
    expression = arguments.get("expression")
    if not expression:
        return "Error in calculator: Missing required parameter 'expression'"
    if _has_forbidden(str(expression)):
        return "Error in calculator: expression contains forbidden operations"
    try:
        return f"{_safe_eval(str(expression))}"
    except Exception as exc:  # noqa: BLE001 - surface error text like the original
        return f"Error in calculator: {exc}"


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def solver_handler(arguments: dict, context: dict | None = None, account=None) -> str:
    """Exec Python/SymPy code defining ``solution()`` and return its string result."""
    code = arguments.get("command")
    if not code:
        return "Error in Solver: Missing required parameter 'command'"
    code = _strip_code_fences(str(code))
    if _has_forbidden(code):
        return "Error in Solver: code contains forbidden operations"
    try:
        runtime = _GenericRuntime(headers=("from sympy import symbols, Eq, solve",))
        runtime.exec_code(code)
        result = runtime.eval_code("solution()")
        if result is None:
            return "Error in Solver: Execution returned None"
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Error in Solver: {exc}"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _plot_output_dir() -> str:
    out = os.environ.get("TERRABOX_PLOT_OUTPUT_DIR", "").strip()
    if not out:
        out = os.path.join(os.getcwd(), "tmp", "plots")
    os.makedirs(out, exist_ok=True)
    return out


def plot_handler(arguments: dict, context: dict | None = None, account=None):
    """Exec matplotlib code defining ``solution()`` returning a Figure; save it."""
    code = arguments.get("command")
    if not code:
        return "Error in Plot: Missing required parameter 'command'"
    code = _strip_code_fences(str(code))
    if _has_forbidden(code):
        return "Error in Plot: code contains forbidden operations"
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        runtime = _GenericRuntime(headers=("import matplotlib.pyplot as plt",))
        runtime.exec_code(code)
        figure = runtime.eval_code("solution()")
        if not isinstance(figure, Figure):
            return "Error in Plot: solution() must return a matplotlib Figure"
        fname = f"plot_{uuid.uuid4().hex[:8]}.png"
        out_path = os.path.join(_plot_output_dir(), fname)
        figure.savefig(out_path, format="png")
        return {"text": f"Plot saved to {fname}", "image_path": out_path}
    except Exception as exc:  # noqa: BLE001
        return f"Error in Plot: {exc}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def setup(registrar):
    """Register the compute toolkit (Calculator / Solver / Plot)."""
    registrar.toolkit(
        name="compute",
        description="General computation tools: math expression calculator, "
        "SymPy code solver, and matplotlib plotter.",
        version="1.0.0",
    )

    registrar.tool(
        ToolSpec(
            slug="compute.calculator",
            name="Calculator",
            description=(
                "Evaluate a Python math expression (operators, math functions, "
                "and built-ins min/max/round/sum/abs/pow). Returns the numeric result. "
                "Example: {\"expression\": \"round(sqrt((1-2)**2+(3-4)**2)*0.5, 2)\"}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python math expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
        ),
        calculator_handler,
    )

    registrar.tool(
        ToolSpec(
            slug="compute.solver",
            name="Solver",
            description=(
                "Execute Python/SymPy code that defines a solution() function "
                "returning a string result. Use for symbolic math and multi-step "
                "numeric computation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Python code defining solution(); may be wrapped in a markdown code block.",
                    }
                },
                "required": ["command"],
            },
        ),
        solver_handler,
    )

    registrar.tool(
        ToolSpec(
            slug="compute.plot",
            name="Plot",
            description=(
                "Execute Python/matplotlib code that defines a solution() function "
                "returning a matplotlib Figure; saves and returns the plot path."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Python plotting code defining solution() that returns a matplotlib Figure.",
                    }
                },
                "required": ["command"],
            },
        ),
        plot_handler,
    )
