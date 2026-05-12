"""
codegen toolkit — LLM-driven dynamic tool generation with Docker sandbox validation.

Registers one tool: codegen.generate_and_run

When the agent encounters a task that no existing tool can handle, it calls this tool
with a description of the needed capability and the input data. The tool:
  1. Asks the LLM to write a self-contained Python script
  2. Runs the script inside a Docker sandbox (isolated, no network, memory-limited)
  3. On error: feeds the traceback back to the LLM and retries up to MAX_RETRIES times
  4. Returns the script output (or the final error if all retries exhausted)

LLM endpoint: dynamically allocated via the Docker GPU/port resource scheduler
Sandbox image: SANDBOX_IMAGE (pulled on first use if not present)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Any

import requests

from ..core.registry import ToolSpec
from ..agent.config import load_config
from ..managers.resource_allocator import (
    ResourceLease,
    acquire_docker_lease,
    labels_for_lease,
    record_service_event,
    remove_container_if_exists,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
CODEGEN_LLM_SERVICE: str = "codegen-llm"
CODEGEN_LLM_CONTAINER_BASE: str = "terrabox-codegen-llm"
CODEGEN_LLM_INTERNAL_PORT: int = 8000
CODEGEN_LLM_HOST: str = "127.0.0.1"
LLM_MODEL: str = "/model"              # model name as served by vLLM
SANDBOX_IMAGE: str = "terrabox/codegen-sandbox:latest"  # lightweight sandbox image (~1.9GB)
SANDBOX_CONTAINER: str = "terrabox-codegen-sandbox"  # preferred: exec into pre-warmed container (fast)
MAX_RETRIES: int = 3                    # max LLM fix attempts on sandbox error
SANDBOX_TIMEOUT_S: int = 30            # per-run Docker timeout (seconds)
LLM_TIMEOUT_S: int = 60               # LLM HTTP request timeout (seconds)
LLM_MAX_TOKENS: int = 1024
CODEGEN_LLM_STARTUP_RETRIES: int = 120
CODEGEN_LLM_STARTUP_POLL_S: float = 5.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_code(text: str) -> str:
    """
    Extract executable Python code from raw LLM output in a model-agnostic way.

    Strategy (tried in order):
    1. Pull content from the first ```python ... ``` or ``` ... ``` fence.
       Most models wrap code in fences regardless of whether they think or not.
    2. Strip all XML-style reasoning blocks (<tag>...</tag>) that appear before
       the code — covers Qwen3 <think>, DeepSeek <think>, and any future tags.
    3. Discard leading lines that look like prose (no Python token at the start)
       until we hit a line that starts a Python statement.

    We intentionally do NOT enumerate model-specific tag names; instead we detect
    code by its structure.
    """
    import re

    # ── Pass 1: extract from markdown code fence ──────────────────────────────
    fence_match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # ── Pass 2: strip any XML-style blocks before the code ────────────────────
    # Matches <anything>...</anything> spanning multiple lines (non-greedy).
    # This handles <think>, <reasoning>, <reflection>, <scratchpad>, etc.
    cleaned = re.sub(r'<[a-zA-Z_][a-zA-Z0-9_]*>.*?</[a-zA-Z_][a-zA-Z0-9_]*>',
                     '', text, flags=re.DOTALL)
    cleaned = cleaned.strip()

    # ── Pass 3: drop leading prose lines until we see Python ─────────────────
    # A line "starts Python" if it begins with a known keyword or pattern.
    _PY_START = re.compile(
        r'^('
        r'import\s|from\s|def\s|class\s|#|@|'   # common starters
        r'[a-zA-Z_]\w*\s*=|'                     # assignment
        r'if\s|for\s|while\s|with\s|try:|'       # control flow
        r'print\(|raise\s|return\s'
        r')'
    )
    lines = cleaned.splitlines()
    for i, line in enumerate(lines):
        if _PY_START.match(line.strip()):
            return '\n'.join(lines[i:]).strip()

    # ── Fallback: return whatever we have after XML stripping ─────────────────
    return cleaned


# ── LLM helpers ──────────────────────────────────────────────────────────────

def _codegen_llm_is_healthy(port: int) -> bool:
    try:
        resp = requests.get(
            f"http://{CODEGEN_LLM_HOST}:{port}/health",
            timeout=1,
            proxies={"http": None, "https": None},
        )
        return resp.status_code == 200
    except Exception:
        return False


def _start_codegen_llm_service(config) -> tuple[str, ResourceLease]:
    """Start a dedicated, temporary vLLM service for code generation."""
    image = os.environ.get("CODEGEN_LLM_DOCKER_IMAGE", getattr(config, "local_llm_docker_image", "terrabox/agent-llm:latest"))
    model_path = os.environ.get("CODEGEN_LLM_MODEL_PATH", getattr(config, "local_llm_model_path", ""))
    tensor_parallel = str(os.environ.get("CODEGEN_LLM_TENSOR_PARALLEL", getattr(config, "local_llm_tensor_parallel", 1)))
    max_model_len = str(os.environ.get("CODEGEN_LLM_MAX_MODEL_LEN", getattr(config, "local_llm_max_model_len", 24576)))
    base_port = int(os.environ.get("CODEGEN_LLM_BASE_PORT", getattr(config, "local_llm_port", 9100)))
    fallback_gpu = os.environ.get("CODEGEN_LLM_GPU_DEVICES", getattr(config, "local_llm_gpu_devices", "0"))

    lease = acquire_docker_lease(
        service=CODEGEN_LLM_SERVICE,
        image=image,
        container_base=CODEGEN_LLM_CONTAINER_BASE,
        host=CODEGEN_LLM_HOST,
        base_port=base_port,
        internal_port=CODEGEN_LLM_INTERNAL_PORT,
        gpu_count=int(tensor_parallel),
        min_free_mib=int(os.environ.get("CODEGEN_LLM_MIN_FREE_MIB", os.environ.get("AGENT_LLM_MIN_FREE_MIB", "16000"))),
        fallback_gpu_devices=fallback_gpu,
        gpu_env_var="CODEGEN_LLM_GPU_DEVICES",
    )
    remove_container_if_exists(lease.container_name, service=CODEGEN_LLM_SERVICE, reason="before_docker_run")

    cmd = [
        "docker", "run", "-d",
        "--name", lease.container_name,
        *labels_for_lease(lease),
        "--gpus", "all",
        "-e", f"CUDA_VISIBLE_DEVICES={lease.gpu_devices}",
        "-p", f"{lease.port}:{lease.internal_port}",
        "-v", f"{model_path}:/model:ro",
        "--shm-size=8g",
        image,
        "--model", "/model",
        "--trust-remote-code",
        "--host", "0.0.0.0",
        "--port", str(CODEGEN_LLM_INTERNAL_PORT),
        "--tensor-parallel-size", tensor_parallel,
        "--max-model-len", max_model_len,
        "--gpu-memory-utilization", os.environ.get("CODEGEN_LLM_GPU_MEMORY_UTILIZATION", "0.85"),
        "--enforce-eager",
    ]

    logger.info(
        "[codegen] Starting temporary LLM service (GPU: %s, port: %s, model: %s)",
        lease.gpu_devices,
        lease.port,
        model_path,
    )
    record_service_event({"event": "start_requested", "service": CODEGEN_LLM_SERVICE, "lease": lease.__dict__})
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        subprocess.run(["docker", "rm", "-f", lease.container_name], capture_output=True)
        record_service_event({
            "event": "start_failed",
            "service": CODEGEN_LLM_SERVICE,
            "stderr": result.stderr,
            "lease": lease.__dict__,
        })
        raise RuntimeError(f"Failed to start codegen LLM container.\nstderr: {result.stderr}")

    for i in range(CODEGEN_LLM_STARTUP_RETRIES):
        if _codegen_llm_is_healthy(lease.port):
            record_service_event({"event": "ready", "service": CODEGEN_LLM_SERVICE, "lease": lease.__dict__})
            return f"http://{lease.host}:{lease.port}/v1", lease
        if i % 6 == 0:
            logger.info("[codegen] Waiting for temporary LLM service... (%ss elapsed)", i * CODEGEN_LLM_STARTUP_POLL_S)
        time.sleep(CODEGEN_LLM_STARTUP_POLL_S)

    _stop_codegen_llm_service(lease)
    raise TimeoutError(f"Codegen LLM service failed to start within timeout. Check: docker logs {lease.container_name}")


def _stop_codegen_llm_service(lease: ResourceLease) -> None:
    logger.info("[codegen] Stopping temporary LLM container %s", lease.container_name)
    subprocess.run(["docker", "rm", "-f", lease.container_name], capture_output=True, timeout=10)
    record_service_event({"event": "stopped", "service": CODEGEN_LLM_SERVICE, "container": lease.container_name})


def _call_llm(api_base: str, messages: list[dict], no_thinking: bool = False) -> str:
    """POST to local vLLM and return the assistant reply text.

    no_thinking=True disables the thinking-token output for models that support it
    (e.g. Qwen3). The model still reasons internally; this only prevents <think>
    blocks and interleaved prose from contaminating the response. Ignored by models
    that do not recognise chat_template_kwargs.
    """
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload: dict = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": LLM_MAX_TOKENS,
    }
    if no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=LLM_TIMEOUT_S,
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc


def _generate_code(api_base: str, description: str, input_data: str) -> str:
    """Ask LLM to write a self-contained Python script for the given task."""
    system = (
        "You are an expert Python programmer. "
        "Write concise, correct Python scripts. "
        "Always wrap your code in a ```python ... ``` fence — nothing outside the fence."
    )
    user = f"""Write a complete, self-contained Python script that does the following:

{description}

CRITICAL requirements:
- The variable INPUT_DATA is ALREADY DEFINED at the top of the script as a JSON string — do NOT redefine it, just call json.loads(INPUT_DATA) to parse it
- Write TOP-LEVEL code only — do NOT wrap everything in a function that is never called
- Available libraries: standard library (json, math, statistics, etc.) PLUS numpy, scipy, pandas, pillow (PIL), opencv (cv2), shapely, rasterio, pyproj, affine, scikit-learn
- Do NOT import libraries outside the above list (e.g. torch, tensorflow, requests are not available)
- The last statement must be: print(json.dumps(<result>))
- Do not include any input() calls, argparse, or sys.stdin reads
- Match the requested output schema exactly. Use the exact field names requested by the user.
- Do NOT add extra output fields unless the user explicitly asks for them.
- If the user asks for rounding or formatting, apply it in the JSON result, not only in comments.

The script skeleton (fill in the logic):
```python
# INPUT_DATA is already defined above this code
data = json.loads(INPUT_DATA)
# ... your logic here ...
result = {{...}}
print(json.dumps(result))
```

INPUT_DATA value for reference:
{input_data}

Output the complete script inside a ```python ... ``` fence:"""

    raw = _call_llm(
        api_base,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        no_thinking=True,
    )
    return _extract_code(raw)


def _fix_code(api_base: str, original_code: str, error: str, description: str) -> str:
    """Ask LLM to fix code given the sandbox error output."""
    user = f"""The following Python script raised an error when executed.

=== TASK ===
{description}

=== SCRIPT ===
{original_code}

=== ERROR ===
{error}

Fix the script so it runs correctly. Output the corrected script inside a ```python ... ``` fence:"""

    return _extract_code(_call_llm(api_base, [{"role": "user", "content": user}], no_thinking=True))


# ── Docker sandbox ────────────────────────────────────────────────────────────

def _container_is_running(name: str) -> bool:
    """Return True if a Docker container with the given name is currently running."""
    try:
        out = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.State.Running}}", name],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out == "true"
    except subprocess.CalledProcessError:
        return False


def _start_sandbox_container() -> None:
    """Start the sandbox container. Removes any stopped container with the same name first."""
    subprocess.run(["docker", "rm", "-f", SANDBOX_CONTAINER], capture_output=True)
    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", SANDBOX_CONTAINER,
            "--network", "none",
            "--memory", "512m",
            "--cpus", "1.0",
            SANDBOX_IMAGE,
            "sleep", "infinity",
        ],
        check=True, capture_output=True,
    )
    logger.info(f"[codegen] Auto-started sandbox container '{SANDBOX_CONTAINER}'")


def _stop_sandbox_container() -> None:
    """Stop and remove the sandbox container."""
    subprocess.run(["docker", "rm", "-f", SANDBOX_CONTAINER], capture_output=True)
    logger.info(f"[codegen] Auto-stopped sandbox container '{SANDBOX_CONTAINER}'")


def _run_in_sandbox(code: str, input_data: str) -> dict[str, Any]:
    """
    Run `code` in a Docker sandbox and return {"success", "output", "error"}.

    Lifecycle:
      - Container already running (pre-warmed): use it, leave it running after.
      - Container not running: auto-start before execution, auto-stop after.

    INPUT_DATA is injected at the top of the script so generated code can use it.
    """
    full_script = (
        f"import json\n"
        f"INPUT_DATA = {repr(input_data)}\n\n"
        f"{code}\n"
    )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="codegen_"
    )
    pre_warmed = _container_is_running(SANDBOX_CONTAINER)
    try:
        tmp.write(full_script)
        tmp.flush()
        tmp.close()

        if not pre_warmed:
            logger.info(f"[codegen] Container '{SANDBOX_CONTAINER}' not running — auto-starting...")
            _start_sandbox_container()

        container_path = f"/tmp/{os.path.basename(tmp.name)}"
        subprocess.run(
            ["docker", "cp", tmp.name, f"{SANDBOX_CONTAINER}:{container_path}"],
            check=True, capture_output=True,
        )
        try:
            proc = subprocess.run(
                ["docker", "exec", SANDBOX_CONTAINER, "python3", container_path],
                capture_output=True, text=True, timeout=SANDBOX_TIMEOUT_S,
            )
        finally:
            subprocess.run(
                ["docker", "exec", SANDBOX_CONTAINER, "rm", "-f", container_path],
                capture_output=True,
            )

        if proc.returncode == 0:
            return {"success": True, "output": proc.stdout.strip(), "error": ""}
        else:
            return {
                "success": False,
                "output": proc.stdout.strip(),
                "error": proc.stderr.strip() or f"exit code {proc.returncode}",
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": f"Timed out after {SANDBOX_TIMEOUT_S}s"}
    except FileNotFoundError:
        return {"success": False, "output": "", "error": "Docker not found — is Docker installed?"}
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else str(exc.stderr)
        return {"success": False, "output": "", "error": f"Docker error: {err}"}
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        if not pre_warmed:
            _stop_sandbox_container()


# ── Tool handler ──────────────────────────────────────────────────────────────

def codegen_generate_and_run_handler(
    arguments: dict, context: dict, account=None
) -> dict:
    """
    Generate a Python tool via LLM, validate + run it in Docker sandbox.

    arguments:
        description (str): Plain-language description of what the function should do,
                           including expected input format and output format.
        input_data  (str): JSON string of the actual input to pass to the function.
    """
    description: str = arguments.get("description", "").strip()
    input_data: str = arguments.get("input_data", "{}").strip()

    if not description:
        return {"error": "Missing required argument: description", "success": False}

    # Validate input_data is parseable JSON
    try:
        json.loads(input_data)
    except json.JSONDecodeError as exc:
        return {"error": f"input_data is not valid JSON: {exc}", "success": False}

    logger.info(f"[codegen] Generating code for: {description[:80]!r}")

    lease: ResourceLease | None = None
    try:
        api_base, lease = _start_codegen_llm_service(load_config())

        # ── Attempt loop ──────────────────────────────────────────────────────
        code = ""
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if attempt == 1:
                    code = _generate_code(api_base, description, input_data)
                else:
                    logger.info(f"[codegen] Attempt {attempt}: fixing code after error")
                    code = _fix_code(api_base, code, last_error, description)
            except RuntimeError as exc:
                return {"error": str(exc), "success": False, "attempts": attempt}

            logger.info(f"[codegen] Running in Docker sandbox (attempt {attempt}/{MAX_RETRIES})")
            result = _run_in_sandbox(code, input_data)

            if result["success"]:
                logger.info(f"[codegen] Success on attempt {attempt}")
                return {
                    "success": True,
                    "output": result["output"],
                    "code": code,
                    "executed_code": code,
                    "attempts": attempt,
                    "llm_service": {
                        "container_name": lease.container_name,
                        "port": lease.port,
                        "gpu_devices": lease.gpu_devices,
                    },
                    "final_answer_instruction": (
                        "When answering the user, include the sandbox output and also include "
                        "the executed_code field as a fenced python code block."
                    ),
                }

            last_error = result["error"]
            logger.warning(f"[codegen] Attempt {attempt} failed: {last_error[:120]}")

        return {
            "success": False,
            "error": f"All {MAX_RETRIES} attempts failed. Last error: {last_error}",
            "code": code,
            "executed_code": code,
            "attempts": MAX_RETRIES,
            "llm_service": {
                "container_name": lease.container_name,
                "port": lease.port,
                "gpu_devices": lease.gpu_devices,
            },
        }
    finally:
        if lease is not None:
            _stop_codegen_llm_service(lease)


# ── Toolkit registration ──────────────────────────────────────────────────────

def setup(registrar) -> None:
    registrar.toolkit(
        name="codegen",
        description="Dynamic tool generation: LLM writes Python code, Docker sandbox validates and runs it.",
        version="1.0.0",
    )

    registrar.tool(
        ToolSpec(
            slug="codegen.generate_and_run",
            name="Generate and Run Custom Code",
            description=(
                "Use this tool when no existing tool can accomplish the required task. "
                "Describe the computation needed and provide the input data as JSON. "
                "The tool will generate Python code, run it in an isolated Docker sandbox, "
                "and return the result plus the exact executed_code. "
                "When the user is testing, auditing, or asking how the computation was done, "
                "include executed_code in the final answer as a fenced python code block. "
                "Example use cases: custom raster algebra, novel index calculations, "
                "multi-step statistical operations not covered by existing tools."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": (
                            "Plain-language description of the computation to perform. "
                            "Include: what the inputs are, what processing to apply, "
                            "and what the output should look like."
                        ),
                    },
                    "input_data": {
                        "type": "string",
                        "description": (
                            "JSON string containing all input data the generated code needs. "
                            "Example: '{\"values\": [0.8, 0.3, 0.9], \"threshold\": 0.6}'"
                        ),
                    },
                },
                "required": ["description", "input_data"],
            },
        ),
        codegen_generate_and_run_handler,
    )
