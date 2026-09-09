"""StableToolBench rollout runner for promptevo prompt experiments.

The runner keeps StableToolBench's official inference pipeline as the source of
truth. It only overrides the static ReAct system prompt for the current
experiment by preloading ``Prompts.ReAct_prompts`` before importing the official
pipeline modules.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from terrabox.agent.llm_provider import pace_remote_llm_request

from .core import (
    DEFAULT_STABLE_TOOLBENCH_ROOT,
    ToolBenchMetricProvider,
    ToolBenchPromptStore,
    ToolBenchTrajectorySource,
)


DEFAULT_TOOLBENCH_EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
DEFAULT_STABLE_GROUPS = (
    "G1_instruction",
    "G1_category",
    "G1_tool",
    "G2_instruction",
    "G2_category",
    "G3_instruction",
)
STABLE_TOOLBENCH_REQUIRED_IMPORTS = (
    "openai",
    "requests",
    "tenacity",
    "termcolor",
    "tqdm",
)


def stable_toolbench_has_core_dependencies(
    stable_root: str = DEFAULT_STABLE_TOOLBENCH_ROOT,
    python_executable: str = sys.executable,
    timeout: int = 90,
) -> tuple[bool, str]:
    """Check whether the official StableToolBench inference pipeline imports."""
    code = r'''
import importlib
import os
import sys

stable_root = os.environ["STABLE_TOOLBENCH_ROOT"]
sys.path.insert(0, os.path.join(stable_root, "toolbench", "inference"))
sys.path.insert(0, stable_root)
missing = []
for name in os.environ["STABLE_TOOLBENCH_REQUIRED_IMPORTS"].split(","):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("missing dependencies: " + "; ".join(missing))
import terrabox.evolution.promptevo.adapters.toolbench.runner as runner
runner._install_stable_import_compat()
from toolbench.inference.Downstream_tasks.rapidapi_multithread import pipeline_runner
print(pipeline_runner.__name__)
'''
    env = os.environ.copy()
    env["STABLE_TOOLBENCH_ROOT"] = stable_root
    env["STABLE_TOOLBENCH_REQUIRED_IMPORTS"] = ",".join(STABLE_TOOLBENCH_REQUIRED_IMPORTS)
    env["PYTHONPATH"] = _toolbench_pythonpath(stable_root, env.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [python_executable, "-c", code],
        cwd=stable_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout).strip()


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def stable_experiment_dir(name: str, output_dir: str = DEFAULT_TOOLBENCH_EXPERIMENTS_DIR) -> str:
    return os.path.abspath(os.path.join(output_dir, name))


def stable_group_input(stable_root: str, group: str) -> str:
    return os.path.join(stable_root, "solvable_queries", "test_instruction", f"{group}.json")


def stable_group_results(experiment_dir: str, group: str) -> str:
    return os.path.join(experiment_dir, "answers", group)


def _toolbench_pythonpath(stable_root: str, existing: str = "") -> str:
    inference_root = os.path.join(stable_root, "toolbench", "inference")
    repo_src = str(Path(__file__).resolve().parents[5])
    parts = [repo_src, os.path.abspath(stable_root), os.path.abspath(inference_root)]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _install_stable_import_compat() -> None:
    """Install import aliases for optional LLM wrappers missing locally.

    The local StableToolBench checkout imports some upstream ToolBench wrappers
    unconditionally, even when the selected backbone is qwen2. Keep the external
    source tree untouched: alias the renamed ChatGPT wrapper and provide stubs
    that fail only if an unavailable backbone is actually selected.
    """
    import importlib
    import types

    try:
        chatgpt_model = importlib.import_module("toolbench.inference.LLM.chatgpt_model")
        sys.modules.setdefault("toolbench.inference.LLM.chatgpt_function_model", chatgpt_model)
        if os.getenv("TERRABOX_TOOLBENCH_REQUEST_PROFILE", "").strip().lower() == "longcat":
            def _longcat_chat_completion_request(
                key,
                base_url,
                messages,
                tools=None,
                tool_choice="required",
                key_pos=None,
                model="LongCat-2.0",
                stop=None,
                process_id=0,
                **args,
            ):
                use_messages = [
                    message
                    for message in messages
                    if not ("valid" in message and message.get("valid") is False)
                ]
                for message in use_messages:
                    message.pop("function_call", None)
                payload = {
                    "model": model,
                    "temperature": 0,
                    "messages": use_messages,
                    "max_tokens": int(os.getenv("TERRABOX_TOOLBENCH_LONGCAT_MAX_TOKENS", "1024")),
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "extra_body": {"thinking": {"type": "disabled"}},
                    **args,
                }
                if stop is not None:
                    payload["stop"] = stop
                if tools is not None:
                    payload["tools"] = tools
                if tool_choice is not None:
                    payload["tool_choice"] = tool_choice
                pace_remote_llm_request(
                    "longcat",
                    workload=os.getenv("TERRABOX_REMOTE_LLM_WORKLOAD", "toolbench"),
                )
                client = OpenAI(base_url=base_url, api_key=key)
                response = client.chat.completions.create(**payload)
                return response.model_dump()

            def _longcat_parse(self, tools, process_id, key_pos=None, **args):
                """StableToolBench's ChatGPT parser without upstream pdb breakpoints."""
                response: dict[str, Any] | None = None
                for attempt in range(self.TRY_TIME):
                    if attempt:
                        time.sleep(min(15, 2 ** attempt))
                    try:
                        response = _longcat_chat_completion_request(
                            self.openai_key,
                            self.base_url,
                            self.conversation_history,
                            tools=tools or None,
                            process_id=process_id,
                            key_pos=key_pos,
                            model=self.model,
                            **args,
                        )
                        total_tokens = int((response.get("usage") or {}).get("total_tokens") or 0)
                        message = dict(response["choices"][0]["message"])
                        if process_id == 0:
                            print(f"[process({process_id})] total tokens: {total_tokens}")
                        return message, 0, total_tokens
                    except Exception as exc:
                        print(
                            f"[process({process_id})] LongCat completion/parse failed "
                            f"({attempt + 1}/{self.TRY_TIME}): {exc!r}",
                            flush=True,
                        )
                return {"role": "assistant", "content": str(response)}, -1, 0

            chatgpt_model.chat_completion_request = _longcat_chat_completion_request
            chatgpt_model.ChatGPTFunction.parse = _longcat_parse
    except Exception:
        pass

    def stub_module(module_name: str, class_name: str) -> None:
        if module_name in sys.modules:
            return
        module = types.ModuleType(module_name)

        class _UnavailableModel:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError(
                    f"{class_name} is not available in this StableToolBench checkout. "
                    "Use qwen2, llama3, or ToolLLaMA_vllm, or install the matching upstream wrapper."
                )

        _UnavailableModel.__name__ = class_name
        setattr(module, class_name, _UnavailableModel)
        sys.modules[module_name] = module

    stub_module("toolbench.inference.LLM.davinci_model", "Davinci")
    stub_module("toolbench.inference.LLM.tool_llama_lora_model", "ToolLLaMALoRA")
    stub_module("toolbench.inference.LLM.tool_llama_model", "ToolLLaMA")
    stub_module("toolbench.inference.LLM.toolace_model", "ToolACEModel")
    try:
        importlib.import_module("toolbench.inference.LLM.retriever")
    except Exception:
        stub_module("toolbench.inference.LLM.retriever", "ToolRetriever")


def _config_with_absolute_paths(config_file: str, stable_root: str) -> dict[str, Any]:
    text = open(config_file, encoding="utf-8").read()
    try:
        import yaml  # type: ignore

        config = yaml.load(text, Loader=yaml.FullLoader) or {}
    except ModuleNotFoundError:
        config = {}
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = value.strip().strip('"').strip("'")
    tool_root = config.get("tool_root_dir")
    if tool_root and not os.path.isabs(str(tool_root)):
        config["tool_root_dir"] = os.path.abspath(os.path.join(stable_root, str(tool_root)))
    return config


def _prepare_query_file(
    input_query_file: str,
    output_dir: str,
    task_ids: Optional[list[str]] = None,
) -> str:
    if not task_ids:
        return input_query_file

    keep = {str(x) for x in task_ids}
    queries = json.load(open(input_query_file, encoding="utf-8"))
    if not isinstance(queries, list):
        raise ValueError(f"StableToolBench query file must be a list: {input_query_file}")

    selected = []
    for idx, row in enumerate(queries):
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("query_id", idx))
        index_id = str(idx)
        if task_id not in keep and index_id not in keep:
            continue
        copied = dict(row)
        copied.setdefault("query_id", idx)
        selected.append(copied)

    if not selected:
        raise ValueError(f"No task_ids matched {input_query_file}: {sorted(keep)[:10]}")

    subset_path = os.path.join(output_dir, "inputs", os.path.basename(input_query_file))
    _write_json(subset_path, selected)
    return subset_path


@dataclass
class StableToolBenchRunConfig:
    """Official StableToolBench inference settings.

    Defaults mirror ``StepTool/scripts_eval/qwen2/inference_qwen2_vllm.sh`` as
    closely as possible while keeping paths experiment-local.
    """

    stable_root: str = DEFAULT_STABLE_TOOLBENCH_ROOT
    group: str = "G1_instruction"
    input_query_file: str = ""
    config_file: str = ""
    method: str = "DFS_woFilter_w2"
    backbone_model: str = "qwen2"
    chatgpt_model: str = "gpt-4-turbo-2024-04-09"
    model_path: str = "qwen2"
    vllm_api_base: str = "http://127.0.0.1:8084/v1/"
    service_url: str = "http://localhost:8081/virtual"
    max_observation_length: int = 1024
    max_source_sequence_length: int = 4096
    max_sequence_length: int = 8192
    single_chain_max_step: int = 12
    max_query_count: int = 30
    observ_compress_method: str = "truncate"
    num_thread: int = 4
    disable_tqdm: bool = False
    overwrite: bool = False
    use_rapidapi_key: bool = False
    api_customization: bool = False
    rapidapi_key: str = ""
    lora: bool = False
    lora_path: str = ""
    extra_args: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)

    def resolved_config_file(self) -> str:
        return self.config_file or os.path.join(self.stable_root, "config.yml")

    def resolved_input_query_file(self) -> str:
        return self.input_query_file or stable_group_input(self.stable_root, self.group)

    def cli_args(self, prompt_file: str, input_query_file: str, output_answer_file: str) -> list[str]:
        args = [
            "--stable-root",
            self.stable_root,
            "--config-file",
            self.resolved_config_file(),
            "--prompt-file",
            prompt_file,
            "--backbone-model",
            self.backbone_model,
            "--chatgpt-model",
            self.chatgpt_model,
            "--model-path",
            self.model_path,
            "--max-observation-length",
            str(self.max_observation_length),
            "--max-source-sequence-length",
            str(self.max_source_sequence_length),
            "--max-sequence-length",
            str(self.max_sequence_length),
            "--single-chain-max-step",
            str(self.single_chain_max_step),
            "--max-query-count",
            str(self.max_query_count),
            "--observ-compress-method",
            self.observ_compress_method,
            "--method",
            self.method,
            "--input-query-file",
            input_query_file,
            "--output-answer-file",
            output_answer_file,
            "--num-thread",
            str(self.num_thread),
            "--vllm-api-base",
            self.vllm_api_base,
            "--service-url",
            self.service_url,
        ]
        if self.disable_tqdm:
            args.append("--disable-tqdm")
        if self.overwrite:
            args.append("--overwrite")
        if self.use_rapidapi_key:
            args.append("--use-rapidapi-key")
        if self.api_customization:
            args.append("--api-customization")
        if self.rapidapi_key:
            args.extend(["--rapidapi-key", self.rapidapi_key])
        if self.lora:
            args.append("--lora")
        if self.lora_path:
            args.extend(["--lora-path", self.lora_path])
        args.extend(self.extra_args)
        return args


class StableToolBenchRolloutRunner:
    """Run StableToolBench with an experiment-local static prompt override."""

    def __init__(
        self,
        prompt_store: ToolBenchPromptStore | None = None,
        output_dir: str = DEFAULT_TOOLBENCH_EXPERIMENTS_DIR,
        python_executable: str = sys.executable,
        run_config: StableToolBenchRunConfig | None = None,
    ):
        self.prompt_store = prompt_store or ToolBenchPromptStore()
        self.output_dir = output_dir
        self.python_executable = python_executable
        self.run_config = run_config or StableToolBenchRunConfig()

    def command(
        self,
        prompt_file: str,
        input_query_file: str,
        output_answer_file: str,
        run_config: StableToolBenchRunConfig | None = None,
    ) -> list[str]:
        cfg = run_config or self.run_config
        return [
            self.python_executable,
            "-m",
            "terrabox.evolution.promptevo.adapters.toolbench.runner",
            "_stable_official",
            *cfg.cli_args(prompt_file, input_query_file, output_answer_file),
        ]

    def run_version(
        self,
        prompt_version: str,
        experiment: str,
        *,
        run_config: StableToolBenchRunConfig | None = None,
        task_ids: Optional[list[str]] = None,
        prompt: str | None = None,
        dry_run: bool = False,
        check: bool = True,
    ) -> str:
        cfg = run_config or self.run_config
        exp_dir = stable_experiment_dir(experiment, self.output_dir)
        os.makedirs(exp_dir, exist_ok=True)

        prompt_text = prompt if prompt is not None else self.prompt_store.load(prompt_version)
        prompt_file = os.path.join(exp_dir, "static_prompt.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt_text.strip() + "\n")

        input_query_file = _prepare_query_file(
            cfg.resolved_input_query_file(),
            exp_dir,
            task_ids=task_ids,
        )
        output_answer_file = stable_group_results(exp_dir, cfg.group)
        if cfg.overwrite and os.path.exists(output_answer_file):
            shutil.rmtree(output_answer_file)
        os.makedirs(output_answer_file, exist_ok=True)

        cmd = self.command(prompt_file, input_query_file, output_answer_file, cfg)
        meta = {
            "experiment": experiment,
            "prompt_version": prompt_version,
            "group": cfg.group,
            "stable_root": cfg.stable_root,
            "input_query_file": input_query_file,
            "output_answer_file": output_answer_file,
            "method": cfg.method,
            "backbone_model": cfg.backbone_model,
            "chatgpt_model": cfg.chatgpt_model,
            "model_path": cfg.model_path,
            "vllm_api_base": cfg.vllm_api_base,
            "service_url": cfg.service_url,
            "command": cmd,
        }
        _write_json(os.path.join(exp_dir, f"run_meta_{cfg.group}.json"), meta)
        # Keep the historical single-run filenames for callers that only run
        # one StableToolBench group per experiment, but avoid losing per-group
        # provenance when the pipeline runs all six groups into one directory.
        _write_json(os.path.join(exp_dir, "run_meta.json"), meta)
        with open(os.path.join(exp_dir, f"command_{cfg.group}.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n")
        with open(os.path.join(exp_dir, "command.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n")

        if dry_run:
            return exp_dir

        ok, reason = stable_toolbench_has_core_dependencies(
            stable_root=cfg.stable_root,
            python_executable=self.python_executable,
        )
        if not ok:
            _write_json(
                os.path.join(exp_dir, "run_status.json"),
                {"status": "blocked", "reason": reason},
            )
            raise RuntimeError(f"StableToolBench dependencies are not ready: {reason}")

        env = os.environ.copy()
        env["PYTHONPATH"] = _toolbench_pythonpath(cfg.stable_root, env.get("PYTHONPATH", ""))
        env.update(cfg.env_overrides)
        env["VLLM_API_BASE"] = cfg.vllm_api_base
        env["SERVICE_URL"] = cfg.service_url
        env.setdefault("no_proxy", "localhost,127.0.0.1")
        log_path = os.path.join(exp_dir, f"{cfg.group}.log")
        with open(log_path, "a", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                cwd=cfg.stable_root,
                env=env,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return exp_dir

    def run(self, prompt: str, task_ids: list[str], experiment: str) -> str:
        return self.run_version(
            prompt_version=f"{experiment}_prompt",
            experiment=experiment,
            task_ids=task_ids,
            prompt=prompt,
        )

    def components_for_experiment(self, experiment: str, group: str | None = None):
        exp_dir = stable_experiment_dir(experiment, self.output_dir)
        results_dir = stable_group_results(exp_dir, group) if group else os.path.join(exp_dir, "answers")
        traces = ToolBenchTrajectorySource(results_dir_fn=lambda _exp: results_dir)
        metrics = ToolBenchMetricProvider(results_dir_fn=lambda _exp: results_dir)
        return self.prompt_store, traces, metrics


def make_stable_toolbench_components(
    results_dir: str,
    *,
    base_prompt_path: str = "",
    versions_dir: str = "evolution_store/promptevo/toolbench/versions",
):
    prompts = ToolBenchPromptStore(versions_dir=versions_dir, base_prompt_path=base_prompt_path)
    traces = ToolBenchTrajectorySource(results_dir_fn=lambda _exp: results_dir)
    metrics = ToolBenchMetricProvider(results_dir_fn=lambda _exp: results_dir)
    runner = StableToolBenchRolloutRunner(prompt_store=prompts)
    return prompts, traces, metrics, runner


def _namespace_from_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        backbone_model=args.backbone_model,
        chatgpt_model=args.chatgpt_model,
        config_file=args.config_file,
        model_path=args.model_path,
        lora=args.lora,
        lora_path=args.lora_path,
        max_observation_length=args.max_observation_length,
        max_source_sequence_length=args.max_source_sequence_length,
        max_sequence_length=args.max_sequence_length,
        single_chain_max_step=args.single_chain_max_step,
        max_query_count=args.max_query_count,
        observ_compress_method=args.observ_compress_method,
        method=args.method,
        input_query_file=args.input_query_file,
        output_answer_file=args.output_answer_file,
        rapidapi_key=args.rapidapi_key,
        use_rapidapi_key=args.use_rapidapi_key,
        api_customization=args.api_customization,
        num_thread=args.num_thread,
        disable_tqdm=args.disable_tqdm,
        overwrite=args.overwrite,
        easy_tool=False,
    )


def _run_stable_official(args: argparse.Namespace) -> int:
    stable_root = os.path.abspath(args.stable_root)
    inference_root = os.path.join(stable_root, "toolbench", "inference")
    sys.path.insert(0, inference_root)
    sys.path.insert(0, stable_root)

    prompt_text = open(args.prompt_file, encoding="utf-8").read().strip()
    import Prompts.ReAct_prompts as react_prompts

    react_prompts.FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION = prompt_text
    _install_stable_import_compat()

    config = _config_with_absolute_paths(args.config_file, stable_root)
    os.environ["OPENAI_API_BASE"] = str(os.getenv("OPENAI_API_BASE") or config.get("api_base") or "")
    os.environ["OPENAI_KEY"] = str(os.getenv("OPENAI_KEY") or config.get("api_key") or "")
    os.environ["TOOLBENCH_KEY"] = str(config.get("toolbench_key") or "")
    os.environ["TOOL_ROOT_DIR"] = str(config.get("tool_root_dir") or os.path.join(stable_root, "server", "tools"))
    os.environ["VLLM_API_BASE"] = args.vllm_api_base
    os.environ["SERVICE_URL"] = args.service_url

    from toolbench.inference.Downstream_tasks.rapidapi_multithread import pipeline_runner

    runner = pipeline_runner(_namespace_from_args(args))
    runner.run()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StableToolBench promptevo runner")
    sub = parser.add_subparsers(dest="cmd", required=True)
    official = sub.add_parser("_stable_official", help=argparse.SUPPRESS)
    official.add_argument("--stable-root", default=DEFAULT_STABLE_TOOLBENCH_ROOT)
    official.add_argument("--config-file", required=True)
    official.add_argument("--prompt-file", required=True)
    official.add_argument("--backbone-model", default="qwen2")
    official.add_argument("--chatgpt-model", default="gpt-4-turbo-2024-04-09")
    official.add_argument("--model-path", default="qwen2")
    official.add_argument("--lora", action="store_true")
    official.add_argument("--lora-path", default="")
    official.add_argument("--max-observation-length", type=int, default=1024)
    official.add_argument("--max-source-sequence-length", type=int, default=4096)
    official.add_argument("--max-sequence-length", type=int, default=8192)
    official.add_argument("--single-chain-max-step", type=int, default=12)
    official.add_argument("--max-query-count", type=int, default=30)
    official.add_argument("--observ-compress-method", default="truncate", choices=["truncate", "filter", "random"])
    official.add_argument("--method", default="DFS_woFilter_w2")
    official.add_argument("--input-query-file", required=True)
    official.add_argument("--output-answer-file", required=True)
    official.add_argument("--rapidapi-key", default="")
    official.add_argument("--use-rapidapi-key", action="store_true")
    official.add_argument("--api-customization", action="store_true")
    official.add_argument("--num-thread", type=int, default=4)
    official.add_argument("--disable-tqdm", action="store_true")
    official.add_argument("--overwrite", action="store_true")
    official.add_argument("--vllm-api-base", default="http://127.0.0.1:8084/v1/")
    official.add_argument("--service-url", default="http://localhost:8081/virtual")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "_stable_official":
        return _run_stable_official(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
