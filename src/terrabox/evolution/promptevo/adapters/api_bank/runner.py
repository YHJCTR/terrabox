"""Optional rollout runner for API-Bank prompt-version experiments."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable, Optional

from .api_utils import (
    _analyze_api_prediction,
    _format_api_call,
    _hash_prompt,
    _rouge_l_f1,
    _tool_search_enabled,
    build_api_call_messages,
    load_api_descriptions,
    samples_from_history,
)
from .files import (
    DEFAULT_API_BANK_EXPERIMENTS_DIR,
    _iter_jsonl_files,
    _prediction_key,
    _read_jsonl,
    _write_jsonl,
    api_bank_prediction_path,
    api_bank_rollout_path,
    load_predictions,
)
from .metrics import APIBankMetricProvider
from .prompts import APIBankPromptStore
from .traces import APIBankTrajectorySource

class APIBankRolloutRunner:
    """Run API-Bank API-call prediction for one static instruction version.

    This is intentionally prompt-slot based: `PromptStore` supplies only the
    static task instruction. API descriptions and chat history remain dynamic.
    """

    def __init__(
        self,
        prompts: Optional[APIBankPromptStore] = None,
        data_dir: str = "",
        api_bank_root: str = "",
        output_dir: str = DEFAULT_API_BANK_EXPERIMENTS_DIR,
        llm: Any = None,
        max_tokens: int = 256,
        enable_thinking: bool = False,
        sleep_s: float = 0.0,
    ):
        self.prompts = prompts or APIBankPromptStore()
        self.data_dir = data_dir
        self.api_bank_root = api_bank_root or self._infer_api_bank_root(data_dir)
        self.output_dir = output_dir
        self.llm = llm
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.sleep_s = sleep_s

    @staticmethod
    def _infer_api_bank_root(data_dir: str) -> str:
        cur = os.path.abspath(data_dir or ".")
        while cur and cur != os.path.dirname(cur):
            if os.path.isdir(os.path.join(cur, "apis")):
                return cur
            cur = os.path.dirname(cur)
        return os.path.abspath(data_dir or ".")

    def _get_llm(self):
        if self.llm is None:
            from terrabox.evolution.shared.llm_client import EvolutionLLMClient
            self.llm = EvolutionLLMClient()
        return self.llm

    def iter_samples(
        self,
        task_ids: Optional[list[str]] = None,
        limit: Optional[int] = None,
        task_kind: str = "api_call",
    ) -> Iterable[dict]:
        keep = set(task_ids or [])
        count = 0
        for path in _iter_jsonl_files(self.data_dir):
            history = _read_jsonl(path)
            for sample in samples_from_history(path, history):
                if task_kind != "both" and sample["kind"] != task_kind:
                    continue
                if keep and sample["task_id"] not in keep:
                    continue
                yield sample
                count += 1
                if limit is not None and count >= limit:
                    return

    def iter_shard_samples(
        self,
        shard_index: int,
        num_shards: int,
        task_ids: Optional[list[str]] = None,
        limit: Optional[int] = None,
        skip_completed: Optional[set[tuple[str, int]]] = None,
        task_kind: str = "api_call",
    ) -> Iterable[dict]:
        keep = set(task_ids or [])
        skip_completed = skip_completed or set()
        yielded = 0
        global_index = 0
        for path in _iter_jsonl_files(self.data_dir):
            history = _read_jsonl(path)
            for sample in samples_from_history(path, history):
                if task_kind != "both" and sample["kind"] != task_kind:
                    continue
                if keep and sample["task_id"] not in keep:
                    continue
                if num_shards > 1 and global_index % num_shards != shard_index:
                    global_index += 1
                    continue
                global_index += 1
                if _prediction_key(sample["file"], sample["id"]) in skip_completed:
                    continue
                yield sample
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    def build_messages(
        self,
        sample: dict,
        static_instruction: str,
        task_kind: str = "api_call",
    ) -> tuple[list[dict[str, str]], str]:
        if task_kind == "api_call" and _tool_search_enabled(self.data_dir):
            api_names = ["ToolSearcher"]
        else:
            api_names = sample.get("apis") or []
        desc_map = load_api_descriptions(self.api_bank_root, api_names)
        descriptions = "\n".join(desc_map[name] for name in sorted(desc_map))
        messages = build_api_call_messages(static_instruction, descriptions, sample["chat_history"])
        return messages, descriptions

    def run_version(
        self,
        prompt_version: str,
        experiment: str,
        task_ids: Optional[list[str]] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
        shard_index: int = 0,
        num_shards: int = 1,
        resume: bool = False,
        task_kind: str = "api_call",
    ) -> str:
        static_instruction = self.prompts.load(prompt_version)
        exp_dir = os.path.join(self.output_dir, experiment)
        prediction_path = api_bank_prediction_path(self.output_dir, experiment)
        rollout_path = api_bank_rollout_path(self.output_dir, experiment)
        os.makedirs(exp_dir, exist_ok=True)

        meta = {
            "experiment": experiment,
            "prompt_version": prompt_version,
            "prompt_sha256_12": _hash_prompt(static_instruction),
            "data_dir": os.path.abspath(self.data_dir),
            "api_bank_root": os.path.abspath(self.api_bank_root),
            "prediction_path": os.path.abspath(prediction_path),
            "rollout_path": os.path.abspath(rollout_path),
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
            "dry_run": dry_run,
            "tool_search_enabled": _tool_search_enabled(self.data_dir),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "resume": resume,
            "task_kind": task_kind,
        }
        with open(os.path.join(exp_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        pred_rows = []
        rollout_rows = []
        completed: set[tuple[str, int]] = set()
        if resume:
            existing_predictions = load_predictions(prediction_path) if os.path.exists(prediction_path) else {}
            existing_rollouts = _read_jsonl(rollout_path) if os.path.exists(rollout_path) else []
            existing_rollouts_by_key = {
                _prediction_key(str(row["file"]), int(row["id"])): row
                for row in existing_rollouts
                if "file" in row and "id" in row
            }
            existing_rollout_keys = set(existing_rollouts_by_key)
            # A prediction without its trace is not a completed task. Keeping
            # the pair aligned makes a resumed run safe after interruption. A
            # transient provider failure is deliberately not terminal: retain
            # its old rows for audit, but let the resumed rollout append a new
            # attempt for that task.
            completed = set(existing_predictions) & existing_rollout_keys
            try:
                from terrabox.agent.llm_provider import is_retryable_remote_error_text
                completed = {
                    key for key in completed
                    if not is_retryable_remote_error_text(
                        str(existing_rollouts_by_key.get(key, {}).get("error") or "")
                    )
                }
            except Exception:
                # Resume must remain usable even when this adapter is imported
                # outside the full Terrabox package.
                pass
            pred_rows = [row for key, row in existing_predictions.items() if key in completed]
            rollout_rows = [
                row for row in existing_rollouts
                if "file" in row and "id" in row
                and _prediction_key(str(row["file"]), int(row["id"])) in completed
            ]
            _write_jsonl(prediction_path, pred_rows)
            _write_jsonl(rollout_path, rollout_rows)
        llm = None if dry_run else self._get_llm()
        for sample in self.iter_shard_samples(
            shard_index=shard_index,
            num_shards=num_shards,
            task_ids=task_ids,
            limit=limit,
            skip_completed=completed,
            task_kind=task_kind,
        ):
            messages, descriptions = self.build_messages(sample, static_instruction, task_kind=sample["kind"])
            pred_text = ""
            error = ""
            started = time.time()
            if dry_run:
                pred_text = ""
            else:
                try:
                    pred_text = str(llm.call(
                        messages[-1]["content"] if len(messages) == 1 else _messages_to_user_prompt(messages[1:]),
                        system=messages[0]["content"],
                        max_tokens=self.max_tokens,
                        enable_thinking=self.enable_thinking,
                    ) or "").strip()
                except Exception as exc:
                    error = repr(exc)
            if sample["kind"] == "api_call":
                analysis = _analyze_api_prediction(pred_text, sample["ground_truth"])
                ok = bool(analysis["ok"])
                reason = str(analysis["reason"])
                flags = dict(analysis["flags"])
            else:
                response_score = round(_rouge_l_f1(str(sample["ground_truth"].get("text") or ""), pred_text), 4)
                analysis = {
                    "ok": response_score >= 0.2,
                    "reason": f"ROUGE_L:{response_score}",
                    "response_rouge_l": response_score,
                }
                ok = bool(analysis["ok"])
                reason = str(analysis["reason"])
                flags = {
                    "empty_prediction": not bool(pred_text.strip()),
                    "low_response_rouge": response_score < 0.2,
                }
            prediction_row = {"file": sample["file"], "id": sample["id"], "pred": pred_text}
            rollout_row = {
                "task_id": sample["task_id"],
                "file": sample["file"],
                "id": sample["id"],
                "kind": sample["kind"],
                "success": ok,
                "pred": pred_text,
                "reason": reason,
                "failure_flags": flags,
                "gold_api": sample["ground_truth"].get("api_name"),
                "gold_args": sample["ground_truth"].get("param_dict") or {},
                "gold_response": sample["ground_truth"].get("text") or "",
                "apis": sample.get("apis") or [],
                "api_descriptions": descriptions,
                "chat_history": sample["chat_history"],
                "ground_truth": sample["ground_truth"],
                "messages": messages,
                "latency_s": round(time.time() - started, 3),
                "error": error,
                "analysis": analysis,
            }
            # Persist one task at a time so --resume never loses a completed
            # provider call if the parent/watchdog is interrupted.
            _append_jsonl(prediction_path, prediction_row)
            _append_jsonl(rollout_path, rollout_row)
            pred_rows.append(prediction_row)
            rollout_rows.append(rollout_row)
            if self.sleep_s:
                time.sleep(self.sleep_s)

        return experiment

    def run(self, prompt: str, task_ids: list[str], experiment: str) -> str:
        version = f"{experiment}_prompt"
        self.prompts.save(version, prompt, {"source": "APIBankRolloutRunner.run"})
        return self.run_version(version, experiment, task_ids=task_ids)


def _messages_to_user_prompt(messages: list[dict[str, str]]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"AI: {content}")
        else:
            lines.append(str(content))
    return "\n".join(lines)


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def make_api_bank_components(
    data_dir: str,
    api_bank_root: str = "",
    output_dir: str = DEFAULT_API_BANK_EXPERIMENTS_DIR,
    versions_dir: str = "evolution_store/promptevo/api_bank/versions",
    llm: Any = None,
    max_tokens: int = 256,
):
    """Build the promptevo components for an API-Bank experiment loop."""
    prompts = APIBankPromptStore(versions_dir=versions_dir)
    pred_fn = lambda exp: api_bank_prediction_path(output_dir, exp)
    rollout_fn = lambda exp: api_bank_rollout_path(output_dir, exp)
    traces = APIBankTrajectorySource(
        data_dir_fn=lambda exp: data_dir,
        prediction_path_fn=pred_fn,
        rollout_path_fn=rollout_fn,
    )
    metrics = APIBankMetricProvider(
        data_dir_fn=lambda exp: data_dir,
        prediction_path_fn=pred_fn,
        rollout_path_fn=rollout_fn,
    )
    runner = APIBankRolloutRunner(
        prompts=prompts,
        data_dir=data_dir,
        api_bank_root=api_bank_root,
        output_dir=output_dir,
        llm=llm,
        max_tokens=max_tokens,
    )
    return prompts, traces, metrics, runner


def write_oracle_predictions(data_dir: str, output_path: str) -> str:
    """Write gold API calls in API-Bank evaluator prediction format.

    Useful for smoke tests and adapter validation; this is not a model runner.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for path in _iter_jsonl_files(data_dir):
            history = _read_jsonl(path)
            for sample in samples_from_history(path, history):
                if sample["kind"] != "api_call":
                    continue
                gt = sample["ground_truth"]
                pred = _format_api_call(str(gt.get("api_name")), gt.get("param_dict") or {})
                out.write(json.dumps({"file": sample["file"], "id": sample["id"], "pred": pred}, ensure_ascii=False) + "\n")
    return os.path.abspath(output_path)
