"""Frozen, conversation-grouped API-Bank comparisons using upstream optimizers."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .api_utils import _analyze_api_prediction, _parse_api_call, load_api_descriptions
from .constants import API_BANK_SYSTEM_PROMPT
from .runner import APIBankRolloutRunner, _messages_to_user_prompt
from terrabox.evolution.promptevo.protocol_patch import protocol_renderer_version


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + "." + uuid.uuid4().hex + ".new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def grouped_split(samples, seed):
    files = sorted({sample["file"] for sample in samples})
    random.Random(seed).shuffle(files)
    a, b = int(len(files) * .6), int(len(files) * .8)
    groups = dict(train=set(files[:a]), dev=set(files[a:b]), test=set(files[b:]))
    result = {name: [s["task_id"] for s in samples if s["file"] in group]
              for name, group in groups.items()}
    if not all(result.values()):
        raise ValueError("All three grouped splits must be nonempty")
    return result


class Comparison:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output_dir) / args.group
        self.root.mkdir(parents=True, exist_ok=True)
        self.runner = APIBankRolloutRunner(data_dir=args.data_dir, api_bank_root=args.api_bank_root)
        self.samples = {s["task_id"]: s for s in self.runner.iter_samples()}
        self.splits = grouped_split(list(self.samples.values()), args.seed)
        self.base = API_BANK_SYSTEM_PROMPT.strip()
        self.manifest = {
            "schema": 1, "seed": args.seed, "provider": "longcat",
            "data_sha256": digest(self.samples), "base_prompt": self.base,
            "splits": self.splits, "actor_max_tokens": 256,
            "optimizer_max_tokens": 4096, "gepa_metric_budget": args.metric_budget,
            "aho_iterations": args.iterations,
            "metric": "API-Bank adapter exact API-name/argument matching; not execution success",
            "scope_protocol": "train-only online strategic learning, frozen dev/test",
            "promptevo_task_profile": getattr(args, "promptevo_task_profile", "generic"),
            "promptevo_patch_composition": getattr(args, "promptevo_patch_composition", "whole"),
            "promptevo_protocol_mode": getattr(args, "promptevo_protocol_mode", "legacy"),
            "promptevo_protocol_renderer": protocol_renderer_version(
                getattr(args, "promptevo_protocol_mode", "legacy")
            ),
            "promptevo_gate_variant": (
                "paired_conditional_v1"
                if getattr(args, "promptevo_protocol_mode", "legacy") == "conditional"
                else "aggregate_legacy_v1"
            ),
        }
        path = self.root / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != self.manifest:
            raise ValueError("Manifest changed; use a new comparison group")
        save(path, self.manifest)
        from terrabox.agent.llm_provider import make_llm_client
        self.client = make_llm_client("longcat")
        identity = dict(provider="longcat", model=self.client.spec.model, api_base=self.client.spec.base_url)
        identity_path = self.root / "provider.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Provider identity changed; use a new group")
        save(identity_path, identity)
        self.method = "preflight"
        self.failure = None

    def state(self, status, **details):
        save(self.root / "status.json", dict(status=status, method=self.method,
             updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **details))
        print(json.dumps(dict(status=status, method=self.method, **details), ensure_ascii=False), flush=True)

    def llm(self, prompt, system=None, max_tokens=4096):
        key = digest([system, prompt, max_tokens])
        path = self.root / "llm_cache" / (key + ".json")
        if path.exists():
            return json.loads(path.read_text())["response"]
        try:
            from terrabox.agent.llm_provider import RemoteChatClient
            client = RemoteChatClient(self.client.spec)
            response = client.call(prompt, system=system, max_tokens=max_tokens, enable_thinking=False)
        except Exception as exc:
            self.failure = exc
            message = str(exc).lower()
            account_error = getattr(exc, "code", None) in (401, 402, 403) or any(
                word in message for word in ("payment", "billing", "insufficient_quota", "quota exceeded", "额度", "余额"))
            save(self.root / self.method / "provider_error.json", dict(
                error_type=type(exc).__name__, account_error=account_error))
            raise
        save(path, dict(system=system, prompt=prompt, max_tokens=max_tokens, response=response,
                       usage=dict(input_tokens=client.cost.in_hit + client.cost.in_miss,
                                  output_tokens=client.cost.out, calls=client.cost.calls)))
        return response

    def messages(self, messages, max_tokens=4096):
        if isinstance(messages, str):
            return self.llm(messages, max_tokens=max_tokens)
        items = [dict(role=m.role, content=m.content) if hasattr(m, "role") else m for m in messages]
        systems = "\n".join(m["content"] for m in items if m["role"] == "system")
        rest = [m for m in items if m["role"] != "system"]
        return self.llm(_messages_to_user_prompt(rest), system=systems or None, max_tokens=max_tokens)

    def evaluate_one(self, prompt, task_id):
        if prompt.startswith('{"evotool_policy":'):
            return self.evotool_evaluate(prompt, task_id)
        sample = self.samples[task_id]
        key = digest([prompt, task_id])
        path = self.root / "evaluations" / (key + ".json")
        if path.exists():
            return json.loads(path.read_text())
        messages, _ = self.runner.build_messages(sample, prompt)
        prediction = self.llm(_messages_to_user_prompt(messages[1:]),
                              system=messages[0]["content"], max_tokens=256).strip()
        analysis = _analyze_api_prediction(prediction, sample["ground_truth"])
        row = dict(task_id=task_id, prompt_sha256=digest(prompt), messages=messages,
                   prediction=prediction, success=bool(analysis["ok"]), analysis=analysis)
        save(path, row)
        return row

    def evaluate(self, prompt, ids):
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            return list(pool.map(lambda task: self.evaluate_one(prompt, task), ids))

    def finish(self, prompt):
        # Freeze before evaluating test. Never expose test feedback to optimizers.
        save(self.root / self.method / "selected_prompt.json", dict(prompt=prompt))
        self.state("running", phase="test", total=len(self.splits["test"]))
        rows = self.evaluate(prompt, self.splits["test"])
        result = dict(status="complete", n=len(rows), correct=sum(r["success"] for r in rows),
                      task_ids=self.splits["test"], prompt_sha256=digest(prompt))
        result["exact_call_accuracy"] = result["correct"] / result["n"]
        save(self.root / self.method / "test_results.json", rows)
        save(self.root / self.method / "summary.json", result)
        self.full_evaluation(prompt)
        self.state("complete", **{k: v for k, v in result.items() if k != "status"})

    def full_evaluation(self, prompt):
        import fcntl
        directory = self.root / self.method
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "full_eval.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (directory / "full_set_summary.json").exists():
                return
            save(directory / "full_eval_status.json", dict(status="running", n=len(self.samples)))
            print(f"Full evaluation: {self.method}, {len(self.samples)} examples", flush=True)
            prompts = {"final": prompt}
            if self.method == "promptevo":
                stages = json.loads((directory / "stage_prompts.json").read_text())
                prompts["stage1"] = stages["stage1"]
            elif self.method == "evotool":
                modules, _, _ = self.evotool_imports()
                policy = modules.initial_policy()
                prompts["modular_base"] = json.dumps({"evotool_policy": {
                    key: getattr(policy, key) for key in modules.MODULES}}, sort_keys=True)
            results = {}
            for version, instruction in prompts.items():
                rows = self.evaluate(instruction, list(self.samples))
                if len(rows) != len(self.samples) or {r["task_id"] for r in rows} != set(self.samples):
                    raise RuntimeError("Full evaluation coverage mismatch")
                save(directory / f"full_set_{version}_results.json", rows)
                correct = sum(r["success"] for r in rows)
                results[version] = dict(n=len(rows), correct=correct,
                    exact_call_accuracy=correct / len(rows), prompt_sha256=digest(instruction))
            save(directory / "full_set_summary.json", dict(status="complete",
                evaluation_scope="All dataset examples, including optimization train/dev; not held-out",
                split_sizes={k: len(v) for k, v in self.splits.items()}, versions=results))
            save(directory / "full_eval_status.json", dict(status="complete", n=len(self.samples)))

    def provenance(self, name, folder, boundary):
        repo = Path(self.args.methods_root) / folder
        sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        path = self.root / name / "provenance.json"
        value = dict(source=str(repo), commit=sha, boundary=boundary)
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("Upstream provenance changed; use a new group")
        save(path, value)
        return repo

    def gepa(self):
        repo = self.provenance("gepa", "gepa", "Official GEPA search; custom API-Bank adapter and LongCat transport")
        sys.path.insert(0, str(repo / "src"))
        import gepa
        from gepa.core.adapter import EvaluationBatch
        outer = self

        class Adapter:
            propose_new_texts = None

            def evaluate(self, batch, candidate, capture_traces=False):
                outer.state("running", phase="gepa_search", batch_size=len(batch))
                rows = outer.evaluate(candidate["instruction"], batch)
                return EvaluationBatch(outputs=[r["prediction"] for r in rows],
                    scores=[float(r["success"]) for r in rows], trajectories=rows if capture_traces else None)

            def make_reflective_dataset(self, candidate, evaluation_batch, components_to_update):
                # Optimizer sees observable inputs/output, not gold answers or gold-derived diagnostics.
                records = [{"Inputs": r["messages"], "Output": r["prediction"],
                            "Feedback": "Check the API format, visible schema and conversation for mistakes."}
                           for r in evaluation_batch.trajectories]
                return {key: records for key in components_to_update}

        result = gepa.optimize(seed_candidate={"instruction": self.base},
            trainset=self.splits["train"], valset=self.splits["dev"], adapter=Adapter(),
            reflection_lm=self.messages, max_metric_calls=self.args.metric_budget,
            reflection_minibatch_size=3, run_dir=str(self.root / "gepa" / "search"),
            seed=self.args.seed, raise_on_exception=True, display_progress_bar=False)
        save(self.root / "gepa" / "selection.json", dict(best_idx=result.best_idx,
             scores=result.val_aggregate_scores, candidates=result.candidates,
             feedback_boundary="Metric selection uses train/dev labels; reflection gets observable context only"))
        return result.best_candidate["instruction"]

    def scope(self):
        repo = self.provenance("scope", "SCOPE", "Official SCOPE strategic rules; train-only single-step adaptation, frozen test; no tactical retry")
        sys.path.insert(0, str(repo))
        from scope.optimizer import SCOPEOptimizer
        from scope.models.base import ModelResponse
        outer = self

        class Model:
            async def generate(self, messages):
                return ModelResponse(content=outer.messages(messages))

        checkpoint = self.root / "scope" / "checkpoint.json"
        state = json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(completed=0, rules={})
        work = self.root / "scope" / "memory"
        # Restore the last committed task, undoing an interrupted partial rule update.
        save(work / "strategic_memory" / "global_rules.json", state["rules"])
        optimizer = SCOPEOptimizer(Model(), str(work), store_history=True)

        async def train():
            nonlocal optimizer
            for i, task in enumerate(self.splits["train"]):
                if i < state["completed"]:
                    continue
                # SCOPE counters/tactical memory are task-local; strategic memory persists on disk.
                optimizer = SCOPEOptimizer(Model(), str(work), store_history=True)
                self.state("running", phase="scope_train", completed=i, total=len(self.splits["train"]))
                prompt = self.base + "\n" + optimizer.get_strategic_rules_for_agent("api_agent")
                row = self.evaluate_one(prompt, task)
                _, _, parse_error = _parse_api_call(row["prediction"])
                await optimizer.on_step_complete(agent_name="api_agent", agent_role="API-call prediction",
                    task=_messages_to_user_prompt(row["messages"]), model_output=row["prediction"],
                    observations="Single next-call prediction; execution output is unavailable.",
                    error=ValueError(parse_error) if parse_error else None,
                    current_system_prompt=prompt, task_id=task, truncate_context=False)
                if self.failure:
                    raise self.failure
                save(checkpoint, dict(completed=i + 1, rules=optimizer.strategic_store.rules))
            return self.base + "\n" + optimizer.get_strategic_rules_for_agent("api_agent")

        return asyncio.run(train())

    def aho(self):
        repo = self.provenance("aho", "agent-harness-optimizer",
            "Official BetterHarness acceptance/history loop; prompt-only API-Bank adapter. "
            "DeepAgents filesystem proposer replaced by one JSON proposal over bounded workspace evidence; adapted, not full harness reproduction.")
        sys.path.insert(0, str(repo))
        try:
            from agent_harness_optimizer.framework.benchmark import Benchmark, CaseScore, SplitScore
            from agent_harness_optimizer.framework.optimizer import OptimizeConfig
            from agent_harness_optimizer.optimizers.better_harness.loop import BetterHarnessOptimizer, BHConfig
        except ModuleNotFoundError as exc:
            # The upstream package imports langchain_core even for its prompt-only
            # loop. Keep the experiment runnable without adding a dependency and
            # preserve the documented iterate -> propose -> accept semantics.
            if exc.name != "langchain_core":
                raise
            return self._aho_prompt_only_fallback()
        outer = self

        class Bank(Benchmark):
            name = "api_bank_exact_call"
            default_model = "longcat"
            default_system_prompt = outer.base

            def build_model(self, model_name):
                return outer.client

            async def score_async(self, prompt, split, output_dir, **kwargs):
                if outer.failure:
                    raise outer.failure
                ids = outer.splits["dev" if split == "holdout" else "train"]
                outer.state("running", phase="aho_" + split, total=len(ids))
                rows = outer.evaluate(prompt, ids)
                cases = [CaseScore(case_id=r["task_id"], passed=r["success"],
                         extra=dict(messages=r["messages"], prediction=r["prediction"])) for r in rows]
                return SplitScore(passed=sum(c.passed for c in cases), total=len(cases), reliability=1., cases=cases)

            def build_asi(self, score, failure_matrix_cases=None):
                return json.dumps(dict(passed=score.passed, total=score.total,
                                       failure_history=failure_matrix_cases), ensure_ascii=False)

            def extract_top_patterns(self, score, n=3):
                ids = [c.case_id for c in score.cases if not c.passed]
                return [dict(key="incorrect_call", count=len(ids), case_ids=ids)] if ids else []

            def write_case_files(self, workspace, score, target_case_ids=None):
                for case in score.cases:
                    group = "passing" if case.passed else "failures"
                    save(workspace / "train_cases" / group / (digest(case.case_id) + ".json"), case.extra)

        class Optimizer(BetterHarnessOptimizer):
            def _run_variant(self, ws_variant, variant_name, variant_system, outer_model, max_turns):
                if outer.failure:
                    # Upstream catches Exception and retries proposers; abort systemic API failures.
                    raise SystemExit("LongCat transport failed; resume after fixing provider")
                packet = {}
                for rel in ("task.md", "asi.md", "history/history.md", "history/failure_matrix.md", "current/system_prompt.txt"):
                    path = ws_variant / rel
                    if path.exists():
                        packet[rel] = path.read_text()
                for group, limit in (("failures", 12), ("passing", 4)):
                    packet[group] = [json.loads(p.read_text()) for p in sorted(
                        (ws_variant / "train_cases" / group).glob("*.json"))[:limit]]
                system = variant_system + '\nThe workspace is supplied as JSON. Return JSON only: {"prompt":"complete replacement static instruction", "proposal":"reason and verdict"}. Do not edit tool schemas or conversation history.'
                try:
                    text = outer.llm(json.dumps(packet, ensure_ascii=False), system=system)
                except Exception:
                    raise SystemExit("LongCat transport failed; comparison stopped")
                proposal = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
                prompt = proposal.get("prompt", "").strip()
                if not prompt:
                    raise ValueError("Empty AHO proposal")
                return prompt, None, str(proposal.get("proposal", "")), 1, 0, 0

        output = self.root / "aho" / "search"
        config = OptimizeConfig(output_dir=output, inner_model="longcat", outer_model="longcat")
        optimizer = Optimizer(Bank(), config, bh_config=BHConfig(max_iterations=self.args.iterations, prompt_only=True))
        optimizer.run()
        if self.failure:
            raise self.failure
        path = output / "current" / "system_prompt.txt"
        return path.read_text().strip() if path.exists() else self.base

    def _aho_prompt_only_fallback(self):
        """Dependency-light BetterHarness-style loop for API-Bank.

        This is intentionally labeled as an adapted prompt-only baseline. It
        keeps bounded workspace evidence, asks for a JSON proposal, and accepts
        only proposals that do not reduce the fixed dev score.
        """
        output = self.root / "aho" / "fallback"
        output.mkdir(parents=True, exist_ok=True)
        prompt_path = output / "current_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()

        train_ids = self.splits["train"]
        dev_ids = self.splits["dev"]
        current = self.base

        def score(rows):
            return sum(bool(row["success"]) for row in rows) / max(1, len(rows))

        history = []
        current_rows = self.evaluate(current, dev_ids)
        current_score = score(current_rows)
        for iteration in range(max(1, int(self.args.iterations))):
            train_rows = self.evaluate(current, train_ids)
            failures = [
                {
                    "task_id": row["task_id"],
                    "prediction": row["prediction"],
                    "reason": row["analysis"].get("reason"),
                    "messages": row["messages"],
                }
                for row in train_rows if not row["success"]
            ][:12]
            passing = [
                {"task_id": row["task_id"], "prediction": row["prediction"]}
                for row in train_rows if row["success"]
            ][:4]
            packet = {
                "iteration": iteration,
                "current_prompt": current,
                "failure_cases": failures,
                "passing_cases": passing,
                "history": history[-3:],
            }
            system = (
                "You are an adapted BetterHarness prompt proposer. Analyze the bounded workspace evidence "
                "and return JSON only: {\"prompt\":\"complete replacement static instruction\", "
                "\"proposal\":\"short rationale\"}. Do not include benchmark gold, API names, "
                "task-specific examples, or user data in the proposed prompt. Preserve the output contract."
            )
            try:
                text = self.llm(json.dumps(packet, ensure_ascii=False), system=system, max_tokens=4096)
                cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                proposal = json.loads(cleaned)
                candidate = str(proposal.get("prompt") or "").strip()
                rationale = str(proposal.get("proposal") or "")
            except Exception as exc:
                if self.failure:
                    raise self.failure
                history.append({"iteration": iteration, "accepted": False, "error": type(exc).__name__})
                continue
            if not candidate:
                history.append({"iteration": iteration, "accepted": False, "reason": "empty_prompt"})
                continue
            candidate_rows = self.evaluate(candidate, dev_ids)
            candidate_score = score(candidate_rows)
            accepted = candidate_score >= current_score
            history.append({
                "iteration": iteration,
                "accepted": accepted,
                "current_score": current_score,
                "candidate_score": candidate_score,
                "proposal": rationale,
            })
            self.state("running", phase="aho_fallback", iteration=iteration,
                       current_score=current_score, candidate_score=candidate_score,
                       accepted=accepted)
            if accepted:
                current = candidate
                current_score = candidate_score
        save(output / "history.json", {"history": history, "final_dev_score": current_score})
        prompt_path.write_text(current + "\n", encoding="utf-8")
        return current

    def evotool_imports(self):
        repo = Path(self.args.methods_root) / "ACL_2026_EvoTool"
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from src.policy import modules
        from src.policy.agent import Episode, Step
        return modules, Episode, Step

    def promptevo(self):
        from types import SimpleNamespace
        from .pipeline import _stage1, _stage2
        from .metrics import APIBankMetricProvider
        from .traces import APIBankTrajectorySource
        from .prompts import APIBankPromptStore
        from .files import _write_jsonl
        outer = self
        root = self.root / "promptevo" / "search"
        save(self.root / "promptevo" / "provenance.json", dict(
            source="Terrabox PromptEvo", optimizer_version="v2", proposal_format="patch",
            stage1_candidates=getattr(self.args, "promptevo_stage1_candidates", 3),
            stage2_candidates=getattr(self.args, "promptevo_stage2_candidates", 3),
            dev_task_ids=self.splits["dev"],
            task_profile=getattr(self.args, "promptevo_task_profile", "generic"),
            patch_composition=getattr(self.args, "promptevo_patch_composition", "whole"),
            protocol_mode=getattr(self.args, "promptevo_protocol_mode", "legacy"),
            protocol_renderer=protocol_renderer_version(
                getattr(self.args, "promptevo_protocol_mode", "legacy")
            ),
            boundary="Train-only traces, separate full dev baseline; original Stage1/Stage2 algorithms; "
                     "tool_search profile is scenario-adapted guidance, not task-specific answers; "
                     "atomic_pairwise composition searches typed patch subsets and records lineage; "
                     "conditional mode preserves trigger/scope and accepts paired non-regressing gains"))
        store = APIBankPromptStore(versions_dir=str(root / "versions"))

        class Runner:
            def run(self, prompt, task_ids, experiment):
                outer.state("running", phase="promptevo_rollout", experiment=experiment, total=len(task_ids))
                rows = outer.evaluate(prompt, task_ids)
                predictions, traces = [], []
                for row in rows:
                    sample = outer.samples[row["task_id"]]
                    _, descriptions = outer.runner.build_messages(sample, prompt)
                    predictions.append(dict(file=sample["file"], id=sample["id"], pred=row["prediction"]))
                    traces.append(dict(task_id=row["task_id"], file=sample["file"], id=sample["id"],
                        kind="api_call", pred=row["prediction"], success=row["success"],
                        analysis=row["analysis"], reason=row["analysis"]["reason"],
                        failure_flags=row["analysis"]["flags"], error="",
                        api_descriptions=descriptions, chat_history=sample["chat_history"]))
                directory = root / experiment
                directory.mkdir(parents=True, exist_ok=True)
                _write_jsonl(str(directory / "predictions.jsonl"), predictions)
                _write_jsonl(str(directory / "rollout.jsonl"), traces)
                return experiment

        paths = dict(data_dir_fn=lambda _: self.args.data_dir,
                     prediction_path_fn=lambda exp: str(root / exp / "predictions.jsonl"),
                     rollout_path_fn=lambda exp: str(root / exp / "rollout.jsonl"))
        source = APIBankTrajectorySource(**paths)
        metrics = APIBankMetricProvider(**paths, api_bank_root=self.args.api_bank_root)
        runner = Runner()
        args = SimpleNamespace(group="stages", output_dir=str(root), provider="longcat",
            optimizer_version="v2", optimizer_max_tokens=4096,
            stage1_candidates=getattr(self.args, "promptevo_stage1_candidates", 3),
            stage2_candidates=getattr(self.args, "promptevo_stage2_candidates", 3),
            validation_tasks=len(self.splits["dev"]),
            task_profile=getattr(self.args, "promptevo_task_profile", "generic"),
            patch_composition=getattr(self.args, "promptevo_patch_composition", "whole"),
            protocol_mode=getattr(self.args, "promptevo_protocol_mode", "legacy"),
            comparison_dev_ids=self.splits["dev"], comparison_dev_base="dev_base",
            comparison_dev_stage1="dev_stage1")
        runner.run(self.base, self.splits["train"], "stages/base")
        runner.run(self.base, self.splits["dev"], "dev_base")
        _stage1(args, store, runner, source, metrics, "stage1")
        first = store.load("stage1")
        runner.run(first, self.splits["train"], "stages/stage1")
        runner.run(first, self.splits["dev"], "dev_stage1")
        _stage2(args, store, runner, source, metrics, "stage1", "stage2")
        # Both prompts are frozen before either stage touches test.
        second = store.load("stage2")
        save(self.root / "promptevo" / "stage_prompts.json", dict(stage1=first, stage2=second))
        save(self.root / "promptevo" / "stage1_test_results.json", self.evaluate(first, self.splits["test"]))
        return second

    def evotool_client(self, max_tokens=4096):
        from src.llm.client import _parse_json
        outer = self

        class Client:
            total_tokens = 0

            def generate(self, messages, **kwargs):
                return outer.messages(messages, max_tokens=kwargs.get("max_tokens") or max_tokens)

            def generate_json(self, messages):
                return _parse_json(self.generate(messages))

        return Client()

    def evotool_instance(self, task_id):
        sample = self.samples[task_id]
        messages, _ = self.runner.build_messages(sample, self.base)
        descriptions = load_api_descriptions(self.args.api_bank_root, sample["apis"])
        tools = [json.loads(value) for value in descriptions.values()]
        for tool in tools:
            tool["parameters"] = tool.get("input_parameters", {})
        return dict(id=task_id, query=_messages_to_user_prompt(messages[1:]),
                    available_tools=tools, benchmark="api_bank_next_call")

    def evotool_evaluate(self, prompt, task_id):
        modules, _, Step = self.evotool_imports()
        specs = json.loads(prompt)["evotool_policy"]
        path = self.root / "evaluations" / (digest([prompt, task_id]) + ".json")
        if path.exists():
            return json.loads(path.read_text())
        instance = self.evotool_instance(task_id)
        client = self.evotool_client(max_tokens=256)
        # API-Bank grades one next call, not the remaining multi-step conversation.
        query = "Predict only the next API call in the supplied conversation. This year is 2023.\n" + instance["query"]
        plan = modules.run_planner(client, specs["planner"], query)
        steps = []
        if plan:
            subgoal = plan[0]
            tool_name = modules.run_selector(client, specs["selector"], query, subgoal, {}, instance["available_tools"])
            tool = next((t for t in instance["available_tools"] if t["name"] == tool_name), None)
            arguments = modules.run_caller(client, specs["caller"], tool, query, subgoal, {}) if tool else {}
            steps.append(Step(subgoal, tool_name, arguments,
                              dict(status="no_execution", output=None)))
        visible_steps = [dict(subgoal=s.subgoal, tool=s.tool, arguments=s.args) for s in steps]
        prediction = self.llm(query + "\nProposed next call:\n" + json.dumps(visible_steps),
            system=specs["synthesizer"] + "\n" + self.base, max_tokens=256).strip()
        analysis = _analyze_api_prediction(prediction, self.samples[task_id]["ground_truth"])
        row = dict(task_id=task_id, prompt_sha256=digest(prompt), prediction=prediction,
                   success=bool(analysis["ok"]), analysis=analysis, plan=plan,
                   steps=[dict(subgoal=s.subgoal, tool=s.tool, args=s.args, observation=s.observation) for s in steps])
        save(path, row)
        return row

    def evotool(self):
        from dataclasses import asdict
        self.provenance("evotool", "ACL_2026_EvoTool",
            "Official blame/mutation/diversity search, 3 epochs. API-Bank next-call adaptation: "
            "planner/selector/caller retained, max_steps=1, synthesizer emits bracketed API call, "
            "no tool execution or gold references in optimizer prompts. Separate modular baseline required; "
            "not same actor-call budget as single-call GEPA/SCOPE/AHO.")
        modules, Episode, Step = self.evotool_imports()
        from src.config import EvoToolConfig
        from src.evolve import loop, select
        from src.policy import agent
        outer = self

        def encode(policy):
            return json.dumps({"evotool_policy": {key: getattr(policy, key) for key in modules.MODULES}}, sort_keys=True)

        def episode(client, policy, instance, max_steps):
            row = outer.evotool_evaluate(encode(policy), instance["id"])
            return Episode(instance=instance, plan=row["plan"], steps=[Step(**s) for s in row["steps"]],
                           answer=row["prediction"], reward=float(row["success"]), success=row["success"])

        def reward_matrix(client, population, instances, max_steps):
            ids = [instance["id"] for instance in instances]
            return [[float(row["success"]) for row in self.evaluate(encode(policy), ids)]
                    for policy in population]

        originals = loop.run_episode, select.run_episode, agent.run_episode, select._reward_matrix
        loop.run_episode = select.run_episode = agent.run_episode = episode
        # Only scheduling changes: preserve matrix order and official selection decisions.
        select._reward_matrix = reward_matrix
        cfg = EvoToolConfig(seed=self.args.seed, benchmark="api_bank_next_call", max_steps=1)
        cfg.evolve.log_test_eval = False
        client = self.evotool_client()
        train = [self.evotool_instance(task) for task in self.splits["train"]]
        dev = [self.evotool_instance(task) for task in self.splits["dev"]]
        try:
            population = loop.evolve(client, cfg, train, dev,
                log=lambda text: self.state("running", phase="evotool_search", detail=text))
            best = select.best_policy(client, cfg, population, dev)
        finally:
            loop.run_episode, select.run_episode, agent.run_episode, select._reward_matrix = originals
        save(self.root / "evotool" / "population.json", [asdict(p) for p in population])
        baseline = self.evaluate(encode(modules.initial_policy()), self.splits["test"])
        save(self.root / "evotool" / "modular_base_test_results.json", baseline)
        return encode(best)


def run_comparison(args):
    import fcntl
    comparison = Comparison(args)
    with (comparison.root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for method in args.methods:
            comparison.method = method
            if ((comparison.root / method / "summary.json").exists()
                    and (comparison.root / method / "full_set_summary.json").exists()):
                continue
            (comparison.root / method / "provider_error.json").unlink(missing_ok=True)
            comparison.state("running", phase="optimization")
            try:
                selected = comparison.root / method / "selected_prompt.json"
                if selected.exists():
                    prompt = json.loads(selected.read_text())["prompt"]
                else:
                    prompt = comparison.base if method == "base" else getattr(comparison, method)()
                comparison.finish(prompt)
            except BaseException as exc:
                comparison.state("failed", error_type=type(exc).__name__)
                raise
        comparison.state("queue_complete", methods=args.methods)


def follow_full_evaluations(args):
    """Attach to already-running old workers without restarting their optimization."""
    import fcntl
    comparison = Comparison(args)
    deadline = time.monotonic() + 72 * 3600
    pending = set(args.methods)
    with (comparison.root / "full_eval_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while pending and time.monotonic() < deadline:
            for method in args.methods:
                if method not in pending:
                    continue
                directory = comparison.root / method
                if (directory / "full_set_summary.json").exists():
                    pending.remove(method)
                    continue
                selected = directory / "selected_prompt.json"
                if not selected.exists() or not (directory / "summary.json").exists():
                    continue
                comparison.method = method
                try:
                    comparison.full_evaluation(json.loads(selected.read_text())["prompt"])
                except BaseException as exc:
                    save(directory / "full_eval_status.json", dict(status="failed", error_type=type(exc).__name__))
                    raise
                pending.remove(method)
            save(comparison.root / "full_eval_followup_status.json", dict(
                status="waiting" if pending else "complete", pending=sorted(pending),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            summaries = {}
            for method in args.methods:
                path = comparison.root / method / "full_set_summary.json"
                if path.exists():
                    summaries[method] = json.loads(path.read_text())
            save(comparison.root / "full_set_comparison.json", dict(
                status="partial" if pending else "complete", methods=summaries))
            if pending and (comparison.root / "queue_summary.json").exists():
                save(comparison.root / "full_eval_followup_status.json", dict(
                    status="incomplete_upstream", pending=sorted(pending)))
                return
            if pending:
                time.sleep(30)
        if pending:
            save(comparison.root / "full_eval_followup_status.json", dict(status="timed_out", pending=sorted(pending)))
