"""Metric-provider implementation for API-Bank."""
from __future__ import annotations

from collections import Counter
import os
from typing import Any, Callable, Optional

from ...interfaces import MetricSpec, TaskMetric
from .api_utils import (
    _analyze_api_prediction,
    _bucket_count,
    _bucket_history,
    _official_execute_prediction,
    _rouge_l_f1,
    _safe_div,
    _tool_search_enabled,
    samples_from_history,
)
from .constants import _OFFICIAL_ERROR_CODES
from .files import (
    _default_data_dir,
    _default_prediction_path,
    _iter_jsonl_files,
    _iter_rollout_rows,
    _prediction_key,
    _read_jsonl,
    load_predictions,
)

class APIBankMetricProvider:
    """Compute API-Bank metrics from prediction JSONL files."""

    def __init__(
        self,
        data_dir_fn=_default_data_dir,
        prediction_path_fn=_default_prediction_path,
        rollout_path_fn: Optional[Callable[[str], str]] = None,
        include_missing: bool = False,
        include_responses: bool = False,
        response_success_threshold: float = 0.2,
        api_bank_root: str = "",
        execute_api_calls: bool = False,
    ):
        self._data_dir = data_dir_fn
        self._predictions = prediction_path_fn
        self._rollouts = rollout_path_fn
        self.include_missing = include_missing
        self.include_responses = include_responses
        self.response_success_threshold = response_success_threshold
        self.api_bank_root = api_bank_root
        self.execute_api_calls = execute_api_calls

    def _rollout_rows(self, experiment: str) -> dict[tuple[str, int], dict]:
        rollout_rows = {}
        if not self._rollouts:
            return rollout_rows
        rollout_path = self._rollouts(experiment)
        if rollout_path and os.path.exists(rollout_path):
            for row in _iter_rollout_rows(rollout_path):
                if "file" in row and "id" in row:
                    rollout_rows[_prediction_key(str(row["file"]), int(row["id"]))] = row
        return rollout_rows

    def _sample_counts(self, data_dir: str) -> Counter:
        counts: Counter = Counter()
        for path in _iter_jsonl_files(data_dir):
            history = _read_jsonl(path)
            for sample in samples_from_history(path, history):
                counts[str(sample["kind"])] += 1
        return counts

    def _api_task_metric(
        self,
        sample: dict,
        pred_text: str,
        rollout: dict,
        data_dir: str,
        has_prediction: bool,
    ) -> TaskMetric:
        analysis = _analyze_api_prediction(pred_text, sample["ground_truth"])
        ok = bool(analysis["ok"])
        flags = dict(analysis["flags"])
        flags["runtime_error"] = bool(rollout.get("error"))
        history_turns = len(sample["chat_history"])
        extra = {
            "kind": "api_call",
            "file": sample["file"],
            "sample_id": sample["id"],
            "has_prediction": has_prediction,
            "apis": sample["apis"],
            "tool_search_enabled": _tool_search_enabled(data_dir),
            "gold_api": analysis["gold_api"],
            "pred_api": analysis["pred_api"],
            "gold_args": analysis["gold_args"],
            "pred_args": analysis["pred_args"],
            "reason": analysis["reason"],
            "exact_match_ok": bool(analysis["ok"]),
            "parse_error": None if analysis["parse_success"] else analysis["reason"],
            "n_history_turns": history_turns,
            "history_bucket": _bucket_history(history_turns),
            "arg_count_bucket": _bucket_count(int(analysis["n_gold_args"])),
            "prediction_chars": len(pred_text),
            "latency_s": float(rollout.get("latency_s") or 0.0),
            "runtime_error": rollout.get("error") or "",
            **{k: v for k, v in analysis.items() if k not in {"ok", "flags"}},
        }
        if self.execute_api_calls:
            root = self.api_bank_root or APIBankRolloutRunner._infer_api_bank_root(data_dir)
            official = _official_execute_prediction(pred_text, sample["ground_truth"], root)
            ok = bool(official["ok"])
            flags["official_error"] = not ok
            for code in _OFFICIAL_ERROR_CODES:
                flags[f"official_{code.lower()}"] = official["code"] == code
            extra.update(
                {
                    "official_ok": ok,
                    "official_error_code": official["code"],
                    "official_error": official["error"],
                    "official_result": official["result"],
                }
            )
        return TaskMetric(
            task_id=sample["task_id"],
            success=ok,
            tool_f1=float(analysis["argument_key_f1"]),
            failure_flags=flags,
            extra=extra,
        )

    def _response_task_metric(self, sample: dict, pred_text: str, rollout: dict, has_prediction: bool) -> TaskMetric:
        gold = str(sample["ground_truth"].get("text") or "")
        score = round(_rouge_l_f1(gold, pred_text), 4)
        flags = {
            "empty_prediction": not bool((pred_text or "").strip()),
            "low_response_rouge": score < self.response_success_threshold,
            "runtime_error": bool(rollout.get("error")),
        }
        history_turns = len(sample["chat_history"])
        return TaskMetric(
            task_id=sample["task_id"],
            success=score >= self.response_success_threshold,
            tool_f1=0.0,
            failure_flags=flags,
            extra={
                "kind": "response",
                "file": sample["file"],
                "sample_id": sample["id"],
                "has_prediction": has_prediction,
                "gold_response": gold,
                "pred_response": pred_text,
                "response_rouge_l": score,
                "n_history_turns": history_turns,
                "history_bucket": _bucket_history(history_turns),
                "prediction_chars": len(pred_text),
                "latency_s": float(rollout.get("latency_s") or 0.0),
                "runtime_error": rollout.get("error") or "",
            },
        )

    def per_task(self, experiment: str) -> dict[str, TaskMetric]:
        data_dir = self._data_dir(experiment)
        predictions = load_predictions(self._predictions(experiment))
        rollout_rows = self._rollout_rows(experiment)
        out: dict[str, TaskMetric] = {}
        for path in _iter_jsonl_files(data_dir):
            history = _read_jsonl(path)
            for sample in samples_from_history(path, history):
                if sample["kind"] == "response" and not self.include_responses:
                    continue
                key = _prediction_key(sample["file"], sample["id"])
                pred = predictions.get(key, {})
                if not pred and not self.include_missing:
                    continue
                rollout = rollout_rows.get(key, {})
                pred_text = str(pred.get("pred") or "")
                if sample["kind"] == "api_call":
                    out[sample["task_id"]] = self._api_task_metric(sample, pred_text, rollout, data_dir, bool(pred))
                elif sample["kind"] == "response":
                    out[sample["task_id"]] = self._response_task_metric(sample, pred_text, rollout, bool(pred))
        return out

    def aggregate(self, experiment: str, task_ids: Optional[list[str]] = None) -> dict[str, Any]:
        data_dir = self._data_dir(experiment)
        metrics = self.per_task(experiment)
        if task_ids is not None:
            keep = set(task_ids)
            metrics = {k: v for k, v in metrics.items() if k in keep}
        vals = list(metrics.values())
        api_vals = [m for m in vals if m.extra.get("kind") == "api_call"]
        response_vals = [m for m in vals if m.extra.get("kind") == "response"]
        n = len(api_vals) or 1

        def rate(flag: str) -> float:
            return sum(m.failure_flags.get(flag, False) for m in api_vals) / n

        apis = Counter(str(m.extra.get("gold_api") or "") for m in api_vals)
        pred_apis = Counter(str(m.extra.get("pred_api") or "") for m in api_vals if m.extra.get("pred_api"))
        files = Counter(str(m.extra.get("file") or "") for m in vals)
        history_buckets = Counter(str(m.extra.get("history_bucket") or "") for m in api_vals)
        arg_buckets = Counter(str(m.extra.get("arg_count_bucket") or "") for m in api_vals)

        def avg_extra(name: str) -> float:
            return sum(float(m.extra.get(name) or 0) for m in api_vals) / n

        def bucket_accuracy(field: str, bucket: str) -> float:
            subset = [m for m in api_vals if m.extra.get(field) == bucket]
            return sum(m.success for m in subset) / len(subset) if subset else 0.0

        per_api_success = {}
        for api in apis:
            subset = [m for m in api_vals if str(m.extra.get("gold_api") or "") == api]
            if subset:
                per_api_success[api] = sum(m.success for m in subset) / len(subset)
        worst_api_success = min(per_api_success.values()) if per_api_success else 0.0
        macro_api_accuracy = sum(per_api_success.values()) / len(per_api_success) if per_api_success else 0.0

        sample_counts = self._sample_counts(data_dir)
        expected_api = sample_counts.get("api_call", 0)
        expected_response = sample_counts.get("response", 0)
        response_n = len(response_vals) or 1
        out = {
            "n": len(api_vals),
            "n_total_evaluated": len(vals),
            "n_expected_api_calls": expected_api,
            "n_expected_responses": expected_response,
            "prediction_coverage_rate": _safe_div(sum(bool(m.extra.get("has_prediction")) for m in api_vals), expected_api),
            "missing_prediction_rate": 1.0 - _safe_div(sum(bool(m.extra.get("has_prediction")) for m in api_vals), expected_api),
            "success_rate": sum(m.success for m in api_vals) / n,
            "tool_f1": sum(m.tool_f1 for m in api_vals) / n,
            "api_call_accuracy": sum(m.success for m in api_vals) / n,
            "exact_match_accuracy": avg_extra("exact_match_ok"),
            "parse_success_rate": avg_extra("parse_success"),
            "called_api_rate": avg_extra("called_api"),
            "api_name_accuracy": avg_extra("api_name_correct"),
            "argument_key_precision": avg_extra("argument_key_precision"),
            "argument_key_recall": avg_extra("argument_key_recall"),
            "argument_key_f1": avg_extra("argument_key_f1"),
            "argument_value_accuracy": avg_extra("argument_value_accuracy"),
            "full_argument_accuracy": avg_extra("full_argument_accuracy"),
            "no_api_call_rate": rate("no_api_call"),
            "format_error_rate": rate("format_error"),
            "empty_prediction_rate": rate("empty_prediction"),
            "api_name_mismatch_rate": rate("api_name_mismatch"),
            "missing_argument_rate": rate("missing_argument"),
            "extra_argument_rate": rate("extra_argument"),
            "argument_value_mismatch_rate": rate("argument_value_mismatch"),
            "runtime_error_rate": rate("runtime_error"),
            "avg_history_turns": avg_extra("n_history_turns"),
            "avg_gold_args": avg_extra("n_gold_args"),
            "avg_pred_args": avg_extra("n_pred_args"),
            "avg_missing_args": avg_extra("n_missing_args"),
            "avg_extra_args": avg_extra("n_extra_args"),
            "avg_mismatched_args": avg_extra("n_mismatched_args"),
            "avg_prediction_chars": avg_extra("prediction_chars"),
            "avg_latency_s": avg_extra("latency_s"),
            "n_unique_gold_apis": len([api for api in apis if api]),
            "n_unique_pred_apis": len([api for api in pred_apis if api]),
            "n_files": len([file for file in files if file]),
            "macro_api_accuracy": macro_api_accuracy,
            "worst_api_accuracy": worst_api_success,
            "history_0_2_accuracy": bucket_accuracy("history_bucket", "0_2"),
            "history_3_6_accuracy": bucket_accuracy("history_bucket", "3_6"),
            "history_7_10_accuracy": bucket_accuracy("history_bucket", "7_10"),
            "history_11_plus_accuracy": bucket_accuracy("history_bucket", "11_plus"),
            "args_0_accuracy": bucket_accuracy("arg_count_bucket", "0"),
            "args_1_accuracy": bucket_accuracy("arg_count_bucket", "1"),
            "args_2_3_accuracy": bucket_accuracy("arg_count_bucket", "2_3"),
            "args_4_plus_accuracy": bucket_accuracy("arg_count_bucket", "4_plus"),
            "n_history_buckets": len([b for b in history_buckets if b]),
            "n_arg_count_buckets": len([b for b in arg_buckets if b]),
        }
        if self.execute_api_calls:
            out["official_execution_accuracy"] = avg_extra("official_ok")
            for code in _OFFICIAL_ERROR_CODES:
                out[f"official_{code.lower()}_rate"] = rate(f"official_{code.lower()}")
        if self.include_responses:
            out.update(
                {
                    "n_responses": len(response_vals),
                    "response_prediction_coverage_rate": _safe_div(sum(bool(m.extra.get("has_prediction")) for m in response_vals), expected_response),
                    "response_rouge_l": sum(float(m.extra.get("response_rouge_l") or 0) for m in response_vals) / response_n,
                    "response_success_rate": sum(m.success for m in response_vals) / response_n,
                    "low_response_rouge_rate": sum(m.failure_flags.get("low_response_rouge", False) for m in response_vals) / response_n,
                }
            )
        return out

    def metric_specs(self) -> list[MetricSpec]:
        return [
            MetricSpec("success_rate", "Same as API-call accuracy for API-Bank call-prediction samples.", "higher_better"),
            MetricSpec("api_call_accuracy", "API-call task success. Exact match by default; official execution correctness when execute_api_calls=True.", "higher_better"),
            MetricSpec("exact_match_accuracy", "Exact API name and argument string match against the gold API call.", "higher_better"),
            MetricSpec("official_execution_accuracy", "API-Bank evaluator-style execution correctness; present when execute_api_calls=True.", "higher_better"),
            MetricSpec("parse_success_rate", "Prediction parses as one bracketed API call.", "higher_better"),
            MetricSpec("called_api_rate", "Prediction contains an API call rather than empty or natural-language text.", "higher_better"),
            MetricSpec("api_name_accuracy", "Predicted API name equals the gold API name.", "higher_better"),
            MetricSpec("argument_key_precision", "Fraction of predicted argument keys that are gold keys.", "higher_better"),
            MetricSpec("argument_key_recall", "Fraction of gold argument keys present in the prediction.", "higher_better"),
            MetricSpec("argument_key_f1", "F1 over predicted vs gold argument keys.", "higher_better"),
            MetricSpec("argument_value_accuracy", "Fraction of gold arguments with exactly correct values.", "higher_better"),
            MetricSpec("full_argument_accuracy", "All and only gold arguments are present with exact values.", "higher_better"),
            MetricSpec("no_api_call_rate", "No bracketed API call was found in the prediction.", "lower_better"),
            MetricSpec("format_error_rate", "Prediction did not parse as [ApiName(key='value')].", "lower_better"),
            MetricSpec("empty_prediction_rate", "The model returned an empty string.", "lower_better"),
            MetricSpec("api_name_mismatch_rate", "Prediction called the wrong API.", "lower_better"),
            MetricSpec("missing_argument_rate", "Prediction omitted at least one required gold argument.", "lower_better"),
            MetricSpec("extra_argument_rate", "Prediction included arguments absent from the gold call.", "lower_better"),
            MetricSpec("argument_value_mismatch_rate", "Prediction used an incorrect value for a gold argument.", "lower_better"),
            MetricSpec("runtime_error_rate", "Runner caught an exception for the sample.", "lower_better"),
            MetricSpec("prediction_coverage_rate", "Fraction of expected API-call samples that have predictions.", "higher_better"),
            MetricSpec("missing_prediction_rate", "Fraction of expected API-call samples with no prediction row.", "lower_better"),
            MetricSpec("avg_missing_args", "Average number of missing gold arguments.", "lower_better"),
            MetricSpec("avg_extra_args", "Average number of extra predicted arguments.", "lower_better"),
            MetricSpec("avg_mismatched_args", "Average number of matched-key arguments with wrong values.", "lower_better"),
            MetricSpec("avg_history_turns", "Average dialogue history length before the target API call.", "neutral"),
            MetricSpec("avg_gold_args", "Average number of gold API arguments.", "neutral"),
            MetricSpec("avg_pred_args", "Average number of predicted API arguments.", "neutral"),
            MetricSpec("avg_prediction_chars", "Average output length in characters.", "neutral"),
            MetricSpec("avg_latency_s", "Average model-call latency in seconds when rollout data is available.", "neutral"),
            MetricSpec("n_unique_gold_apis", "Number of distinct gold APIs in the evaluated set.", "neutral"),
            MetricSpec("n_unique_pred_apis", "Number of distinct predicted APIs in the evaluated set.", "neutral"),
            MetricSpec("macro_api_accuracy", "Mean exact-call accuracy averaged equally over gold APIs.", "higher_better"),
            MetricSpec("worst_api_accuracy", "Worst exact-call accuracy among APIs represented in the run.", "higher_better"),
            MetricSpec("history_0_2_accuracy", "Exact-call accuracy for samples with 0-2 prior history turns.", "higher_better"),
            MetricSpec("history_3_6_accuracy", "Exact-call accuracy for samples with 3-6 prior history turns.", "higher_better"),
            MetricSpec("history_7_10_accuracy", "Exact-call accuracy for samples with 7-10 prior history turns.", "higher_better"),
            MetricSpec("history_11_plus_accuracy", "Exact-call accuracy for samples with 11+ prior history turns.", "higher_better"),
            MetricSpec("args_0_accuracy", "Exact-call accuracy for samples with no gold arguments.", "higher_better"),
            MetricSpec("args_1_accuracy", "Exact-call accuracy for samples with one gold argument.", "higher_better"),
            MetricSpec("args_2_3_accuracy", "Exact-call accuracy for samples with 2-3 gold arguments.", "higher_better"),
            MetricSpec("args_4_plus_accuracy", "Exact-call accuracy for samples with 4+ gold arguments.", "higher_better"),
            MetricSpec("response_rouge_l", "Mean Rouge-L F1 over post-API dialogue responses; present when include_responses=True.", "higher_better"),
            MetricSpec("response_success_rate", "Fraction of response samples above the configured Rouge-L threshold.", "higher_better"),
            MetricSpec("low_response_rouge_rate", "Fraction of response samples below the configured Rouge-L threshold.", "lower_better"),
        ]
