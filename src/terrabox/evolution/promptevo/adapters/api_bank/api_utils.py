"""API-Bank parsing, prompt-context, and scoring helpers."""
from __future__ import annotations

import ast
import copy
from functools import lru_cache
import glob
import hashlib
import json
import os
import re
import sys
import types
from typing import Any, Iterable, Optional

from ...interfaces import Step
from .constants import _API_CALL_RE, _OFFICIAL_ERROR_CODES
from .files import _iter_jsonl_files, _read_jsonl, _sample_file_id, _sample_task_id

def _split_top_level_params(params: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: Optional[str] = None
    escaped = False
    depth = 0
    for ch in params:
        if quote:
            current.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            current.append(ch)
            continue
        if ch in "([{":
            depth += 1
            current.append(ch)
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            current.append(ch)
            continue
        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _strip_api_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_api_call(text: str) -> tuple[Optional[str], dict[str, str], Optional[str]]:
    match = _API_CALL_RE.search(text or "")
    if not match:
        return None, {}, "NO_API_CALL"
    api_name = match.group(1)
    params = (match.group(2) or "").strip()
    parsed: dict[str, str] = {}
    if not params:
        return api_name, parsed, None
    for part in _split_top_level_params(params):
        if "=" not in part:
            return api_name, parsed, "FAILED_PARSE_API_CALL"
        key, value = part.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"\w+", key):
            return api_name, parsed, "FAILED_PARSE_API_CALL"
        parsed[key] = _strip_api_value(value)
    return api_name, parsed, None


def _format_api_call(api_name: str, params: dict[str, Any]) -> str:
    args = ", ".join(f"{key}={_format_api_arg(value)}" for key, value in params.items())
    return f"[{api_name}({args})]"


def _format_api_arg(value: Any) -> str:
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        return text
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return repr(text)


def _safe_literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _api_class_info(path: str) -> Optional[dict[str, Any]]:
    """Extract API metadata from one API-Bank `apis/*.py` file without import."""
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
    except Exception:
        return None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        info: dict[str, Any] = {"name": node.name}
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "description",
                    "input_parameters",
                    "output_parameters",
                }:
                    info[target.id] = _safe_literal_eval(stmt.value)
        if {"description", "input_parameters", "output_parameters"} <= set(info):
            return info
    return None


def load_api_descriptions(api_bank_root: str, api_names: Iterable[str]) -> dict[str, str]:
    """Return API-Bank ToolManager-style JSON descriptions without imports.

    Importing API-Bank's ToolManager initializes all APIs and optional retrieval
    dependencies. For prompt-only rollout we only need the static metadata.
    """
    apis_dir = os.path.join(api_bank_root, "apis")
    wanted = set(api_names)
    out: dict[str, str] = {}
    for path in glob.glob(os.path.join(apis_dir, "*.py")):
        if os.path.basename(path) in {"__init__.py", "api.py"}:
            continue
        info = _api_class_info(path)
        if not info or info["name"] not in wanted:
            continue
        out[str(info["name"])] = json.dumps(info, ensure_ascii=False)
    return out


def _tool_search_enabled(data_dir: str) -> bool:
    return not os.path.basename(os.path.abspath(data_dir)).endswith("given-desc")


@lru_cache(maxsize=4)
def _get_tool_manager(api_bank_root: str) -> Any:
    root = os.path.abspath(api_bank_root)
    old_cwd = os.getcwd()
    if root not in sys.path:
        sys.path.insert(0, root)
    if "sentence_transformers" not in sys.modules:
        try:
            __import__("sentence_transformers")
        except ModuleNotFoundError:
            stub = types.ModuleType("sentence_transformers")

            class _MissingSentenceTransformer:
                def __init__(self, *args, **kwargs):
                    raise ModuleNotFoundError("No module named 'sentence_transformers'")

            stub.SentenceTransformer = _MissingSentenceTransformer
            stub.util = types.SimpleNamespace(cos_sim=lambda *args, **kwargs: None)
            sys.modules["sentence_transformers"] = stub
    if "rank_bm25" not in sys.modules:
        try:
            __import__("rank_bm25")
        except ModuleNotFoundError:
            stub = types.ModuleType("rank_bm25")

            class _MissingBM25Okapi:
                def __init__(self, *args, **kwargs):
                    raise ModuleNotFoundError("No module named 'rank_bm25'")

            stub.BM25Okapi = _MissingBM25Okapi
            sys.modules["rank_bm25"] = stub
    try:
        os.chdir(root)
        from tool_manager import ToolManager
        return ToolManager()
    finally:
        os.chdir(old_cwd)


def _classify_official_exception(exc: BaseException) -> str:
    text = str(exc)
    if "Parse API Call Error" in text:
        return "FAILED_PARSE_API_CALL"
    if "missing" in text and "required positional argument" in text:
        return "MISS_INPUT_ARGUMENT"
    if "invalid parameter name" in text:
        return "INVALID_INPUT_PARAMETER"
    if "The API name is not correct" in text or "invalid tool name" in text:
        return "API_NAME_MISMATCH"
    return "HAS_EXCEPTION"


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


@lru_cache(maxsize=4)
def _load_init_databases(api_bank_root: str) -> dict[str, Any]:
    init_dir = os.path.join(os.path.abspath(api_bank_root), "init_database")
    databases: dict[str, Any] = {}
    if not os.path.isdir(init_dir):
        return databases
    for path in glob.glob(os.path.join(init_dir, "*.json")):
        with open(path, encoding="utf-8") as f:
            databases[os.path.splitext(os.path.basename(path))[0]] = json.load(f)
    return databases


@lru_cache(maxsize=256)
def _api_module_name(api_bank_root: str, api_name: str) -> str:
    apis_dir = os.path.join(os.path.abspath(api_bank_root), "apis")
    fallback = _camel_to_snake(api_name)
    for path in glob.glob(os.path.join(apis_dir, "*.py")):
        if os.path.basename(path) in {"__init__.py", "api.py"}:
            continue
        info = _api_class_info(path)
        if info and info.get("name") == api_name:
            return os.path.splitext(os.path.basename(path))[0]
    return fallback


def _import_api_class(api_bank_root: str, api_name: str) -> type:
    import importlib

    root = os.path.abspath(api_bank_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    old_cwd = os.getcwd()
    try:
        os.chdir(root)
        module = importlib.import_module(f"apis.{_api_module_name(root, api_name)}")
        return getattr(module, api_name)
    finally:
        os.chdir(old_cwd)


def _new_api_instance(api_bank_root: str, api_name: str, token_checker: Any = None) -> Any:
    cls = _import_api_class(api_bank_root, api_name)
    args = []
    db_name = getattr(cls, "database_name", None)
    databases = _load_init_databases(api_bank_root)
    if db_name in databases:
        args.append(copy.deepcopy(databases[db_name]))
    input_parameters = getattr(cls, "input_parameters", {})
    if api_name != "CheckToken" and "token" in input_parameters:
        args.append(token_checker or _new_api_instance(api_bank_root, "CheckToken"))
    return cls(*args)


def _coerce_api_params(api_cls: type, params: dict[str, str]) -> dict[str, Any]:
    input_parameters = getattr(api_cls, "input_parameters", {})
    processed: dict[str, Any] = {}
    for input_key, input_value in params.items():
        assert input_key in input_parameters, f"invalid parameter name. parameter: {input_key}"
        required_type = input_parameters[input_key].get("type")
        if required_type == "int":
            if isinstance(input_value, str):
                assert input_value.isdigit(), f"invalid parameter type. parameter: {input_value}"
            processed[input_key] = int(input_value)
        elif required_type == "float":
            if isinstance(input_value, str):
                assert input_value.replace(".", "", 1).isdigit(), "invalid parameter type."
            processed[input_key] = float(input_value)
        elif required_type == "bool":
            processed[input_key] = input_value == "True"
        elif required_type in {"str", "list", "list(str)"}:
            processed[input_key] = input_value
        else:
            raise Exception("invalid parameter type.")
    return processed


def _lazy_api_call(api_bank_root: str, api_name: str, params: dict[str, str]) -> tuple[Any, Any]:
    api_cls = _import_api_class(api_bank_root, api_name)
    token_checker = _new_api_instance(api_bank_root, "CheckToken") if api_name != "CheckToken" and "token" in getattr(api_cls, "input_parameters", {}) else None
    api = _new_api_instance(api_bank_root, api_name, token_checker=token_checker)
    processed = _coerce_api_params(api_cls, params)
    return api, api.call(**processed)


def _official_execute_prediction(prediction: str, ground_truth: dict, api_bank_root: str) -> dict[str, Any]:
    pred_name, pred_params, parse_error = _parse_api_call(prediction)
    if parse_error == "NO_API_CALL":
        return {"ok": False, "code": "NO_API_CALL", "result": None, "error": ""}
    if parse_error:
        return {"ok": False, "code": "FAILED_PARSE_API_CALL", "result": None, "error": parse_error}

    gold_name = str(ground_truth.get("api_name") or "")
    if pred_name != gold_name:
        return {
            "ok": False,
            "code": "API_NAME_MISMATCH",
            "result": None,
            "error": f"{pred_name}!={gold_name}",
        }

    try:
        api, result = _lazy_api_call(api_bank_root, str(pred_name), pred_params)
        try:
            correct = bool(api.check_api_call_correctness(result, ground_truth.get("result") or {}))
        except KeyError as exc:
            return {"ok": False, "code": "KEY_ERROR", "result": result, "error": repr(exc)}
    except AssertionError as exc:
        return {"ok": False, "code": _classify_official_exception(exc), "result": None, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "code": _classify_official_exception(exc), "result": None, "error": repr(exc)}

    if correct:
        return {"ok": True, "code": "OK", "result": result, "error": ""}
    if isinstance(result, dict):
        if result.get("exception"):
            return {"ok": False, "code": "HAS_EXCEPTION", "result": result, "error": str(result.get("exception"))}
        gold_result = ground_truth.get("result") or {}
        if result.get("output") != gold_result.get("output"):
            return {"ok": False, "code": "OUTPUT_MISMATCH", "result": result, "error": ""}
        if result.get("input") != gold_result.get("input"):
            return {"ok": False, "code": "INPUT_MISMATCH", "result": result, "error": ""}
    if isinstance(result, str) and result.startswith("KeyError"):
        return {"ok": False, "code": "KEY_ERROR", "result": result, "error": result}
    return {"ok": False, "code": "OUTPUT_MISMATCH", "result": result, "error": ""}


def _rouge_l_f1(reference: str, hypothesis: str) -> float:
    ref = (reference or "").split()
    hyp = (hypothesis or "").split()
    if not ref or not hyp:
        return 0.0
    prev = [0] * (len(hyp) + 1)
    for token in ref:
        cur = [0] * (len(hyp) + 1)
        for j, other in enumerate(hyp, start=1):
            cur[j] = prev[j - 1] + 1 if token == other else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    recall = lcs / len(ref)
    precision = lcs / len(hyp)
    return _safe_div(2 * precision * recall, precision + recall)


def _api_result_message(item: dict) -> str:
    api_name = str(item.get("api_name") or "")
    params = item.get("param_dict") or {}
    args = ", ".join(f"{k}={_format_api_arg(v)}" for k, v in params.items())
    result = item.get("result") or {}
    output = result.get("output") if isinstance(result, dict) else result
    return f"[{api_name}({args})] Response: {output}"


def build_api_call_messages(
    static_instruction: str,
    api_descriptions: str,
    chat_history: list[dict],
) -> list[dict[str, str]]:
    """Build API-Bank evaluator-style chat messages.

    `static_instruction` is the only promptevo target. API descriptions and
    dialogue history are dynamic task context and must not be saved as prompt
    versions.
    """
    messages = [{"role": "system", "content": static_instruction.rstrip() + "\n" + api_descriptions}]
    for item in chat_history:
        role = item.get("role")
        if role == "User":
            messages.append({"role": "user", "content": str(item.get("text") or "")})
        elif role == "AI":
            messages.append({"role": "assistant", "content": str(item.get("text") or "")})
        elif role == "API":
            messages.append({"role": "system", "content": _api_result_message(item)})
    return messages


def _hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _api_descriptions_from_history(history: list[dict]) -> list[str]:
    names = []
    seen = set()
    for item in history:
        if item.get("role") == "API":
            name = str(item.get("api_name") or "")
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    return names


def samples_from_history(path: str, history: list[dict]) -> list[dict]:
    """Return API-Bank evaluator-style samples from a full conversation."""
    api_names = set(_api_descriptions_from_history(history))
    samples = []
    for idx, item in enumerate(history):
        if item.get("role") == "API":
            samples.append(
                {
                    "task_id": _sample_task_id(path, len(samples)),
                    "file": _sample_file_id(path),
                    "id": len(samples),
                    "kind": "api_call",
                    "chat_history": history[:idx],
                    "apis": sorted(api_names),
                    "ground_truth": item,
                }
            )
            if idx + 1 < len(history) and history[idx + 1].get("role") == "AI":
                samples.append(
                    {
                        "task_id": _sample_task_id(path, len(samples)),
                        "file": _sample_file_id(path),
                        "id": len(samples),
                        "kind": "response",
                        "chat_history": history[: idx + 1],
                        "apis": sorted(api_names),
                        "ground_truth": history[idx + 1],
                    }
                )
    return samples


def _steps_from_history(history: list[dict], prediction: str = "") -> list[Step]:
    steps: list[Step] = []
    for item in history:
        role = item.get("role")
        if role == "User":
            steps.append(Step(role="user", text=str(item.get("text") or "")))
        elif role == "AI":
            steps.append(Step(role="assistant", text=str(item.get("text") or "")))
        elif role == "API":
            api_name = str(item.get("api_name") or "")
            params = item.get("param_dict") or {}
            steps.append(Step(role="assistant", tool=api_name, args=params))
            steps.append(Step(role="tool", text=json.dumps(item.get("result"), ensure_ascii=False)))
    if prediction:
        api_name, params, err = _parse_api_call(prediction)
        if api_name:
            steps.append(Step(role="assistant", text=prediction, tool=api_name, args=params))
        else:
            steps.append(Step(role="assistant", text=prediction, errored=err is not None))
    return steps


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _bucket_count(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2_3"
    return "4_plus"


def _bucket_history(n: int) -> str:
    if n <= 2:
        return "0_2"
    if n <= 6:
        return "3_6"
    if n <= 10:
        return "7_10"
    return "11_plus"


def _analyze_api_prediction(prediction: str, ground_truth: dict) -> dict[str, Any]:
    gold_name = str(ground_truth.get("api_name") or "")
    gold_params = {str(k): str(v) for k, v in (ground_truth.get("param_dict") or {}).items()}
    pred_name, pred_params_raw, parse_error = _parse_api_call(prediction)
    pred_params = {str(k): str(v) for k, v in pred_params_raw.items()}

    gold_keys = set(gold_params)
    pred_keys = set(pred_params)
    matched_keys = gold_keys & pred_keys
    missing = gold_keys - pred_keys
    extra = pred_keys - gold_keys
    mismatched = {k for k in matched_keys if pred_params[k] != gold_params[k]}
    value_matches = matched_keys - mismatched

    flags = {
        "no_api_call": parse_error == "NO_API_CALL",
        "api_name_mismatch": False,
        "missing_argument": False,
        "extra_argument": False,
        "argument_value_mismatch": False,
        "format_error": parse_error is not None,
        "empty_prediction": not bool((prediction or "").strip()),
    }
    if parse_error:
        return {
            "ok": False,
            "reason": parse_error,
            "flags": flags,
            "gold_api": gold_name,
            "pred_api": pred_name,
            "gold_args": gold_params,
            "pred_args": pred_params,
            "missing_args": sorted(missing),
            "extra_args": sorted(extra),
            "mismatched_args": sorted(mismatched),
            "api_name_correct": False,
            "parse_success": False,
            "called_api": False,
            "argument_key_precision": 0.0,
            "argument_key_recall": 0.0,
            "argument_key_f1": 0.0,
            "argument_value_accuracy": 0.0,
            "full_argument_accuracy": 0.0,
            "n_gold_args": len(gold_params),
            "n_pred_args": len(pred_params),
            "n_missing_args": len(missing),
            "n_extra_args": len(extra),
            "n_mismatched_args": len(mismatched),
        }

    api_name_correct = pred_name == gold_name
    flags["api_name_mismatch"] = not api_name_correct
    flags["missing_argument"] = bool(missing)
    flags["extra_argument"] = bool(extra)
    flags["argument_value_mismatch"] = bool(mismatched)
    if not gold_keys and not pred_keys:
        key_precision = key_recall = key_f1 = value_accuracy = 1.0
    else:
        key_precision = _safe_div(len(matched_keys), len(pred_keys))
        key_recall = _safe_div(len(matched_keys), len(gold_keys))
        key_f1 = _safe_div(2 * key_precision * key_recall, key_precision + key_recall)
        value_accuracy = _safe_div(len(value_matches), len(gold_keys))
    full_argument_accuracy = float(not (missing or extra or mismatched))
    ok = api_name_correct and bool(full_argument_accuracy)
    reason = "OK" if ok else (
        f"API_NAME_MISMATCH:{pred_name}!={gold_name}" if not api_name_correct
        else f"ARG_MISMATCH missing={sorted(missing)} extra={sorted(extra)} mismatch={sorted(mismatched)}"
    )
    return {
        "ok": ok,
        "reason": reason,
        "flags": flags,
        "gold_api": gold_name,
        "pred_api": pred_name,
        "gold_args": gold_params,
        "pred_args": pred_params,
        "missing_args": sorted(missing),
        "extra_args": sorted(extra),
        "mismatched_args": sorted(mismatched),
        "api_name_correct": api_name_correct,
        "parse_success": True,
        "called_api": True,
        "argument_key_precision": key_precision,
        "argument_key_recall": key_recall,
        "argument_key_f1": key_f1,
        "argument_value_accuracy": value_accuracy,
        "full_argument_accuracy": full_argument_accuracy,
        "n_gold_args": len(gold_params),
        "n_pred_args": len(pred_params),
        "n_missing_args": len(missing),
        "n_extra_args": len(extra),
        "n_mismatched_args": len(mismatched),
    }


def _score_api_prediction(prediction: str, ground_truth: dict) -> tuple[bool, str, dict[str, bool]]:
    analysis = _analyze_api_prediction(prediction, ground_truth)
    return bool(analysis["ok"]), str(analysis["reason"]), dict(analysis["flags"])
