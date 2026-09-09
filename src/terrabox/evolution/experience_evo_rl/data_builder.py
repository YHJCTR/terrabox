"""Build strict no-label veRL data from OEA tasks and ExperienceEvo policies.

This module intentionally does not expose benchmark task ids, task types,
expected tools, gold calls, gold answers, or evaluation metrics to the prompt,
reward model, retrieval text, or ``extra_info``. The reward model receives only
ExperienceEvo rollout-derived policy summaries and public tool schemas.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TRAIN_DATA = REPO_ROOT / "data/oea_full_sft/openearth/train.jsonl"
DEFAULT_TOOL_CATALOG = REPO_ROOT / "data/oea_full_sft/tools_catalog.json"
DEFAULT_STORE = REPO_ROOT / "evolution_store/experience_evo/oea_train2000_v4_clean_longcat_20260814"
FORBIDDEN_KEYS = {
    "task_id",
    "id",
    "task_type",
    "type",
    "expected_tools",
    "gold_tool_calls",
    "ground_truth",
    "metrics",
    "f1",
    "final_answer",
}
ALLOWED_FORBIDDEN_KEY_PATHS = {
    ("reward_model", "ground_truth"),  # veRL-required name; value is policy JSON, not benchmark gold.
}
ALLOWED_FORBIDDEN_KEY_PARENTS = {
    "tool_schemas",  # Public JSON Schema uses keys such as "type".
}


@dataclass(frozen=True)
class PublicTask:
    question: str
    images: tuple[str, ...] = ()
    data_files: tuple[str, ...] = ()
    data_dir: str = ""


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        return [row for row in data["tasks"] if isinstance(row, dict)]
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    raise ValueError(f"Unsupported task file format: {path}")


def public_task_from_row(row: dict[str, Any]) -> PublicTask:
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError("Task row missing public question")
    images = tuple(str(x) for x in (row.get("images") or []) if x)
    data_files = tuple(str(x) for x in (row.get("data_files") or []) if x)
    data_dir = str(row.get("data_dir") or "")
    return PublicTask(question=question, images=images, data_files=data_files, data_dir=data_dir)


def load_public_tasks(path: str | Path, *, limit: int | None = None) -> list[PublicTask]:
    rows = _read_json_or_jsonl(Path(path))
    tasks = [public_task_from_row(row) for row in rows]
    return tasks[:limit] if limit is not None else tasks


def load_tool_catalog(path: str | Path = DEFAULT_TOOL_CATALOG) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected tool catalog list: {path}")
    return data


def _tool_call_name(tool: dict[str, Any], *, tool_name_style: str = "slug") -> str:
    slug = str(tool.get("slug") or "")
    if tool_name_style == "function_name":
        return str(tool.get("function_name") or slug.replace(".", "__"))
    return slug


def compact_tool_catalog(
    tool_catalog: list[dict[str, Any]], *, max_description_chars: int = 220, tool_name_style: str = "slug"
) -> str:
    lines: list[str] = []
    for tool in tool_catalog:
        name = _tool_call_name(tool, tool_name_style=tool_name_style)
        slug = str(tool.get("slug") or name)
        if not name:
            continue
        desc = str(tool.get("description") or "").replace("\n", " ").strip()
        if len(desc) > max_description_chars:
            desc = desc[: max_description_chars - 3].rstrip() + "..."
        params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
        required = params.get("required") if isinstance(params, dict) else []
        required = required if isinstance(required, list) else []
        suffix = f" Terrabox slug: {slug}." if tool_name_style == "function_name" and slug != name else ""
        lines.append(f"- {name}: {desc}{suffix} Required args: {required}")
    return "\n".join(lines)


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", text or "")}


def _safe_family_doc(family: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("intent_signature", "input_product_state", "target_product_state"):
        value = family.get(key)
        if isinstance(value, list):
            parts.extend(str(x) for x in value)
        elif value:
            parts.append(str(value))
    product = family.get("product_experience") if isinstance(family.get("product_experience"), dict) else {}
    for key in ("goal", "experience", "downstream_rule"):
        if product.get(key):
            parts.append(str(product[key]))
    for policy in family.get("tool_policies") or []:
        if not isinstance(policy, dict):
            continue
        for key in ("tool", "required_input_roles", "parameter_binding_rules", "output_contract", "experience"):
            value = policy.get(key)
            if isinstance(value, list):
                parts.extend(str(x) for x in value)
            elif value:
                parts.append(str(value))
    return " ".join(parts)


def _policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "tool",
        "required_input_roles",
        "parameter_binding_rules",
        "output_contract",
        "post_checks",
        "downstream_rule",
        "recovery",
        "experience",
        "q",
        "n",
        "risk",
    }
    return {k: v for k, v in policy.items() if k in allowed_keys}


def load_experience_families(store_dir: str | Path = DEFAULT_STORE, *, max_families: int | None = None) -> list[dict[str, Any]]:
    path = Path(store_dir) / "families_v2.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing ExperienceEvo families file: {path}")
    families: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            family = json.loads(line)
            if not isinstance(family, dict):
                continue
            product = family.get("product_experience") if isinstance(family.get("product_experience"), dict) else {}
            policies = [_policy_summary(p) for p in family.get("tool_policies") or [] if isinstance(p, dict)]
            clean = {
                "family_id": str(family.get("family_id") or ""),
                "intent_signature": str(family.get("intent_signature") or "general"),
                "input_product_state": list(family.get("input_product_state") or []),
                "target_product_state": list(family.get("target_product_state") or []),
                "product_experience": {
                    "goal": product.get("goal"),
                    "experience": product.get("experience"),
                    "downstream_rule": product.get("downstream_rule"),
                    "q": float(product.get("q") or 0.0),
                    "n": int(product.get("n") or 0),
                    "risk": float(product.get("risk") or 0.0),
                },
                "tool_policies": policies,
            }
            clean["_doc"] = _safe_family_doc(clean)
            clean["_tokens"] = _tokens(clean["_doc"])
            families.append(clean)
            if max_families and len(families) >= max_families:
                break
    return families


def retrieve_families(question: str, families: list[dict[str, Any]], *, top_k: int = 5) -> list[dict[str, Any]]:
    q_tokens = _tokens(question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for fam in families:
        fam_tokens = fam.get("_tokens") or set()
        overlap = len(q_tokens & fam_tokens) / math.sqrt(max(1, len(q_tokens)) * max(1, len(fam_tokens)))
        product = fam.get("product_experience") or {}
        prior = float(product.get("q") or 0.0) - 0.5 * float(product.get("risk") or 0.0)
        support = min(1.0, math.log1p(float(product.get("n") or 0.0)) / 6.0)
        score = overlap + 0.15 * prior + 0.05 * support
        scored.append((score, fam))
    scored.sort(key=lambda item: item[0], reverse=True)
    result: list[dict[str, Any]] = []
    for score, fam in scored[:top_k]:
        copied = {k: v for k, v in fam.items() if not k.startswith("_")}
        copied["retrieval_score"] = round(float(score), 6)
        result.append(copied)
    return result


def build_experience_block(families: list[dict[str, Any]], *, max_chars: int = 2800) -> str:
    lines = ["Retrieved ExperienceEvo rollout-derived process guidance:"]
    for i, fam in enumerate(families, 1):
        product = fam.get("product_experience") or {}
        lines.append(
            f"[{i}] target={fam.get('target_product_state')} q={product.get('q'):.3f} "
            f"risk={product.get('risk'):.3f} n={product.get('n')}"
        )
        if product.get("experience"):
            lines.append(f"    product: {product['experience']}")
        for policy in (fam.get("tool_policies") or [])[:2]:
            lines.append(
                f"    tool={policy.get('tool')} q={float(policy.get('q') or 0.0):.3f} "
                f"risk={float(policy.get('risk') or 0.0):.3f}; {policy.get('experience') or ''}"
            )
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 20].rstrip() + "\n...[truncated]"
    return text


def build_prompt(
    task: PublicTask,
    tool_catalog: list[dict[str, Any]],
    families: list[dict[str, Any]],
    *,
    tool_name_style: str = "slug",
    online: bool = False,
) -> list[dict[str, str]]:
    experience_text = f"\n\n{build_experience_block(families)}" if families else ""
    action_note = (
        "Use the function names exactly as exposed by the tool schemas. The Terrabox slug is shown only for audit."
        if tool_name_style == "function_name"
        else "Use the canonical Terrabox tool slugs exactly as shown."
    )
    final_note = (
        "You may call tools over multiple turns. When enough evidence is available, produce a concise final answer."
        if online
        else "This RL stage trains process actions only: do not include a final answer field."
    )
    if online:
        tool_block = "Available tools are provided by the function schemas for this turn."
    else:
        tool_block = "Available tools:\n" + compact_tool_catalog(tool_catalog, tool_name_style=tool_name_style)
    system = (
        "You are a Terrabox geospatial tool-use agent. Use tools to solve the task with observable evidence. "
        f"{final_note}\n\n"
        "Use only tools from the public tool schemas. Do not invent tool names or arguments. "
        f"{action_note}\n\n"
        "Required action-only format:\n"
        '{"thought":"short reason","actions":[{"tool":"tool_name","arguments":{"required_arg":"value"}}]}\n\n'
        "Invalid tool outputs: empty actions, more than one action in one turn, markdown, commentary, "
        "or an invented tool name.\n\n"
        f"{tool_block}"
        f"{experience_text}"
    )
    user = f"Question: {task.question}"
    if task.images:
        user += "\n\nImage files:\n" + "\n".join(f"- {path}" for path in task.images)
    if task.data_files:
        shown = task.data_files[:50]
        user += "\n\nData files:\n" + "\n".join(f"- {path}" for path in shown)
        if len(task.data_files) > 50:
            user += f"\n... and {len(task.data_files) - 50} more files"
    if task.data_dir:
        user += f"\n\nData directory: {task.data_dir}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def reward_policy_payload(families: list[dict[str, Any]], tool_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    schemas = {}
    for tool in tool_catalog:
        slug = str(tool.get("slug") or "")
        if slug:
            schemas[slug] = tool.get("parameters") or {}
    recommended: list[dict[str, Any]] = []
    for fam in families:
        for policy in fam.get("tool_policies") or []:
            tool = str(policy.get("tool") or "")
            if not tool:
                continue
            recommended.append(
                {
                    "tool": tool,
                    "q": float(policy.get("q") or 0.0),
                    "risk": float(policy.get("risk") or 0.0),
                    "n": int(policy.get("n") or 0),
                    "required_input_roles": policy.get("required_input_roles") or [],
                    "output_contract": policy.get("output_contract") or [],
                }
            )
    return {
        "policy_source": "experience_evo_v4_clean_strict_rollout_only" if families else "pure_schema_baseline",
        "allowed_tools": sorted(schemas),
        "tool_schemas": schemas,
        "recommended_tools": recommended[:12],
        "strict_nolabel": True,
    }


def build_verl_rows(
    tasks: Iterable[PublicTask],
    *,
    tool_catalog: list[dict[str, Any]],
    families: list[dict[str, Any]],
    top_k: int = 5,
    prompt_top_k: int | None = None,
    reward_top_k: int | None = None,
    data_source: str = "oea_experience_evo_rl_strict",
    online: bool = False,
    artifact_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prompt_k = top_k if prompt_top_k is None else prompt_top_k
    reward_k = top_k if reward_top_k is None else reward_top_k
    for sample_index, task in enumerate(tasks):
        prompt_families = retrieve_families(task.question, families, top_k=prompt_k) if prompt_k > 0 else []
        reward_families = retrieve_families(task.question, families, top_k=reward_k) if reward_k > 0 else []
        tool_name_style = "function_name" if online else "slug"
        tools_kwargs: dict[str, Any] = {}
        tool_selection: list[str] = []
        if online:
            for tool in tool_catalog:
                name = _tool_call_name(tool, tool_name_style="function_name")
                if not name:
                    continue
                tool_selection.append(name)
                tools_kwargs[name] = {
                    "create_kwargs": {
                        "sample_index": sample_index,
                        "images": list(task.images),
                        "data_files": list(task.data_files),
                        "data_dir": task.data_dir,
                        "artifact_root": str(artifact_root or "tmp/experience_evo_rl/online_artifacts"),
                    }
                }
        rows.append(
            {
                "data_source": data_source,
                "agent_name": "terrabox_tool_agent" if online else "single_turn_agent",
                "prompt": build_prompt(
                    task,
                    tool_catalog,
                    prompt_families,
                    tool_name_style=tool_name_style,
                    online=online,
                ),
                "reward_model": {"ground_truth": json.dumps(reward_policy_payload(reward_families, tool_catalog), ensure_ascii=False)},
                "extra_info": {
                    "sample_index": sample_index,
                    "source": "openearth",
                    "images": list(task.images),
                    "data_files": list(task.data_files),
                    "data_dir": task.data_dir,
                    "strict_nolabel": True,
                    "online_tools": online,
                    "prompt_top_k": prompt_k,
                    "reward_top_k": reward_k,
                    "tool_selection": tool_selection,
                    "need_tools_kwargs": online,
                    "tools_kwargs": tools_kwargs,
                },
            }
        )
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_parquet(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    import pandas as pd

    data = list(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(out)
    return len(data)


def scan_forbidden_payload(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []

    def walk(obj: Any, path: tuple[str, ...], row_idx: int) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_l = str(key).lower()
                child_path = path + (key_l,)
                if key_l in FORBIDDEN_KEYS:
                    if child_path not in ALLOWED_FORBIDDEN_KEY_PATHS and not any(
                        parent in child_path for parent in ALLOWED_FORBIDDEN_KEY_PARENTS
                    ):
                        hits.append({"row": row_idx, "key": key_l, "path": ".".join(child_path)})
                walk(value, child_path, row_idx)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, path + (str(i),), row_idx)

    for idx, row in enumerate(rows):
        walk(row, (), idx)
    return {"ok": not hits, "hits": hits[:50], "num_hits": len(hits)}


def prepare_dataset(
    *,
    task_file: str | Path,
    tool_catalog_file: str | Path,
    store_dir: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    train_ratio: float = 0.95,
    top_k: int = 5,
    prompt_top_k: int | None = None,
    reward_top_k: int | None = None,
    parquet: bool = True,
    online: bool = False,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    tasks = load_public_tasks(task_file, limit=limit)
    catalog = load_tool_catalog(tool_catalog_file)
    families = load_experience_families(store_dir)
    rows = build_verl_rows(
        tasks,
        tool_catalog=catalog,
        families=families,
        top_k=top_k,
        prompt_top_k=prompt_top_k,
        reward_top_k=reward_top_k,
        data_source="oea_experience_evo_rl_online" if online else "oea_experience_evo_rl_strict",
        online=online,
        artifact_root=artifact_root,
    )
    leak_scan = scan_forbidden_payload(rows)
    if not leak_scan["ok"]:
        raise RuntimeError(f"Forbidden strict no-label fields found in veRL rows: {leak_scan}")
    split = max(1, int(len(rows) * train_ratio)) if rows else 0
    if split >= len(rows) and len(rows) > 1:
        split = len(rows) - 1
    train_rows = rows[:split]
    val_rows = rows[split:] or rows[:1]
    out = Path(output_dir)
    write_jsonl(train_rows, out / "train.jsonl")
    write_jsonl(val_rows, out / "val.jsonl")
    if parquet:
        write_parquet(train_rows, out / "train.parquet")
        write_parquet(val_rows, out / "val.parquet")
    stats = {
        "task_file": str(task_file),
        "store_dir": str(store_dir),
        "tool_catalog_file": str(tool_catalog_file),
        "num_tasks": len(tasks),
        "num_families": len(families),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "top_k": top_k,
        "prompt_top_k": top_k if prompt_top_k is None else prompt_top_k,
        "reward_top_k": top_k if reward_top_k is None else reward_top_k,
        "parquet": parquet,
        "strict_nolabel": True,
        "online_tools": online,
        "artifact_root": str(artifact_root) if artifact_root else None,
        "leak_scan": leak_scan,
        "forbidden_fields": sorted(FORBIDDEN_KEYS),
    }
    (out / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict ExperienceEvo-guided veRL data")
    parser.add_argument("--task-file", default=str(DEFAULT_TRAIN_DATA))
    parser.add_argument("--tool-catalog", default=str(DEFAULT_TOOL_CATALOG))
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prompt-top-k", type=int, help="ExperienceEvo families included in the model prompt; defaults to --top-k")
    parser.add_argument("--reward-top-k", type=int, help="ExperienceEvo families used by the reward policy; defaults to --top-k")
    parser.add_argument("--online", action="store_true", help="Build veRL multi-turn online tool-agent rows")
    parser.add_argument("--artifact-root", help="Root directory for per-episode online tool artifacts")
    parser.add_argument("--no-parquet", action="store_true")
    args = parser.parse_args()
    stats = prepare_dataset(
        task_file=args.task_file,
        tool_catalog_file=args.tool_catalog,
        store_dir=args.store_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        train_ratio=args.train_ratio,
        top_k=args.top_k,
        prompt_top_k=args.prompt_top_k,
        reward_top_k=args.reward_top_k,
        parquet=not args.no_parquet,
        online=args.online,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
