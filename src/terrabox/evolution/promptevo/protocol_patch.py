"""类型化协议补丁的解析、校验和编译。

这个模块不绑定任何数据集或工具名称。它把优化器的结构化输出编译成一份
静态 system prompt，并在编译前检查冲突、占位符和动态上下文锚点。
旧的整段 prompt 改写流程不会经过这里，只有显式选择 patch 模式时启用。
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from .schemas import PatchProposal, PatchValidationReport, ProtocolPatch


PATCH_KINDS = (
    "tool_selection",
    "argument_validation",
    "error_recovery",
    "termination_and_repetition",
)
PATCH_RISKS = ("low", "medium", "high")
_PLACEHOLDER_RE = re.compile(r"(\{[A-Za-z_][A-Za-z0-9_]*\}|<[^>\n]{1,80}>)")
_DYNAMIC_ANCHOR_KEYWORDS = (
    "tool",
    "api",
    "schema",
    "context",
    "history",
    "input",
    "output",
    "example",
    "constraint",
    "instruction",
)
_TASK_SPECIFIC_RE = re.compile(
    r"(?:task[_ -]?id\s*[:=]|(?:/home|/data|/tmp)/|(?:oea|agentdojo|tau2|toolbench)[_-](?:test|task)[_-]?\d+)",
    re.IGNORECASE,
)


class ProtocolPatchError(ValueError):
    """结构化补丁不满足编译契约。"""


def _nonempty_lines(text: str) -> list[str]:
    return [line for line in text.rstrip().splitlines() if line.strip()]


def _placeholders(text: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(text))


def _trailing_anchor(text: str) -> str:
    lines = _nonempty_lines(text)
    if not lines:
        return ""
    candidate = lines[-1].strip()
    if not candidate.endswith(":"):
        return ""
    label = candidate[:-1].strip().lower()
    return candidate if any(keyword in label for keyword in _DYNAMIC_ANCHOR_KEYWORDS) else ""


def _as_patch(raw: Any, index: int) -> ProtocolPatch:
    if not isinstance(raw, dict):
        raise ProtocolPatchError(f"patches[{index}] 必须是对象")
    required = ("patch_id", "kind", "trigger", "rule")
    missing = [key for key in required if not str(raw.get(key, "")).strip()]
    if missing:
        raise ProtocolPatchError(f"patches[{index}] 缺少字段: {', '.join(missing)}")
    kind = str(raw["kind"]).strip()
    if kind not in PATCH_KINDS:
        raise ProtocolPatchError(f"patches[{index}] kind 不支持: {kind}")
    risk = str(raw.get("risk", "low")).strip().lower()
    if risk not in PATCH_RISKS:
        raise ProtocolPatchError(f"patches[{index}] risk 不支持: {risk}")
    try:
        priority = int(raw.get("priority", 50))
    except (TypeError, ValueError) as exc:
        raise ProtocolPatchError(f"patches[{index}] priority 必须是整数") from exc
    if not 0 <= priority <= 100:
        raise ProtocolPatchError(f"patches[{index}] priority 必须在 0..100")
    evidence = raw.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        raise ProtocolPatchError(f"patches[{index}] evidence 必须是字符串列表")
    return ProtocolPatch(
        patch_id=str(raw["patch_id"]).strip(),
        kind=kind,
        trigger=str(raw["trigger"]).strip(),
        rule=str(raw["rule"]).strip(),
        scope=str(raw.get("scope", "global")).strip() or "global",
        priority=priority,
        evidence=[str(item).strip() for item in evidence if str(item).strip()],
        risk=risk,
    )


def validate_patches(base_prompt: str, patches: Iterable[ProtocolPatch]) -> PatchValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    duplicate_rules: list[str] = []
    seen_ids: set[str] = set()
    seen_rules: dict[tuple[str, str, str], ProtocolPatch] = {}
    patch_list = list(patches)
    if not base_prompt.strip():
        errors.append("base prompt 不能为空")
    for index, patch in enumerate(patch_list):
        if not isinstance(patch, ProtocolPatch):
            errors.append(f"patches[{index}] 不是 ProtocolPatch")
            continue
        if patch.patch_id in seen_ids:
            errors.append(f"重复 patch_id: {patch.patch_id}")
        seen_ids.add(patch.patch_id)
        if len(patch.rule) > 1200 or len(patch.trigger) > 600:
            errors.append(f"补丁过长: {patch.patch_id}")
        if _TASK_SPECIFIC_RE.search(patch.rule) or _TASK_SPECIFIC_RE.search(patch.trigger):
            errors.append(f"补丁包含任务/路径专属信息: {patch.patch_id}")
        key = (patch.kind, patch.scope.lower(), patch.trigger.lower())
        previous = seen_rules.get(key)
        if previous is not None:
            if previous.rule.lower() == patch.rule.lower():
                duplicate_rules.append(patch.patch_id)
                warnings.append(f"重复规则已忽略: {patch.patch_id}")
            else:
                errors.append(
                    f"同一触发条件存在冲突规则: {previous.patch_id} 与 {patch.patch_id}"
                )
        else:
            seen_rules[key] = patch
        if patch.risk == "high":
            warnings.append(f"高风险补丁不会默认编译: {patch.patch_id}")
    placeholders_ok = True
    # This is checked again after compilation; keeping the field here makes
    # the report useful to callers that only need preflight validation.
    if not _placeholders(base_prompt):
        placeholders_ok = True
    return PatchValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        preserved_placeholders=placeholders_ok,
        duplicate_rules=duplicate_rules,
    )


def compile_patches(base_prompt: str, patches: Iterable[ProtocolPatch]) -> str:
    """将有效补丁编译到 base prompt，保持动态锚点为最后一行。"""
    patch_list = list(patches)
    report = validate_patches(base_prompt, patch_list)
    if not report.valid:
        raise ProtocolPatchError("; ".join(report.errors))
    duplicate_ids = set(report.duplicate_rules)
    active = [patch for patch in patch_list if patch.patch_id not in duplicate_ids and patch.risk != "high"]
    if not active:
        return base_prompt
    existing_rules = {line.strip().lstrip("- ").strip().lower() for line in base_prompt.splitlines()}
    grouped: dict[str, list[ProtocolPatch]] = defaultdict(list)
    for patch in active:
        if patch.rule.lower() not in existing_rules:
            grouped[patch.kind].append(patch)
    additions: list[str] = []
    for kind in PATCH_KINDS:
        rules = sorted(grouped.get(kind, []), key=lambda item: (-item.priority, item.patch_id))
        if rules:
            additions.append(f"[{kind}]")
            additions.extend(f"- {patch.rule}" for patch in rules)
    if not additions:
        return base_prompt
    block = "Protocol rules added by the prompt compiler:\n" + "\n".join(additions)
    anchor = _trailing_anchor(base_prompt)
    if anchor:
        lines = base_prompt.rstrip().splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        body = "\n".join(lines[:-1]).rstrip()
        return f"{body}\n\n{block}\n\n{anchor}\n"
    return f"{base_prompt.rstrip()}\n\n{block}\n"


def compile_protocol_prompt(base_prompt: str, patches: Iterable[ProtocolPatch]) -> tuple[str, PatchValidationReport]:
    """编译并返回包含占位符/动态锚点检查结果的 prompt。"""
    patch_list = list(patches)
    preflight = validate_patches(base_prompt, patch_list)
    if not preflight.valid:
        raise ProtocolPatchError("; ".join(preflight.errors))
    compiled = compile_patches(base_prompt, patch_list)
    placeholders_ok = _placeholders(base_prompt) <= _placeholders(compiled)
    anchor = _trailing_anchor(base_prompt)
    compiled_lines = _nonempty_lines(compiled)
    anchor_ok = not anchor or (compiled_lines and compiled_lines[-1].strip() == anchor)
    errors = list(preflight.errors)
    if not placeholders_ok:
        errors.append("编译后丢失原始占位符")
    if not anchor_ok:
        errors.append("编译后没有保留末尾动态上下文锚点")
    report = PatchValidationReport(
        valid=not errors,
        errors=errors,
        warnings=list(preflight.warnings),
        preserved_placeholders=placeholders_ok,
        preserved_trailing_anchor=anchor_ok,
        duplicate_rules=list(preflight.duplicate_rules),
    )
    if not report.valid:
        raise ProtocolPatchError("; ".join(report.errors))
    return compiled, report


def parse_patch_proposal(data: Any, base_prompt: str) -> PatchProposal:
    """严格解析优化器 JSON，并立即执行确定性编译检查。"""
    if not isinstance(data, dict):
        raise ProtocolPatchError("协议补丁提案必须是 JSON 对象")
    raw_patches = data.get("patches")
    if not isinstance(raw_patches, list):
        raise ProtocolPatchError("协议补丁提案必须包含 patches 列表")
    patches = [_as_patch(raw, index) for index, raw in enumerate(raw_patches)]
    compiled, _ = compile_protocol_prompt(base_prompt, patches)
    return PatchProposal(
        base_prompt=base_prompt,
        patches=patches,
        rationale=str(data.get("rationale", "")).strip(),
        diagnosis=list(data.get("diagnosis", []) or []),
        compiled_prompt=compiled,
    )


def build_patch_prompt(base_prompt: str, trace_text: str, *, metric_block: str = "", comparison: str = "") -> str:
    """生成 Stage1/Stage2 共用的领域无关补丁提案提示词。"""
    return f"""You are evolving a static system prompt as a typed behavior protocol.
Do not rewrite the whole prompt. Propose only small, general, prompt-fixable protocol patches.
Use one or more of these kinds only: tool_selection, argument_validation, error_recovery, termination_and_repetition.
Do not include task IDs, file paths, entity names, benchmark names, fixed answers, or task-specific workflows.
Do not propose a patch for missing data, unavailable services, model limits, or evaluator/runtime failures.
Each rule must be useful across unseen tasks in the same agent interface.

================ BASE PROMPT ================
{base_prompt}
================ EVIDENCE ================
{comparison}
{trace_text}
{metric_block}
================ OUTPUT ================
Return strict JSON only:
{{
  "diagnosis": [{{"issue": "general behavior pattern", "evidence": "supporting evidence", "fixability": "prompt_fixable|not_prompt_fixable|uncertain"}}],
  "patches": [{{"patch_id": "stable_general_id", "kind": "one allowed kind", "trigger": "when this rule applies", "rule": "one concise general rule for the agent", "scope": "global", "priority": 50, "evidence": ["short evidence"], "risk": "low|medium|high"}}],
  "rationale": "why these patches are conservative and general"
}}
"""
