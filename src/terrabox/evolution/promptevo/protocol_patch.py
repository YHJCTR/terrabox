"""类型化协议补丁的解析、校验和编译。

这个模块不绑定任何数据集或工具名称。它把优化器的结构化输出编译成一份
静态 system prompt，并在编译前检查冲突、占位符和动态上下文锚点。
旧的整段 prompt 改写流程不会经过这里，只有显式选择 patch 模式时启用。
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable

from .schemas import PatchProposal, PatchValidationReport, ProtocolPatch


PATCH_KINDS = (
    "tool_selection",
    "query_abstraction",
    "argument_validation",
    "error_recovery",
    "termination_and_repetition",
    "answer_contract",
    "tool_output_security",
    "candidate_metric_guard",
    "state_transition",
)
PATCH_RISKS = ("low", "medium", "high")
PROTOCOL_MODES = ("legacy", "conditional")
PROTOCOL_PATCH_RENDERER_VERSIONS = {
    "legacy": "legacy-v1",
    "conditional": "conditional-v1",
}
QUERY_ABSTRACTION_SCOPE = "discovery/query fields only"
_COMPILED_RULE_MARKER = "Protocol rules added by the prompt compiler:"
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


def protocol_renderer_version(protocol_mode: str = "legacy") -> str:
    try:
        return PROTOCOL_PATCH_RENDERER_VERSIONS[protocol_mode]
    except KeyError as exc:
        raise ProtocolPatchError(f"protocol_mode 不支持: {protocol_mode}") from exc


def _validate_protocol_mode(protocol_mode: str) -> None:
    if protocol_mode not in PROTOCOL_MODES:
        raise ProtocolPatchError(f"protocol_mode 不支持: {protocol_mode}")


def _render_patch(patch: ProtocolPatch, protocol_mode: str = "legacy") -> str:
    """Render only typed patch fields; evidence never enters the agent prompt."""
    _validate_protocol_mode(protocol_mode)
    if protocol_mode == "conditional":
        return f"- IF {patch.trigger} THEN {patch.rule} [Scope: {patch.scope}]"
    return f"- {patch.rule}"


def _normalize_patch(patch: ProtocolPatch, protocol_mode: str) -> ProtocolPatch:
    """Keep typed query edits from becoming an unintended global instruction."""
    if (
        protocol_mode == "conditional"
        and patch.kind == "query_abstraction"
        and patch.scope.strip().lower() != QUERY_ABSTRACTION_SCOPE
    ):
        return replace(patch, scope=QUERY_ABSTRACTION_SCOPE)
    return patch


def _normalize_rule_line(value: str) -> str:
    return value.strip().lstrip("- ").strip().lower()


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
    evidence_items = [str(item).strip() for item in evidence if str(item).strip()]
    forbidden_evidence = {"short evidence", "supporting evidence", "evidence", "n/a", "none"}
    if not evidence_items or any(item.strip().lower() in forbidden_evidence for item in evidence_items):
        raise ProtocolPatchError(f"patches[{index}] evidence 必须引用真实轨迹现象或任务证据")
    return ProtocolPatch(
        patch_id=str(raw["patch_id"]).strip(),
        kind=kind,
        trigger=str(raw["trigger"]).strip(),
        rule=str(raw["rule"]).strip(),
        scope=str(raw.get("scope", "global")).strip() or "global",
        priority=priority,
        evidence=evidence_items,
        risk=risk,
    )


def validate_patches(
    base_prompt: str,
    patches: Iterable[ProtocolPatch],
    protocol_mode: str = "legacy",
) -> PatchValidationReport:
    _validate_protocol_mode(protocol_mode)
    errors: list[str] = []
    warnings: list[str] = []
    duplicate_rules: list[str] = []
    seen_ids: set[str] = set()
    seen_rules: dict[tuple[str, str, str], ProtocolPatch] = {}
    patch_list = [
        _normalize_patch(patch, protocol_mode) if isinstance(patch, ProtocolPatch) else patch
        for patch in patches
    ]
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
        if any("\n" in value or "\r" in value for value in (patch.trigger, patch.rule, patch.scope)):
            errors.append(f"补丁字段必须是单行文本: {patch.patch_id}")
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


def compile_patches(
    base_prompt: str,
    patches: Iterable[ProtocolPatch],
    protocol_mode: str = "legacy",
) -> str:
    """将有效补丁编译到 base prompt，保持动态锚点为最后一行。

    Stage2 的 base prompt 可能已经包含 Stage1 的编译规则。新规则合并
    到已有 compiler block，避免每一轮都重复添加同名 block。
    """
    _validate_protocol_mode(protocol_mode)
    patch_list = [
        _normalize_patch(patch, protocol_mode) if isinstance(patch, ProtocolPatch) else patch
        for patch in patches
    ]
    report = validate_patches(base_prompt, patch_list, protocol_mode)
    if not report.valid:
        raise ProtocolPatchError("; ".join(report.errors))
    duplicate_ids = set(report.duplicate_rules)
    active = [patch for patch in patch_list if patch.patch_id not in duplicate_ids and patch.risk != "high"]
    if not active:
        return base_prompt
    existing_rules = {_normalize_rule_line(line) for line in base_prompt.splitlines()}
    grouped: dict[str, list[ProtocolPatch]] = defaultdict(list)
    for patch in active:
        rendered = _render_patch(patch, protocol_mode)
        if _normalize_rule_line(rendered) not in existing_rules:
            grouped[patch.kind].append(patch)
    additions_by_kind: dict[str, list[ProtocolPatch]] = {}
    for kind in PATCH_KINDS:
        rules = sorted(grouped.get(kind, []), key=lambda item: (-item.priority, item.patch_id))
        if rules:
            additions_by_kind[kind] = rules
    if not additions_by_kind:
        return base_prompt

    anchor = _trailing_anchor(base_prompt)
    lines = base_prompt.rstrip().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if anchor and lines and lines[-1].strip() == anchor:
        lines.pop()
    body = "\n".join(lines).rstrip()

    # A second-stage patch extends the existing compiler block. This keeps the
    # compiled prompt inspectable and avoids duplicate section headers.
    marker_index = body.rfind(_COMPILED_RULE_MARKER)
    if marker_index >= 0:
        prefix = body[:marker_index].rstrip()
        block_lines = body[marker_index:].splitlines()
        for kind in PATCH_KINDS:
            rules = additions_by_kind.get(kind)
            if not rules:
                continue
            header = f"[{kind}]"
            try:
                start = block_lines.index(header)
            except ValueError:
                block_lines.extend([header, *[_render_patch(patch, protocol_mode) for patch in rules]])
                continue
            end = start + 1
            while end < len(block_lines) and not (
                block_lines[end].startswith("[") and block_lines[end].endswith("]")
            ):
                end += 1
            block_lines[end:end] = [_render_patch(patch, protocol_mode) for patch in rules]
        merged = "\n".join(block_lines).rstrip()
        result = f"{prefix}\n\n{merged}" if prefix else merged
        if anchor:
            result += f"\n\n{anchor}"
        return result + "\n"

    additions: list[str] = [_COMPILED_RULE_MARKER]
    for kind in PATCH_KINDS:
        rules = additions_by_kind.get(kind)
        if not rules:
            continue
        additions.append(f"[{kind}]")
        additions.extend(_render_patch(patch, protocol_mode) for patch in rules)
    block = "\n".join(additions)
    result = f"{body}\n\n{block}" if body else block
    if anchor:
        result += f"\n\n{anchor}"
    return result + "\n"


def compile_protocol_prompt(
    base_prompt: str,
    patches: Iterable[ProtocolPatch],
    protocol_mode: str = "legacy",
) -> tuple[str, PatchValidationReport]:
    """编译并返回包含占位符/动态锚点检查结果的 prompt。"""
    _validate_protocol_mode(protocol_mode)
    patch_list = list(patches)
    preflight = validate_patches(base_prompt, patch_list, protocol_mode)
    if not preflight.valid:
        raise ProtocolPatchError("; ".join(preflight.errors))
    compiled = compile_patches(base_prompt, patch_list, protocol_mode)
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


def parse_patch_proposal(
    data: Any,
    base_prompt: str,
    protocol_mode: str = "legacy",
) -> PatchProposal:
    """严格解析优化器 JSON，并立即执行确定性编译检查。"""
    if not isinstance(data, dict):
        raise ProtocolPatchError("协议补丁提案必须是 JSON 对象")
    raw_patches = data.get("patches")
    if not isinstance(raw_patches, list):
        raise ProtocolPatchError("协议补丁提案必须包含 patches 列表")
    patches = [
        _normalize_patch(_as_patch(raw, index), protocol_mode)
        for index, raw in enumerate(raw_patches)
    ]
    compiled, _ = compile_protocol_prompt(base_prompt, patches, protocol_mode)
    return PatchProposal(
        base_prompt=base_prompt,
        patches=patches,
        rationale=str(data.get("rationale", "")).strip(),
        diagnosis=list(data.get("diagnosis", []) or []),
        compiled_prompt=compiled,
    )


def build_patch_prompt(
    base_prompt: str,
    trace_text: str,
    *,
    metric_block: str = "",
    comparison: str = "",
    protocol_mode: str = "legacy",
) -> str:
    """生成 Stage1/Stage2 共用的领域无关补丁提案提示词。"""
    _validate_protocol_mode(protocol_mode)
    mode_guidance = (
        "In conditional mode, trigger must name an observable state or condition, "
        "rule must state the action, and scope must state where the action applies. "
        "Use state_transition for cross-turn dependencies. The compiler will preserve "
        "these conditions, so do not hide them in evidence."
        if protocol_mode == "conditional" else
        "In legacy mode, keep trigger as typed metadata and keep each rule concise."
    )
    return f"""You are evolving a static system prompt as a typed behavior protocol.
Do not rewrite the whole prompt. Propose only small, general, prompt-fixable protocol patches.
Use one or more of these kinds only: {', '.join(PATCH_KINDS)}.
{mode_guidance}
If the evidence concerns a search or discovery query field, use query_abstraction. In conditional mode,
its scope must be exactly "discovery/query fields only": preserve the user's operation, target object,
and any entity or identifier that the API schema needs, while omitting only conversational filler or
other details unrelated to the query. Ground the concise query in the available descriptions. Do not
put a literal query, API name, example, or gold answer in the patch. Do not include task IDs, file paths,
benchmark names, fixed answers, or task-specific workflows. A patch may describe preserving
schema-significant entity values without naming a concrete entity from the evidence.
For state_transition, describe missing observable dependency state generically when a dependent action
is emitted too early; do not name a particular prerequisite tool or authentication workflow. For
argument_validation, it is valid to require exact schema-key spelling and schema-grounded values, but
never rewrite or discard a user-provided entity merely to make it shorter.
Do not propose a patch for missing data, unavailable services, model limits, or evaluator/runtime failures.
Each rule must be useful across unseen tasks in the same agent interface.
Each patch must cite concrete evidence from the supplied traces or metric comparison. Do not use placeholder evidence such as "short evidence" or "supporting evidence".

================ BASE PROMPT ================
{base_prompt}
================ EVIDENCE ================
{comparison}
{trace_text}
{metric_block}
================ OUTPUT ================
Return strict JSON only:
{{
  "diagnosis": [{{"issue": "general behavior pattern", "evidence": "specific trace or metric evidence", "fixability": "prompt_fixable|not_prompt_fixable|uncertain"}}],
  "patches": [{{"patch_id": "stable_general_id", "kind": "one allowed kind", "trigger": "when this rule applies", "rule": "one concise general rule for the agent", "scope": "discovery/query fields only for query_abstraction, otherwise global or a narrower boundary", "priority": 50, "evidence": ["specific trace or metric evidence"], "risk": "low|medium|high"}}],
  "rationale": "why these patches are conservative and general"
}}
"""
