"""对比式优化器(领域无关):看两版提示词 + 两版配对轨迹 + 指标差,
做跨版本对比归因(credit assignment),产出下一版候选。

降方差:多批配对各诊断一次,聚合反复出现的归因;再 best-of-N 出候选。
LLM 通过 interfaces.LLMClient 注入,核心不绑定 terrabox。
"""
from __future__ import annotations

import re
from typing import Optional

from .interfaces import LLMClient
from .schemas import PairedCase, Attribution
from .candidate_selection import choose_static_candidate

_SYSTEM = (
    "You improve static prompts for LLM agents. Compare two prompt versions and "
    "their task traces, identify what got better and what got worse, then suggest "
    "small general edits that keep the gains and reduce the regressions."
)

# 默认优化目标(通用兜底,不预设要优化哪个具体指标);使用者可通过 objective 覆盖。
_DEFAULT_OBJECTIVE = (
    "Improve overall performance. Metrics marked higher_better should go up, "
    "metrics marked lower_better should go down, and no important metric should "
    "drop sharply. Prefer a small reliable improvement over trading one failure "
    "mode for another."
)

_DIAGNOSE = """Compare two versions of a static system prompt on the same tasks. Explain which prompt changes helped and which changes caused regressions.

================ New Prompt Version ================
{prompt_b}

================ Actual Changes from Old to New ================
{real_diff}

================ Metric Changes: Old -> New ================
{metric_summary}
Metric directions differ. Use each metric description to decide whether a change is good or bad.

================ Objective ================
{objective}

================ Paired Traces: Same Task under Old and New Prompt ================
{cases}

Task:
Explain both the main improvements and the main regressions. Do not focus only on failures.
For each important behavior change:
1. Describe what changed in the new traces compared with the old traces.
2. Quote the exact phrase or clause from the Actual Changes list that most likely caused it.
3. Connect the behavior change to the affected metric, and mark the effect as help, hurt, or mixed.

Use only the listed prompt changes as causes. If a metric change cannot be explained by those changes, put it in unexplained. Be careful with rules that are useful for action/answer completeness or format, but also make the model rewrite values, drop context, invent fields, add unsupported fields, or become too strict.
When a change has both benefits and regressions, separate the helpful subclause from the harmful subclause. Do not label the whole change as harmful if one part clearly improved action/answer completeness.
For value-related regressions, look for surface-form changes in the traces: paraphrasing, changing capitalization, adding/removing words, replacing placeholders, converting data structures, normalizing times/dates beyond the expected output contract, or choosing a synonym. These are usually fixed by exact-copy value rules, not by weakening action completeness.

Return strict JSON only:
{{"attributions":[{{"edit_id":"edit-1","behavior":"behavior change observed in traces",
  "offending_clause":"exact phrase or clause that caused the behavior",
  "effect":"help|hurt|mixed",
  "dims":["affected metric names"],"evidence":"supporting tasks or observations"}}],
  "unexplained":["metric changes not explained by listed prompt changes"]}}
"""

_PROPOSE = """Create the next static system prompt from the comparison below.

================ Objective ================
{objective}

================ Current Prompt ================
{prompt_b}

================ What Helped or Hurt ================
{attributions}

Rules:
1. Keep changes that helped. Revert or narrow changes that hurt.
2. If a rule has both useful and harmful parts, edit only the harmful phrase. Do not delete the useful part.
3. If the current prompt is already better overall than the previous one, inherit it by default. Do not weaken a useful strong rule into vague advice; add a narrow condition or exception instead.
4. For mixed rules, keep the format, structure, or action-selection benefit while preventing value rewriting, lost context, invented fields, or over-restrictive behavior.
5. If a rule only caused harm and has no clear benefit, it may be removed. If evidence is weak, stay close to the previous prompt.
6. Preserve the current prompt structure unless it is clearly harmful: section names, input/output placeholders, format examples, dynamic-context anchors, constraints, and multiline layout. Do not collapse a multi-line prompt into one paragraph.
7. If the current prompt ends with an anchor where dynamic context is appended, keep that anchor exactly.
8. Keep the prompt general and domain-neutral. Do not add specific tool names, domain terms, task names, fixed workflows, or examples from the logs.
9. Keep the edit small. Prefer adding one guardrail or changing one phrase. Do not rewrite the whole prompt.
10. Do not optimize only for regressed examples. The new prompt should keep the behavior that improved under the current prompt and patch only the behavior that got worse.
11. Do not add a new permission to skip output, return nothing, or emit an empty output unless the current prompt already allowed it.
12. The revised prompt must be the full static prompt, with all unchanged text copied through exactly.
13. Do not fix value or argument regressions by allowing the model to adjust, correct, infer, or rewrite provided values. Prefer narrower rules such as exact-copying context values, using only information supported by the task/context, and not adding unsupported fields.
14. If the current prompt improved action/answer completeness, preserve that requirement. Patch regressions by adding a support condition such as "use only supported information" or "do not add unsupported fields", not by weakening the requirement to act.
15. If value mismatches remain, add a concrete but general copy-fidelity guardrail: copy values exactly as they appear in the provided context unless the original prompt or interface explicitly requires a different representation. Do not paraphrase, change capitalization, add/remove words, replace placeholder-like values, or normalize values only for style.

Return strict JSON only. The changes list must quote the exact old phrase and exact new phrase:
{{"revised_prompt":"the full next static prompt","rationale":"one-sentence reason for the change",
  "changes":[{{"edit_id":"which attribution this addresses","offending_clause":"the phrase judged to cause the issue",
    "old_phrase":"exact text before the change","new_phrase":"exact text after the change","why":"why this phrase needed to change"}}]}}
"""

_PROPOSE_TEXT = """Create the next static system prompt from the comparison below.

================ Objective ================
{objective}

================ Current Prompt ================
{prompt_b}

================ What Helped or Hurt ================
{attributions}

Rules:
1. Keep changes that helped. Revert or narrow changes that hurt.
2. If a rule has both useful and harmful parts, edit only the harmful phrase. Do not delete the useful part.
3. If the current prompt is already better overall than the previous one, inherit it by default. Do not weaken a useful strong rule into vague advice; add a narrow condition or exception instead.
4. For mixed rules, keep the format, structure, or action-selection benefit while preventing value rewriting, lost context, invented fields, or over-restrictive behavior.
5. If a rule only caused harm and has no clear benefit, it may be removed. If evidence is weak, stay close to the previous prompt.
6. Preserve the current prompt structure unless it is clearly harmful: section names, input/output placeholders, format examples, dynamic-context anchors, constraints, and multiline layout. Do not collapse a multi-line prompt into one paragraph.
7. If the current prompt ends with an anchor where dynamic context is appended, keep that anchor exactly.
8. Keep the prompt general and domain-neutral. Do not add specific tool names, domain terms, task names, fixed workflows, or examples from the logs.
9. Keep the edit small. Prefer adding one guardrail or changing one phrase. Do not rewrite the whole prompt.
10. Do not optimize only for regressed examples. The new prompt should keep the behavior that improved under the current prompt and patch only the behavior that got worse.
11. Do not add a new permission to skip output, return nothing, or emit an empty output unless the current prompt already allowed it.
12. The revised prompt must be the full static prompt, with all unchanged text copied through exactly.
13. Do not fix value or argument regressions by allowing the model to adjust, correct, infer, or rewrite provided values. Prefer narrower rules such as exact-copying context values, using only information supported by the task/context, and not adding unsupported fields.
14. If the current prompt improved action/answer completeness, preserve that requirement. Patch regressions by adding a support condition such as "use only supported information" or "do not add unsupported fields", not by weakening the requirement to act.
15. If value mismatches remain, add a concrete but general copy-fidelity guardrail: copy values exactly as they appear in the provided context unless the original prompt or interface explicitly requires a different representation. Do not paraphrase, change capitalization, add/remove words, replace placeholder-like values, or normalize values only for style.

Do not use JSON. Use this exact plain-text format so prompt text can contain
quotes safely:

<candidate>
<rationale>
one sentence
</rationale>
<changes>
- old: exact old phrase
  new: exact new phrase
  why: why this phrase needed to change
</changes>
<revised_prompt>
the full next static prompt
</revised_prompt>
</candidate>
"""


def build_emphasized_prompt(old_prompt: str, new_prompt: str,
                            changes: Optional[list] = None,
                            header: str = "## Important changes in this prompt version") -> str:
    """在 new_prompt 末尾追加"需特别遵守"小节,提升新规则在 rollout 时的显著性(近因效应)。
    **领域无关、确定性**:用代码 diff 取出 new 相对 old 改动的"新规则那侧"(地面真值,不靠 LLM 自述);
    `changes` 里 LLM 的 why 只作注解,且**只能附在代码验证过的真实改动上**(锚定,编不出清单外的)。
    无改动则原样返回。供 rollout 端用 toggle 决定加不加。"""
    import difflib
    a = old_prompt.strip().splitlines()
    b = new_prompt.strip().splitlines()
    sm = difflib.SequenceMatcher(None, [l.strip() for l in a], [l.strip() for l in b])
    new_lines = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            new_lines += [l.strip() for l in b[j1:j2] if l.strip()]
    if not new_lines:
        return new_prompt
    whys = {}
    for ch in (changes or []):
        if isinstance(ch, dict):
            npz = (ch.get("new_phrase") or "").strip()
            why = (ch.get("why") or "").strip()
            if npz and why:
                whys[npz[:30]] = why
    lines = [new_prompt.strip(), "", header]
    for nl in new_lines:
        note = ""
        for k, w in whys.items():
            if k and k in nl:
                note = f"\n  Why it matters: {w}"
                break
        clean = nl[2:].strip() if nl.startswith("- ") else nl
        lines.append(f"- {clean}{note}")
    return "\n".join(lines)


def _prompt_units(prompt: str) -> list[str]:
    """Split prompt text into stable, readable diff units.

    Long one-line prompts often contain several independent rules. Sentence-ish
    units make attribution more precise than treating the whole paragraph as one
    edit, while preserving section headers and short lines.
    """
    import re
    units: list[str] = []
    for line in prompt.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= 120 or line.endswith(":"):
            units.append(line)
            continue
        parts = re.split(r"(?<=[.!?。！？])\s+", line)
        units.extend(part.strip() for part in parts if part.strip())
    return units


def _compute_diff(prompt_a: str, prompt_b: str) -> str:
    """Compute the exact prompt diff rendered as edit blocks for the LLM."""
    import difflib
    a = _prompt_units(prompt_a)
    b = _prompt_units(prompt_b)
    sm = difflib.SequenceMatcher(None, [l.strip() for l in a], [l.strip() for l in b])
    out, eid = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        eid += 1
        old = " / ".join(x.strip() for x in a[i1:i2]) or "(none)"
        new = " / ".join(x.strip() for x in b[j1:j2]) or "(deleted)"
        out.append(f"[edit-{eid}] ({tag})\n  old: {old}\n  new: {new}")
    return "\n\n".join(out) if out else "(no changes)"


def _fmt_cases(cases: list[PairedCase], max_cases: int = 6) -> str:
    out = []
    for c in cases[:max_cases]:
        out.append(f"## task {c.task_id}  delta={c.metric_delta}  regressed_dims={c.regressed_dims}\n"
                   f"[A]\n{c.a_render}\n[B]\n{c.b_render}")
    return "\n\n".join(out)


def _balanced_cases(cases: list[PairedCase]) -> list[PairedCase]:
    """Interleave regressions and improvements before batching.

    sample_paired returns high-value cases, but regressions are listed first.
    Sequentially taking the first batches can hide verified gains, causing the
    optimizer to weaken useful rules. Keep this domain-neutral by using only
    per-case metric deltas and regressed_dims.
    """
    regressions: list[PairedCase] = []
    improvements: list[PairedCase] = []
    neutral: list[PairedCase] = []
    for case in cases:
        vals = [v for v in case.metric_delta.values() if isinstance(v, (int, float))]
        if case.regressed_dims or any(v < -0.05 for v in vals):
            regressions.append(case)
        elif any(v > 0.05 for v in vals):
            improvements.append(case)
        else:
            neutral.append(case)
    out: list[PairedCase] = []
    max_len = max(len(regressions), len(improvements), len(neutral), 0)
    for i in range(max_len):
        if i < len(regressions):
            out.append(regressions[i])
        if i < len(improvements):
            out.append(improvements[i])
        if i < len(neutral):
            out.append(neutral[i])
    return out


def _fmt_metric_summary(agg_a: dict, agg_b: dict, specs=None) -> str:
    """渲染'指标: A -> B (Δ) — 简介(方向)'。不假设正负好坏,方向交给简介+LLM 判断。"""
    desc = {s.name: (s.description, s.direction) for s in (specs or [])}
    keys = [k for k in agg_b if k not in ("n",)]
    rows = []
    for k in keys:
        a, b = agg_a.get(k), agg_b.get(k)
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            continue
        d, direction = desc.get(k, ("", "unknown"))
        note = f"  — {d}" if d else ""
        if direction and direction != "unknown":
            note += f" [{direction}]"
        rows.append(f"  {k}: {a:.3f} -> {b:.3f} (Δ {b-a:+.3f}){note}")
    return "\n".join(rows)


class ContrastiveOptimizer:
    def __init__(self, llm: Optional[LLMClient] = None):
        if llm is None:
            from ..shared.llm_client import EvolutionLLMClient
            llm = EvolutionLLMClient()
        self.llm = llm

    def diagnose(self, prompt_a: str, prompt_b: str, agg_a: dict, agg_b: dict,
                 cases: list[PairedCase], n_batches: int = 2,
                 batch_size: int = 4, metric_specs=None,
                 objective: str = _DEFAULT_OBJECTIVE,
                 max_tokens: int = 2500) -> list[Attribution]:
        """多批配对各诊断一次,聚合反复出现的归因(降方差)。
        metric_specs: 使用者提供的各指标简介(含方向),让 LLM 自判改善/退步。
        objective: 优化目标;默认通用兜底(不预设优化某个具体指标),使用者可覆盖。"""
        from collections import Counter
        tally: Counter = Counter()
        store: dict[str, Attribution] = {}
        metric_summary = _fmt_metric_summary(agg_a, agg_b, metric_specs)
        real_diff = _compute_diff(prompt_a, prompt_b)   # 代码算精确改动清单(hybrid:防臆造)
        ordered_cases = _balanced_cases(cases)
        batches = [
            ordered_cases[i:i + batch_size]
            for i in range(0, max(1, len(ordered_cases)), batch_size)
        ][:n_batches] or [ordered_cases]
        for batch in batches:
            data = self.llm.call_json(
                _DIAGNOSE.format(prompt_b=prompt_b, real_diff=real_diff,
                                 metric_summary=metric_summary, objective=objective,
                                 cases=_fmt_cases(batch)),
                system=_SYSTEM, max_tokens=max_tokens)
            if not isinstance(data, dict):
                continue
            id2text = {}
            for blk in real_diff.split("\n\n"):
                if blk.startswith("[edit-"):
                    eid = blk[1:blk.index("]")]
                    id2text[eid] = blk.replace("\n", " ")[:120]
            for a in data.get("attributions", []) or []:
                if not isinstance(a, dict):
                    continue
                eid = str(a.get("edit_id") or a.get("edit_ref", ""))
                ref = f"{eid}: {id2text.get(eid, '(清单外!)')}"
                clause = str(a.get("offending_clause", "")).strip()
                behavior = str(a.get("behavior", "")).strip()
                ev = (f"behavior: {behavior} | clause: {clause} | {a.get('evidence', '')}"
                      if clause or behavior else str(a.get("evidence", "")))
                key = (a.get("effect", ""), tuple(sorted(a.get("dims", []))))
                tally[key] += 1
                store.setdefault(str(key), Attribution(
                    edit_ref=ref, effect=str(a.get("effect", "")),
                    dims=list(a.get("dims", []) or []), evidence=ev))
        ranked = sorted(store.values(), key=lambda at: -tally[(at.effect, tuple(sorted(at.dims)))])
        return ranked

    @staticmethod
    def _grounded(cand: dict, prompt_b: str) -> bool:
        """M3 自动门:候选是否"说到做到"——
        (a) revised 必须真的不同于 B;
        (b) changes 里每条 new_phrase 必须真出现在 revised 里(否则=假改动,如之前 edit-2)。"""
        rev = (cand.get("revised_prompt") or "").strip()
        if not rev or rev == prompt_b.strip():
            return False
        for ch in cand.get("changes", []) or []:
            if not isinstance(ch, dict):
                continue
            npz = (ch.get("new_phrase") or "").strip()
            if npz and npz[:40] not in rev:      # 声称改成的短语没出现 → 假改动
                return False
        return True

    def propose_candidates(self, prompt_b: str, attributions: list[Attribution],
                           n: int = 3, max_tokens: int = 3500,
                           objective: str = _DEFAULT_OBJECTIVE) -> list[dict]:
        """best-of-N:生成 n 个下一版候选,并用自动门过滤"说了没做"的假候选。
        objective: 优化目标;默认通用兜底,使用者可覆盖。"""
        attr_txt = "\n".join(
            f"- [{a.effect}] {a.edit_ref}  (维度 {a.dims}; 证据 {a.evidence[:160]})"
            for a in attributions) or "(无显著归因)"
        cands = []
        for _ in range(n):
            data = None
            if hasattr(self.llm, "call"):
                raw = self.llm.call(
                    _PROPOSE_TEXT.format(prompt_b=prompt_b, attributions=attr_txt, objective=objective),
                    system=_SYSTEM,
                    max_tokens=max_tokens,
                )
                data = self._parse_text_candidate(raw)
            if not data:
                data = self.llm.call_json(
                    _PROPOSE.format(prompt_b=prompt_b, attributions=attr_txt, objective=objective),
                    system=_SYSTEM, max_tokens=max_tokens)
            if isinstance(data, dict) and data.get("revised_prompt"):
                cands.append(data)
        grounded = [c for c in cands if self._grounded(c, prompt_b)]
        return grounded if grounded else cands

    @staticmethod
    def _parse_text_candidate(raw: str) -> Optional[dict]:
        """Parse the quote-safe text candidate format."""

        if not raw:
            return None
        block_match = re.search(r"<candidate\b[^>]*>(.*?)</candidate>", raw, flags=re.DOTALL | re.IGNORECASE)
        block = block_match.group(1) if block_match else raw
        prompt_match = re.search(r"<revised_prompt>(.*?)</revised_prompt>", block, flags=re.DOTALL | re.IGNORECASE)
        if not prompt_match:
            return None
        revised = prompt_match.group(1).strip()
        if not revised:
            return None
        rationale_match = re.search(r"<rationale>(.*?)</rationale>", block, flags=re.DOTALL | re.IGNORECASE)
        changes_match = re.search(r"<changes>(.*?)</changes>", block, flags=re.DOTALL | re.IGNORECASE)
        changes_text = changes_match.group(1).strip() if changes_match else ""
        changes = [{"why": changes_text}] if changes_text else []
        return {
            "revised_prompt": revised,
            "rationale": rationale_match.group(1).strip() if rationale_match else "",
            "changes": changes,
        }

    def choose_candidate(self, prompt_b: str, candidates: list[dict]) -> tuple[dict, list[dict]]:
        """Choose one second-stage candidate with static checks."""

        best, scores = choose_static_candidate(prompt_b, candidates, stage="stage2", min_score=4.0)
        return best, [score.to_dict() for score in scores]
