"""Prompt Proposal:让 LLM **自己阅读日志 + 原始静态提示词**,自行诊断哪些问题是
提示词没写好导致的,并改写提示词。

两个关键立场:
1. **开放式自发现**:不给 LLM 任何预设的失败标签/统计;问题由它从原始日志里看出来,
   这样能发现我们没预想到的协议缺陷,也更符合"由 LLM 自行设计"的目标。
2. **领域无关**:这段"指挥 LLM 优化提示词"的元提示词不含任何遥感/地理/具体工具的字眼,
   换成 code agent、金融 agent、对话式 agent 等任意 agent 场景都能直接用。

反膨胀:用"保持克制、最小改动、禁任务专属内容"来约束(不写死行数/字数),改写后做
体量增幅检查仅作软提示。
"""
from __future__ import annotations

import re
from typing import Optional

from .schemas import PromptEdit, PromptProposal, PatchProposal
from .candidate_selection import choose_static_candidate, score_static_candidates
from .protocol_patch import build_patch_prompt, parse_patch_proposal, ProtocolPatchError


_OPTIMIZER_SYSTEM = (
    "You are an expert at designing concise system prompts for LLM agents. "
    "You distinguish prompt-fixable behavior from model/data/evaluator limits, "
    "and you make only low-regression-risk, general prompt edits."
)

_META_PROMPT_BODY = """You will see the current static system prompt used by an LLM agent, plus a small set of execution logs from tasks where it was used. The logs may include reasoning, replies, plans, searches, tool/API/code calls, observations, or final answers.

Your job is to make a conservative improvement to the static prompt. Fix only problems that are clearly likely to improve through prompt wording, and avoid changes that could break tasks that already work.

First decide whether each problem is suitable for a prompt fix:
- Suitable: unstable output format, weak instruction following, missing checks, ignoring provided context or interface constraints, inventing information, or rewriting values without a reason.
- Suitable: returning no action/answer when the prompt, task context, and provided interface clearly contain enough information to produce the required output.
- Not suitable: missing information, need for new external knowledge or retrieval ability, evaluator requirements that cannot be inferred from the logs, model capability limits, or environment/tool failures.

Only edit the prompt for problems that are suitable and low risk. If there is not enough evidence, keep the new prompt very close to the original.
Prefer narrow additions over rewriting existing sentences. If the original prompt already contains an output format, example, placeholder, or context insertion anchor, keep it exactly unless the logs prove that specific text is harmful. If the prompt ends with a context insertion anchor, place any new static rule before that anchor, not after it.
Treat the current prompt as a working contract, not as a rough draft. Most good edits should add one general guardrail or refine one phrase while preserving the rest of the contract.

================ Current Static Prompt ================
{base_prompt}

================ Execution Logs ================
{trace_text}
{metric_block}
================ Rules ================
1. Add or modify only general behavior rules: rules that should still help in a new task or domain.
2. Do not add task-specific content: no specific tool names, API names, entity names, fixed solutions, fixed action sequences, or new examples copied from the logs.
3. Preserve the original output contract and useful instructions. Make the smallest change that can plausibly fix the issue. Prefer adding one short rule to replacing a whole paragraph.
4. Avoid strong words such as "always", "must", or "strictly" unless the logs show that such a constraint is low risk.
5. For fields, values, or answers taken from context, prefer general rules such as: do not invent information, do not rewrite provided values without a reason, and preserve the information given in the task.
6. Preserve the original prompt structure unless it is clearly harmful: section names, input/output placeholders, format examples, dynamic-context anchors, and multiline layout.
7. Do not weaken an existing exact output format rule. If format failures exist, add a clarifying rule next to the original contract rather than replacing the contract.
8. When logs show both format failures and value-matching failures, avoid fixes that improve format by increasing value rewriting. Add a value-preservation condition if needed.
9. Do not add a new permission to skip output, return nothing, or emit an empty output unless the original prompt already allowed it.
10. Do not remove an existing requirement unless you explicitly identify that exact requirement as harmful and low-risk to remove.
11. If logs show avoidable empty/no-op outputs, prefer a general rule like: when the task context and provided interface make the required output clear, produce that output rather than a blank or unrelated response.
12. When adding such an action/answer rule, pair it with a context-fidelity rule: use only information supported by the task context, and preserve provided values unless the prompt already allows normalization.
13. If no low-risk prompt fix is supported by the logs, return a prompt that is nearly unchanged.
14. Do not append static rules after a trailing dynamic-context anchor. Keep the anchor as the final line if it was final in the original prompt.
15. A good general action-completeness rule does not name a domain. Prefer concrete-but-general wording like: when the user asks for an operation, lookup, computation, status, update, or other result and a matching interface is available, use that interface to produce the required structured output.
"""

_META_PROMPT = _META_PROMPT_BODY + """
================ Output Format ================
Return strict JSON only:
{{
  "diagnosis": [
    {{"issue": "one-sentence problem",
      "evidence": "what in the logs supports this",
      "prompt_gap": "why the current prompt failed to prevent it",
      "fixability": "prompt_fixable|not_prompt_fixable_or_high_risk|uncertain",
      "regression_risk": "low|medium|high"}}
  ],
  "edits": [
    {{"op": "add|modify|remove",
      "target": "the original text being changed, or where to add the new text",
      "new_text": "new text, empty for remove",
      "addresses": "which diagnosis item this edit addresses",
      "generality_note": "why this rule is general and should transfer to other tasks"}}
  ],
  "revised_prompt": "the full revised static prompt",
  "rationale": "one-sentence summary of what changed and why"
}}
"""

_META_PROMPT_BODY_V2 = """You will see the current static system prompt used by an LLM agent, plus a compact set of execution logs and optional aggregate metrics from tasks where it was used. The logs may include reasoning, replies, plans, searches, tool/API/code calls, observations, or final answers.

Your job is to produce the next version of the static prompt. The new prompt should be general enough to transfer to other tasks that use the same agent interface. Optimize the static instruction only; do not encode task-specific answers, tool names, dataset names, private examples, or fixed workflows.

Work in this order:
1. Read the current prompt as a contract. Identify the exact behavior it already requires and preserve that contract unless there is direct evidence that one phrase causes regressions.
2. Separate failures into prompt-fixable and not-prompt-fixable causes.
   - Prompt-fixable: ignoring explicit instructions, failing to use an available interface when the task clearly requires it, unsupported invention, value rewriting, weak final-answer discipline, avoidable repetition, or stopping before producing a required result.
   - Not prompt-fixable: missing data, unavailable tools, external service failures, context-window limits, evaluator/runtime bugs, or tasks that require capability not exposed in the interface.
3. Look for repeated behavior patterns across failures and compare them with successful traces. Prefer one rule that addresses a repeated pattern over several rules that only explain one case.
4. Make the smallest prompt edit that can plausibly improve the repeated prompt-fixable behavior while keeping successful behavior intact.

================ Current Static Prompt ================
{base_prompt}

================ Execution Logs ================
{trace_text}
{metric_block}
================ Editing Rules ================
1. Keep the prompt domain-neutral. Do not mention specific tool names, API names, suite names, entity names, task IDs, file names, or examples from the logs.
2. Preserve existing output formats, placeholders, dynamic-context anchors, section headers, and examples unless the logs directly prove that exact text is harmful.
3. Prefer a narrow addition or a short phrase refinement over rewriting the whole prompt.
4. Do not add broad absolute rules unless they are already implied by the prompt and supported by multiple logs.
5. If the agent repeated the same failed call or action, add a general rule to inspect the latest observation and change strategy rather than repeating identical arguments.
6. If the agent invented or rewrote values, add a general context-fidelity rule: use only supported information and copy provided values exactly unless the interface or prompt requires another representation.
7. If the agent stopped early or returned a blank/unrelated answer, add a general completion rule: when the task and available interface make the required result clear, produce the result instead of stopping.
8. If the issue is environment/tool/runtime/context limits, do not pretend a prompt can fix it; leave the prompt close to the original.
9. Keep the revised prompt concise enough to be usable as a static system prompt. Do not turn it into an experiment report.
10. If no low-risk, repeated, prompt-fixable pattern is visible, return a prompt that is nearly unchanged and explain that restraint in the rationale.
"""

_META_PROMPT_V2 = _META_PROMPT_BODY_V2 + """
================ Output Format ================
Return strict JSON only:
{{
  "diagnosis": [
    {{"issue": "one-sentence behavior pattern",
      "evidence": "which logs or metrics support it",
      "prompt_gap": "why the current prompt did not prevent it",
      "fixability": "prompt_fixable|not_prompt_fixable_or_high_risk|uncertain",
      "regression_risk": "low|medium|high"}}
  ],
  "edits": [
    {{"op": "add|modify|remove",
      "target": "the original text being changed, or where to add the new text",
      "new_text": "new text, empty for remove",
      "addresses": "which diagnosis item this edit addresses",
      "generality_note": "why this rule transfers beyond the sampled tasks"}}
  ],
  "revised_prompt": "the full revised static prompt",
  "rationale": "one-sentence summary of the conservative change"
}}
"""

_MULTI_CANDIDATE_PROMPT = _META_PROMPT_BODY + """

Generate {n} different conservative candidates. Each candidate must follow the same rules above.

Use distinct edit styles:
- Candidate 1: context fidelity. Keep the original text nearly intact and add one general rule about preserving provided values/facts and not adding unsupported fields.
- Candidate 2: interface-use completeness. Add one general rule that says: when the user asks for an operation, lookup, computation, status, update, or other result and a matching interface is available, use that interface to produce the required structured output rather than leaving it blank or unrelated.
- Candidate 3 and later: combined conservative rule. Combine interface-use completeness with context fidelity in one short sentence.

For every candidate:
- Preserve all section headers, placeholders, format examples, and trailing dynamic-context anchors exactly unless a specific one is proven harmful.
- If the original prompt's final line is a dynamic-context anchor, it must remain the final line. Insert new static rules before it.
- Do not shorten the prompt by deleting existing constraints merely to make it cleaner.
- Do not replace a concrete format rule with a looser paraphrase.
- The revised prompt must be the full static prompt, not a patch or summary.

Do not use JSON for this multi-candidate response. Use this exact plain-text
format so prompt text can contain quotes safely:

<candidate id="1">
<diagnosis>
one or two short bullets
</diagnosis>
<rationale>
one sentence
</rationale>
<revised_prompt>
full revised static prompt
</revised_prompt>
</candidate>
"""

_MULTI_CANDIDATE_PROMPT_V2 = _META_PROMPT_BODY_V2 + """

Generate {n} different conservative candidates. Each candidate must follow the same rules above and must be a full replacement for the static prompt.

Use distinct edit styles:
- Candidate 1: minimal repeated-pattern fix. Keep the original almost intact and add or refine one general rule for the most repeated prompt-fixable failure.
- Candidate 2: context-fidelity and completion. Add one compact rule that preserves provided values and asks the agent to produce the required result when the interface clearly supports it.
- Candidate 3 and later: repetition-control. Add one compact rule that prevents repeating identical failed actions and requires changing strategy after an observation.

For every candidate:
- Preserve all section headers, placeholders, format examples, and trailing dynamic-context anchors exactly unless a specific one is proven harmful.
- If the original prompt's final line is a dynamic-context anchor, it must remain the final line. Insert new static rules before it.
- Do not delete existing constraints merely to make the prompt cleaner.
- Do not replace a concrete format rule with a looser paraphrase.
- The revised prompt must be the full static prompt, not a patch or summary.

Do not use JSON for this multi-candidate response. Use this exact plain-text
format so prompt text can contain quotes safely:

<candidate id="1">
<diagnosis>
one or two short bullets
</diagnosis>
<rationale>
one sentence
</rationale>
<revised_prompt>
full revised static prompt
</revised_prompt>
</candidate>
"""


class PromptOptimizer:
    """用 EvolutionLLMClient,基于(原始提示词 + 原始日志)产出一次改写提案。"""

    def __init__(self, llm_client=None, max_growth_ratio: float = 1.5, meta_prompt_version: str = "v1"):
        if llm_client is None:
            from ..shared.llm_client import EvolutionLLMClient
            llm_client = EvolutionLLMClient()
        self.llm = llm_client
        self.max_growth_ratio = max_growth_ratio  # 仅软提示:改写后体量超过原文的此倍数则标记
        self.meta_prompt_version = meta_prompt_version

    def _check_restraint(self, base: str, revised: str) -> tuple[bool, str]:
        ratio = len(revised) / max(1, len(base))
        ok = ratio <= self.max_growth_ratio
        note = f"体量为原文的 {ratio:.2f}x (软上限 {self.max_growth_ratio}x)"
        return ok, note

    def propose_protocol_patches(
        self,
        base_prompt: str,
        trace_text: str,
        max_tokens: int = 3500,
        metric_block: str = "",
        comparison: str = "",
    ) -> Optional[PatchProposal]:
        """以显式 patch 模式生成并编译类型化行为协议。

        该入口不改变 propose()/propose_candidates() 的历史完整 prompt 语义。
        解析或契约检查失败时返回 None，由调用方明确记录失败，不静默退回整段改写。
        """
        prompt = build_patch_prompt(
            base_prompt,
            trace_text,
            metric_block=metric_block,
            comparison=comparison,
        )
        data = self.llm.call_json(prompt, system=_OPTIMIZER_SYSTEM, max_tokens=max_tokens)
        try:
            return parse_patch_proposal(data, base_prompt)
        except ProtocolPatchError:
            return None

    def propose(self, base_prompt: str, trace_text: str,
                max_tokens: int = 3500, metric_block: str = "") -> Optional[PromptProposal]:
        """生成一次改写提案;LLM 失败或解析失败返回 None。

        metric_block: **可选**的绝对指标块(默认空 = 行为与改前完全一致,纯开放式自发现)。
        想喂指标时传入 `_fmt_metric_summary(agg, agg, specs)` 之类的渲染文本即可。
        """
        block = (f"\n================ Current Metrics (for context; metric directions are described) ================\n"
                 f"{metric_block}\n") if metric_block.strip() else ""
        template = _META_PROMPT_V2 if self.meta_prompt_version == "v2" else _META_PROMPT
        prompt = template.format(base_prompt=base_prompt, trace_text=trace_text,
                                 metric_block=block)
        data = self.llm.call_json(prompt, system=_OPTIMIZER_SYSTEM, max_tokens=max_tokens)
        return self._proposal_from_data(base_prompt, data)

    def propose_candidates(
        self,
        base_prompt: str,
        trace_text: str,
        n: int = 3,
        max_tokens: int = 3500,
        metric_block: str = "",
    ) -> list[PromptProposal]:
        """Generate multiple first-stage candidates.

        This is still lightweight: it only makes additional LLM calls and does
        not run the target agent. Use `propose_best` to select one candidate by
        static text checks before launching any expensive rollout.
        """

        candidates: list[PromptProposal] = []
        if n > 1:
            block = (f"\n================ Current Metrics (for context; metric directions are described) ================\n"
                     f"{metric_block}\n") if metric_block.strip() else ""
            template = _MULTI_CANDIDATE_PROMPT_V2 if self.meta_prompt_version == "v2" else _MULTI_CANDIDATE_PROMPT
            prompt = template.format(
                base_prompt=base_prompt,
                trace_text=trace_text,
                metric_block=block,
                n=n,
            )
            raw = self.llm.call(prompt, system=_OPTIMIZER_SYSTEM, max_tokens=max_tokens * max(1, n))
            candidates.extend(self._parse_text_candidates(base_prompt, raw))
            if candidates:
                return candidates[:n]
        for _ in range(max(1, n)):
            proposal = self.propose(
                base_prompt,
                trace_text,
                max_tokens=max_tokens,
                metric_block=metric_block,
            )
            if proposal is not None:
                candidates.append(proposal)
        return candidates

    def _parse_text_candidates(self, base_prompt: str, raw: str) -> list[PromptProposal]:
        out: list[PromptProposal] = []
        blocks = re.findall(r"<candidate\b[^>]*>(.*?)</candidate>", raw or "", flags=re.DOTALL | re.IGNORECASE)
        for block in blocks:
            prompt_match = re.search(r"<revised_prompt>(.*?)</revised_prompt>", block, flags=re.DOTALL | re.IGNORECASE)
            if not prompt_match:
                continue
            revised = prompt_match.group(1).strip()
            if not revised:
                continue
            rationale_match = re.search(r"<rationale>(.*?)</rationale>", block, flags=re.DOTALL | re.IGNORECASE)
            diagnosis_match = re.search(r"<diagnosis>(.*?)</diagnosis>", block, flags=re.DOTALL | re.IGNORECASE)
            restrained, note = self._check_restraint(base_prompt, revised)
            out.append(PromptProposal(
                base_prompt=base_prompt,
                revised_prompt=revised,
                edits=[],
                rationale=(rationale_match.group(1).strip() if rationale_match else ""),
                diagnosis=[{"text": diagnosis_match.group(1).strip()}] if diagnosis_match else [],
                restrained=restrained,
                size_note=note,
            ))
        return out

    def _proposal_from_data(self, base_prompt: str, data) -> Optional[PromptProposal]:
        if not isinstance(data, dict) or "revised_prompt" not in data:
            return None
        revised = str(data["revised_prompt"]).strip()
        if not revised:
            return None
        edits = []
        for e in data.get("edits", []):
            if not isinstance(e, dict):
                continue
            edits.append(PromptEdit(
                op=str(e.get("op", "modify")),
                target=str(e.get("target", "")),
                new_text=str(e.get("new_text", "")),
                addresses=str(e.get("addresses", "")),
                generality_note=str(e.get("generality_note", "")),
            ))
        restrained, note = self._check_restraint(base_prompt, revised)
        return PromptProposal(
            base_prompt=base_prompt,
            revised_prompt=revised,
            edits=edits,
            rationale=str(data.get("rationale", "")),
            diagnosis=list(data.get("diagnosis", []) or []),
            restrained=restrained,
            size_note=note,
        )

    def propose_best(
        self,
        base_prompt: str,
        trace_text: str,
        n: int = 3,
        max_tokens: int = 3500,
        metric_block: str = "",
    ) -> tuple[Optional[PromptProposal], list[dict]]:
        """Generate candidates and choose one using text-only checks."""

        proposals = self.propose_candidates(
            base_prompt,
            trace_text,
            n=n,
            max_tokens=max_tokens,
            metric_block=metric_block,
        )
        if not proposals:
            return None, []
        raw_candidates = [
            {
                "revised_prompt": proposal.revised_prompt,
                "proposal": proposal,
            }
            for proposal in proposals
        ]
        all_scores = [score.to_dict() for score in score_static_candidates(base_prompt, raw_candidates, stage="stage1")]
        try:
            best, scores = choose_static_candidate(
                base_prompt,
                raw_candidates,
                stage="stage1",
                min_score=4.0,
            )
        except ValueError as exc:
            return None, [{"error": str(exc), "scores": all_scores}]
        return best["proposal"], [score.to_dict() for score in scores]
