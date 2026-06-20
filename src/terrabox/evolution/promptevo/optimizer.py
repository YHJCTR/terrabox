"""Prompt Proposal:让 LLM **自己阅读日志 + 原始静态提示词**,自行诊断哪些问题是
提示词没写好导致的,并改写提示词。

两个关键立场:
1. **开放式自发现**:不给 LLM 任何预设的失败标签/统计;问题由它从原始日志里看出来,
   这样能发现我们没预想到的协议缺陷,也更符合"由 LLM 自行设计"的目标。
2. **领域无关**:这段"指挥 LLM 优化提示词"的元提示词不含任何遥感/地理/具体工具的字眼,
   换成 code agent、金融 agent 等任意工具型 agent 都能直接用。

反膨胀:用"保持克制、最小改动、禁任务专属内容"来约束(不写死行数/字数),改写后做
体量增幅检查仅作软提示。
"""
from __future__ import annotations

from typing import Optional

from .schemas import PromptEdit, PromptProposal


_OPTIMIZER_SYSTEM = (
    "You are an expert at designing system prompts (behavioral protocols) for "
    "tool-using LLM agents. From execution logs you diagnose whether recurring "
    "failures are caused by an underspecified system prompt, and you improve the "
    "prompt with GENERAL, domain-agnostic rules."
)

_META_PROMPT = """你将看到一个工具型 LLM agent 当前使用的【静态系统提示词】,以及它在若干任务上的【执行日志】(含每一步的思考、工具调用与工具返回)。

你的任务分两步:
第一步 诊断:仔细阅读日志,**自己归纳**出 agent 反复出现的、**很可能是因为静态系统提示词没写好/没约束到位**而导致的行为问题(而不是模型能力不足或工具本身的故障)。不要依赖任何预设的问题清单,完全靠你从日志里观察。
第二步 改写:修改这份静态系统提示词,使其能从根本上、系统性地避免你诊断出的这些问题。

================ 当前静态系统提示词 ================
{base_prompt}

================ 执行日志(节选若干条,工具返回已截断) ================
{trace_text}
{metric_block}
================ 必须遵守的约束(非常重要) ================
1. 只增加/修改**通用的行为规则**——即对"一类情况"都成立、换一个全新任务甚至全新领域也照样有用的规则。
2. **禁止**写入与具体任务或具体领域绑定的内容:某个任务的解法、固定的工具调用顺序/workflow、任何示例或样例输入输出、任何具体实体名(人名/地名/文件名/领域术语)。判据:一条规则若只对你在日志里见到的那几个任务有用,就不要写。
3. **保持克制**:尽量在原提示词基础上做**最小改动**——能改写或合并已有规则,就不要新增;不要因为个别日志案例而堆砌冗长的特例。改完后的提示词应当仍然紧凑、清晰,不要显著变长。
4. 每一处改动都要说明它对应你在第一步诊断出的哪个问题。
5. 保持原提示词的语言、风格与结构。

================ 输出格式(严格 JSON,不要多余文字) ================
{{
  "diagnosis": [
    {{"issue": "你归纳的问题(一句话)",
      "evidence": "日志里支持该问题的现象",
      "prompt_gap": "原提示词为什么没能防住它"}}
  ],
  "edits": [
    {{"op": "add|modify|remove",
      "target": "被改/删的原文片段(add 时填所属位置)",
      "new_text": "新文本(remove 时为空)",
      "addresses": "对应上面哪个 issue",
      "generality_note": "为什么这条规则通用、换领域也成立"}}
  ],
  "revised_prompt": "改写后的完整静态系统提示词",
  "rationale": "一句话总结这次改了什么、针对哪些问题"
}}
"""


class PromptOptimizer:
    """用 EvolutionLLMClient,基于(原始提示词 + 原始日志)产出一次改写提案。"""

    def __init__(self, llm_client=None, max_growth_ratio: float = 1.5):
        if llm_client is None:
            from ..shared.llm_client import EvolutionLLMClient
            llm_client = EvolutionLLMClient()
        self.llm = llm_client
        self.max_growth_ratio = max_growth_ratio  # 仅软提示:改写后体量超过原文的此倍数则标记

    def _check_restraint(self, base: str, revised: str) -> tuple[bool, str]:
        ratio = len(revised) / max(1, len(base))
        ok = ratio <= self.max_growth_ratio
        note = f"体量为原文的 {ratio:.2f}x (软上限 {self.max_growth_ratio}x)"
        return ok, note

    def propose(self, base_prompt: str, trace_text: str,
                max_tokens: int = 3500, metric_block: str = "") -> Optional[PromptProposal]:
        """生成一次改写提案;LLM 失败或解析失败返回 None。

        metric_block: **可选**的绝对指标块(默认空 = 行为与改前完全一致,纯开放式自发现)。
        想喂指标时传入 `_fmt_metric_summary(agg, agg, specs)` 之类的渲染文本即可。
        """
        block = (f"\n================ 当前实验指标(绝对值,供参考哪个维度本就弱;方向见简介) ================\n"
                 f"{metric_block}\n") if metric_block.strip() else ""
        prompt = _META_PROMPT.format(base_prompt=base_prompt, trace_text=trace_text,
                                     metric_block=block)
        data = self.llm.call_json(prompt, system=_OPTIMIZER_SYSTEM, max_tokens=max_tokens)
        if not isinstance(data, dict) or "revised_prompt" not in data:
            return None

        revised = str(data["revised_prompt"]).strip()
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
