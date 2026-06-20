"""promptevo — 静态提示词(行为协议)自进化模块。

定位:从 rollout 日志中挖掘**通用的、协议层面的**提示词缺陷,让 LLM 自动改写
静态 system prompt,再用 dev 集指标做回归验证后才接受。

与 memrl / expel 等"经验库"方法的关键区别:
- 经验库:保存**单个任务的 workflow / 经验**,随任务增多而膨胀,且偏记忆。
- promptevo:只产出**适用于一类失败的通用协议规则**,prompt 体量受预算约束,
  不存任务级 workflow,不存示例(避免静态提示词爆炸)。

流水线(对标 Self-Harness: 自发现 → Proposal → Validation):
    sample_traces(trajectories) -> str                     # 采样原始日志(领域无关渲染)
    PromptOptimizer.propose(base_prompt, trace_text)       # LLM 自诊断 + 改写
    validate(before_metrics, after_metrics) -> 接受/拒绝    # 回归门(用确定性指标)

注:优化路径是**开放式自发现**——只把原始提示词 + 采样日志交给 LLM,问题由它自己看出来;
不把预设的失败标签喂给优化器。`mine_weaknesses` 仅作确定性指标视角,供人核对与 validate 用。
"""
from .schemas import Weakness, PromptEdit, PromptProposal, ValidationResult
from .trace_sampler import sample_traces
from .weakness_miner import mine_weaknesses
from .optimizer import PromptOptimizer
from .validator import validate_proposal
from .prompt_injector import PromptevoAugmenter

__all__ = [
    "Weakness",
    "PromptEdit",
    "PromptProposal",
    "ValidationResult",
    "sample_traces",
    "mine_weaknesses",
    "PromptOptimizer",
    "validate_proposal",
    "PromptevoAugmenter",
]
