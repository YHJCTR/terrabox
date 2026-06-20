"""promptevo v2 —— 领域无关接口层。

核心引擎(sampler / optimizer / loop)**只依赖本文件**,不 import 任何 terrabox 代码。
换到别的 agent 系统时,用户只需实现下面几个 Protocol(见 adapters_terrabox.py 的默认实现):
    PromptStore        原始/版本静态提示词的读写
    TrajectorySource   轨迹输入(统一成 Trace)
    MetricProvider     实验结果/指标输入
    RolloutRunner      (可选)用某提示词跑 dev 验证
    LLMClient          (可选)LLM 调用(默认接 EvolutionLLMClient)
    Renderer           轨迹 -> 文本(compressor 落地前的占位;领域无关默认实现见 sampler)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, runtime_checkable


# ---- 统一后的轨迹结构(领域无关) -------------------------------------------------

@dataclass
class Step:
    role: str                          # "user" | "assistant" | "tool"
    text: str = ""                     # 思考 / 问题 / 工具返回(原文)
    tool: Optional[str] = None         # assistant 步调用的工具名
    args: Optional[dict] = None
    errored: bool = False              # 该步(工具返回)是否报错


@dataclass
class Trace:
    task_id: str
    query: str
    steps: list[Step]
    success: bool
    final_answer: str = ""
    raw: dict = field(default_factory=dict)   # 保留原始,便于领域特定字段


@dataclass
class TaskMetric:
    task_id: str
    success: bool
    tool_f1: float = 0.0
    category_f1: dict[str, float] = field(default_factory=dict)   # {"perception":.., "gis":..}
    failure_flags: dict[str, bool] = field(default_factory=dict)  # {"repeat_call":True,..}
    extra: dict = field(default_factory=dict)


@dataclass
class MetricSpec:
    """使用者为每个聚合指标提供的简介,让 LLM 自主判断'改善还是退步'。

    框架**不假设方向**(去掉了'负=退步'的硬编码);好坏由 description 描述,LLM 据此判断。
    direction 可选,仅作辅助提示;留空则完全由 description 让 LLM 推断。
    """
    name: str
    description: str                    # 该指标含义 + 方向,如 "撞回合上限比例,越低越好"
    direction: str = "unknown"         # "higher_better" | "lower_better" | "neutral" | "unknown"


# ---- 用户换领域需实现的 Protocol ------------------------------------------------

@runtime_checkable
class PromptStore(Protocol):
    def load(self, version: str) -> str: ...
    def save(self, version: str, prompt: str, meta: dict) -> str: ...


@runtime_checkable
class TrajectorySource(Protocol):
    def traces(self, experiment: str) -> Iterable[Trace]: ...


@runtime_checkable
class MetricProvider(Protocol):
    def per_task(self, experiment: str) -> dict[str, TaskMetric]: ...
    def aggregate(self, experiment: str,
                  task_ids: Optional[list[str]] = None) -> dict[str, Any]: ...
    def metric_specs(self) -> list["MetricSpec"]:
        """返回 aggregate() 里各指标的简介(含义/方向),供 LLM 自主判断改善/退步。
        使用者决定放哪些指标、怎么描述方向。"""
        ...


@runtime_checkable
class RolloutRunner(Protocol):
    def run(self, prompt: str, task_ids: list[str], experiment: str) -> str: ...
    # 用给定提示词在指定任务上跑出一个新 experiment,返回其名/路径


@runtime_checkable
class LLMClient(Protocol):
    def call_json(self, prompt: str, system: Optional[str] = None,
                  max_tokens: int = 3000) -> Any: ...


@runtime_checkable
class Renderer(Protocol):
    def render(self, trace: Trace, budget_chars: int = 1600) -> str: ...
