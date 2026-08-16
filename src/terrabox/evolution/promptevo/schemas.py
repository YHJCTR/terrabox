"""promptevo 数据结构。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Weakness:
    """一类(而非一条)被聚合统计出来的提示词级失败模式。"""
    pattern_id: str                 # 如 "empty_planning_turn"
    description: str                # 人话描述这个失败模式
    rate: float                     # 出现该模式的轨迹占比 0~1
    count: int                      # 绝对条数
    severity: str = "mid"           # high / mid / low(由 rate + 与失败的相关性推断)
    success_gap: float = 0.0        # 命中该模式 vs 未命中 的 success_rate 差(越负越该修)
    affected_task_types: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)  # 少量**匿名片段**(非整条轨迹)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptEdit:
    """对静态提示词的一处结构化修改。"""
    op: str                         # "add" | "modify" | "remove"
    target: str                     # 被改/删的原文片段(add 时为空或所属小节名)
    new_text: str                   # add/modify 的新文本
    addresses: str                  # 对应 LLM 自己归纳的哪个 issue(自由文本)
    generality_note: str            # 为什么这是通用规则,换领域也成立

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptProposal:
    """一次提示词改写提案。"""
    base_prompt: str
    revised_prompt: str
    edits: list[PromptEdit]
    rationale: str
    diagnosis: list[dict[str, Any]] = field(default_factory=list)  # LLM 自发现的问题清单
    restrained: bool = True         # 改动是否克制(未大幅膨胀)
    size_note: str = ""             # 体量变化说明(仅供人看,非硬约束)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProtocolPatch:
    """一条可编译的、领域无关的静态行为协议补丁。"""
    patch_id: str
    kind: str                         # tool_selection / argument_validation / error_recovery / termination_and_repetition
    trigger: str                      # 何时适用
    rule: str                         # 写给 agent 的通用规则
    scope: str = "global"
    priority: int = 50
    evidence: list[str] = field(default_factory=list)
    risk: str = "low"                # low / medium / high

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatchProposal:
    """结构化补丁提案及其编译后的静态 prompt。"""
    base_prompt: str
    patches: list[ProtocolPatch]
    rationale: str
    diagnosis: list[dict[str, Any]] = field(default_factory=list)
    compiled_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatchValidationReport:
    """协议补丁编译前后的确定性检查结果。"""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preserved_placeholders: bool = True
    preserved_trailing_anchor: bool = True
    duplicate_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """回归验证结果。"""
    accepted: bool
    before: dict[str, float]
    after: dict[str, float]
    delta: dict[str, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ===== v2 对比式迭代(见 DESIGN_v2_contrastive.md) =====

@dataclass
class PairedCase:
    """同一 task 在两版提示词下的配对证据。"""
    task_id: str
    query: str
    a_render: str                       # A(旧版)轨迹渲染
    b_render: str                       # B(新版)轨迹渲染
    metric_delta: dict[str, float]      # B - A 的分维度指标差
    regressed_dims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Attribution:
    """对'某条改动在某维度有益/有害'的归因。"""
    edit_ref: str                       # 指向哪条改动(文字描述)
    effect: str                         # "help" | "hurt" | "mixed"
    dims: list[str]                     # 影响的维度(如 ["gis","logic"])
    evidence: str                       # 支撑的现象/任务

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UpdateResult:
    """一次对比式 update 的结果。"""
    accepted: bool
    new_version: str
    revised_prompt: str
    diagnosis: list[Attribution] = field(default_factory=list)
    candidates_tried: int = 0
    dev_before: dict[str, float] = field(default_factory=dict)
    dev_after: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    protocol_patches: list[ProtocolPatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
