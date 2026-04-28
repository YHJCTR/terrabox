from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    TOOL_CALL = "tool_call"
    KNOWLEDGE_QA = "knowledge_qa"
    CASUAL_CHAT = "casual_chat"
    SESSION_MGMT = "session_mgmt"


# Session management: deterministic keyword check, no LLM needed
_SESSION_KEYWORDS = {"清空", "重置", "删除历史", "新对话", "clear", "reset", "new chat"}

# Keyword scoring for fast-path (avoids LLM call on clear-cut cases)
_TOOL_KEYWORDS = {
    "分析", "检测", "计算", "提取", "分割", "识别", "分类", "生成",
    "遥感", "影像", "栅格", "矢量", "NDVI", "NDWI", "EVI",
    "洪水", "地震", "火灾", "滑坡", "灾害",
    "analyze", "detect", "calculate", "extract", "segment", "classify",
    "remote sensing", "satellite", "raster", "flood", "earthquake",
}
_KNOWLEDGE_KEYWORDS = {
    "什么是", "怎么", "如何", "为什么", "解释", "介绍", "说明", "定义",
    "区别", "比较", "原理", "概念",
    "what is", "how to", "why", "explain", "describe", "difference",
}

# Keyword score threshold: above this we trust keywords and skip the LLM call
_KEYWORD_CONFIDENCE_THRESHOLD = 2


def _keyword_classify(query: str, has_tools: bool, has_kbs: bool) -> IntentType:
    q = query.lower()
    tool_score = sum(1 for kw in _TOOL_KEYWORDS if kw in q)
    know_score = sum(1 for kw in _KNOWLEDGE_KEYWORDS if kw in q)

    if tool_score > 0 and has_tools:
        return IntentType.TOOL_CALL if tool_score >= know_score else (
            IntentType.KNOWLEDGE_QA if has_kbs else IntentType.TOOL_CALL
        )
    if know_score > 0:
        return IntentType.KNOWLEDGE_QA if has_kbs else IntentType.CASUAL_CHAT
    return IntentType.CASUAL_CHAT


def _llm_classify(query: str, has_tools: bool, has_kbs: bool) -> IntentType:
    from .llm import call_llm_json
    tool_hint = "工具可用（图像分析/遥感/检测等）" if has_tools else "无工具"
    kb_hint = "知识库可用" if has_kbs else "无知识库"
    data = call_llm_json(
        system=(
            f"将用户问题分类为以下意图之一（{tool_hint}，{kb_hint}）：\n"
            "- tool_call: 需要调用工具执行分析、检测、计算等\n"
            "- knowledge_qa: 需要从知识库检索文档回答问题\n"
            "- casual_chat: 闲聊、问候，不需要工具或知识库\n"
            "- session_mgmt: 清空历史、重置对话等会话管理操作\n"
            '仅输出 JSON，格式：{"intent": "...", "confidence": 0.0}'
        ),
        user=query,
    )
    intent_str = data.get("intent", "")
    return IntentType(intent_str)


class IntentClassifier:
    def classify(self, query: str, has_tools: bool = True, has_kbs: bool = False) -> IntentType:
        q = query.lower().strip()

        # 1. Session management: always keyword-only (fast, deterministic)
        if any(kw in q for kw in _SESSION_KEYWORDS):
            return IntentType.SESSION_MGMT

        # 2. High-confidence keyword match: skip LLM call
        tool_score = sum(1 for kw in _TOOL_KEYWORDS if kw in q)
        know_score = sum(1 for kw in _KNOWLEDGE_KEYWORDS if kw in q)
        if max(tool_score, know_score) >= _KEYWORD_CONFIDENCE_THRESHOLD:
            return _keyword_classify(query, has_tools, has_kbs)

        # 3. Ambiguous: use LLM classification with keyword fallback
        try:
            return _llm_classify(query, has_tools, has_kbs)
        except Exception as exc:
            logger.warning("LLM intent classification failed, using keywords: %s", exc)
            return _keyword_classify(query, has_tools, has_kbs)
