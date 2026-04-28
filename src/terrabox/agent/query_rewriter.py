from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """You are a query rewriter. Given a conversation history and the latest user message, rewrite the user's message into a standalone, self-contained query that captures the full intent.

Rules:
1. If the message refers to previous context (e.g., "it", "that", "the result"), resolve the reference
2. If the message is already self-contained, return it as-is
3. Keep the rewritten query concise and natural
4. Output ONLY the rewritten query, nothing else

Conversation history:
{history}

Latest user message: {query}

Rewritten query:"""


class QueryRewriter:
    def __init__(self, max_history_turns: int = 3):
        self.max_history_turns = max_history_turns

    def rewrite(self, query: str, history: list, llm=None) -> str:
        if not history:
            return query
        recent = history[-(self.max_history_turns * 2):]
        history_text = ""
        for msg in recent:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content[:200]}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content[:200]}\n"
        if not history_text.strip():
            return query
        has_reference = any(
            kw in query for kw in ["它", "这个", "那个", "上面", "之前", "刚才", "it", "this", "that", "the above", "previous"]
        )
        if not has_reference and len(query.split()) > 3:
            return query
        if llm is None:
            return query
        try:
            result = llm.invoke([
                {"role": "user", "content": _REWRITE_PROMPT.format(history=history_text, query=query)}
            ])
            rewritten = result.content.strip()
            return rewritten if rewritten else query
        except Exception as e:
            logger.warning("Query rewrite failed: %s", e)
            return query
