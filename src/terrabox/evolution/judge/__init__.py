"""答案正确性判分(LLM-as-judge)。后端解耦:本地 vLLM 或外部 DeepSeek 自由选择。"""
from .answer_judge import AnswerJudge, EVAL_PROMPT, extract_numbers, numeric_match

__all__ = ["AnswerJudge", "EVAL_PROMPT", "extract_numbers", "numeric_match"]
