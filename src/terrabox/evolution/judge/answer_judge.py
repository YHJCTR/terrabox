"""答案正确性判分(Answer Accuracy Score,对齐 OEA 作者口径)。

解耦:`AnswerJudge` 只负责「判分逻辑」,LLM 后端由**注入的 client** 决定
(本地 vLLM 的 EvolutionLLMClient,或外部 DeepSeek 的 RemoteChatClient,
两者同接口 `.call()`)。调用方通过 `agent.llm_provider.make_llm_client(provider)` 自由选择。

- 评分规则复用作者的 `EVAL_PROMPT`(数值 ±10% 容差、语义等价、"数值题答 unknown→0")。
- 结果按 `hash(question, ground_truth, predicted)` **磁盘缓存**,跨实验答案没变即复用,边际成本≈0。
- 可选 `numeric_match`:单数字答案用代码 ±10% 直接判,免 LLM(省钱)。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional

# 作者原版评分规则(逐字搬运,保证 answer_acc 与论文同口径)。
EVAL_PROMPT = """
You are an evaluation assistant responsible for measuring the Answer Accuracy Score (a value between 0 and 1) for a geospatial agent's prediction.

You are given:

- A Question
- A Ground Truth Answer
- A Predicted Answer produced by the agent

Your task is to evaluate how accurately the Predicted Answer matches the Ground Truth Answer, considering both numerical precision and semantic correctness.
When numerical values are present, allow a ±10% tolerance range for acceptable variation. For descriptive or categorical outputs, judge based on meaningful semantic equivalence.

Evaluation Guidelines

1. Numerical or Quantitative Comparisons
- A predicted numeric value is correct if it lies within ±10% of the corresponding value in the ground truth.
- For bounding boxes, consider the prediction correct if the IoU (Intersection over Union) >= 0.5.
- If multiple numeric values exist:
    - Compute element-wise difference.
    - All values within ±10% range -> Score = 0.95 - 1.00 (highly accurate)
    - Most values within ±10% range -> Score = 0.80 - 0.94 (minor acceptable deviations)
    - Some values within ±10% range -> Score = 0.30 - 0.79 (partially correct)
    - Most values within ±20% range -> Score = 0.10 - 0.29 (partially aligned)
    - All values incorrect or nonsensical -> Score = 0.00 (completely incorrect)
- Ignore extraneous numbers that do not directly answer the question.

2. Non-Numerical or Categorical Comparisons
    - Evaluate semantic equivalence:
    - Perfect semantic match -> Score ~ 1.0
    - Minor phrasing or synonym difference -> 0.8-0.9
    - Partially correct (missing non-critical details) -> 0.4-0.8
    - Vague or weakly related -> 0.1-0.4
    - Contradictory or incorrect -> 0.0

3. Handling Missing or Contradictory Content
- If the prediction omits all required answer elements -> Score = 0.0
- If the prediction introduces fabricated entities, relationships, or placeholders -> Score = 0.00
- If a numeric answer is required and the model responds with "no data", "unknown", or similar -> Score = 0.00

You must provide your evaluation strictly in the following format and nothing else:
{
    "Score": <float between 0.0 and 1.0>,
    "Justification": "<1-2 concise sentences explaining the score based on the comparison between GT and Pred>"
}
"""

_NUM_RE = re.compile(r"-?\d+\.?\d*")
_DEFAULT_CACHE = os.path.expanduser("~/.verl_cache/answer_judge_cache")


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text or ""):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def numeric_match(ground_truth: str, predicted: str, tol: float = 0.10) -> Optional[float]:
    """仅当 GT 是**单个数字**时用代码 ±tol 判(免 LLM);否则返回 None 交给 LLM。
    返回 1.0(命中)/0.0(未命中)/None(不适用)。"""
    g = extract_numbers(ground_truth)
    if len(g) != 1:
        return None
    p = extract_numbers(predicted)
    if not p:
        return 0.0
    gt = g[0]
    denom = abs(gt) if gt != 0 else 1.0
    return 1.0 if any(abs(x - gt) / denom <= tol for x in p) else 0.0


def _extract_score(text: str) -> float:
    """从 judge 输出抠出 Score(优先 JSON,失败回退首个数字),对齐作者 extract_score。"""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            return float(obj.get("Score", obj.get("score", 0.0)))
    except Exception:
        pass
    before = text.split("Justification")[0]
    m = re.search(r"[-+]?\d*\.\d+|\d+", before)
    try:
        return float(m.group()) if m else 0.0
    except ValueError:
        return 0.0


class AnswerJudge:
    """注入式判分器。client 需有 `.call(prompt, system=None, max_tokens=...)`。"""

    def __init__(self, client, cache_dir: str = _DEFAULT_CACHE,
                 use_cache: bool = True, numeric_shortcut: bool = True):
        self.client = client
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.numeric_shortcut = numeric_shortcut
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)

    def _key(self, question: str, gt: str, pred: str) -> str:
        h = hashlib.sha256(f"{question}\x00{gt}\x00{pred}".encode()).hexdigest()
        return os.path.join(self.cache_dir, h + ".json")

    def score(self, question: str, ground_truth: str, predicted: str) -> dict:
        """返回 {score, source, justification}。source: cache|numeric|llm。"""
        if not predicted:
            return {"score": 0.0, "source": "empty", "justification": "no prediction"}
        gt, pred = str(ground_truth), str(predicted)

        if self.use_cache:
            kp = self._key(question, gt, pred)
            if os.path.exists(kp):
                try:
                    d = json.load(open(kp)); d["source"] = "cache"; return d
                except Exception:
                    pass

        if self.numeric_shortcut:
            nm = numeric_match(gt, pred)
            if nm is not None:
                res = {"score": nm, "source": "numeric",
                       "justification": "single-number ±10% code check"}
                self._save(question, gt, pred, res)
                return res

        prompt = f"Question: {question}\nGround Truth Answer: {gt}\nPredicted Answer: {pred}\n"
        raw = self.client.call(prompt, system=EVAL_PROMPT, max_tokens=200)
        res = {"score": _extract_score(raw), "source": "llm", "justification": raw[:300]}
        self._save(question, gt, pred, res)
        return res

    def _save(self, question: str, gt: str, pred: str, res: dict) -> None:
        if not self.use_cache:
            return
        try:
            json.dump(res, open(self._key(question, gt, pred), "w"), ensure_ascii=False)
        except Exception:
            pass
