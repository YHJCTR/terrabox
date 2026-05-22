#!/usr/bin/env python3
"""
Short-term memory (session history) evaluation script.

Tests different summary_keep_recent values (5, 10, 15) vs no-memory baseline.
Uses disaster_sft_dataset.json to construct multi-turn conversations.

Metrics:
  - task_completion_rate: whether the agent produced a meaningful final answer
  - tool_match_f1: overlap between expected_tools and actually called tools
  - avg_turns: average number of agent turns per task

Usage:
    no_proxy=localhost,127.0.0.1 python scripts/eval_memory_ablation.py

Output is printed to stdout (no extra files created).
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# ── Make project importable ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_sft_data(path: str | None = None) -> list[dict]:
    """Load OpenEarth dataset and return list of task dicts."""
    if path is None:
        path = os.path.join(_PROJECT_ROOT, "data", "openearth", "train.json")
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # OpenEarth dataset is already a list
    samples = data if isinstance(data, list) else data.get("samples", [])
    if not samples:
        print("[ERROR] No samples found in dataset", file=sys.stderr)
        sys.exit(1)
    return samples


def build_multi_turn_pairs(samples: list[dict], n: int = 50) -> list[dict]:
    """Construct multi-turn conversation pairs from SFT samples.

    Each pair has:
      - turn1: a context-setting question (from one sample)
      - turn2: a follow-up question that references turn1's context (from another sample)

    This tests whether the session memory preserves information across turns.
    """
    random.seed(42)
    pairs = []
    shuffled = list(samples)
    random.shuffle(shuffled)

    for i in range(min(n, len(shuffled) // 2)):
        s1 = shuffled[i * 2]
        s2 = shuffled[i * 2 + 1]

        # Support both old (question/expected_tools) and new (prompt/tool_calls) formats
        q1 = s1.get("question", "") or s1.get("prompt", "")
        q2_raw = s2.get("question", "") or s2.get("prompt", "")

        # Extract expected tools from various formats
        def _extract_tools(sample):
            # OpenEarth format: conversation -> actions
            conversation = sample.get("conversation", [])
            tools = []
            for msg in conversation:
                if msg.get("from") == "gpt":
                    value = msg.get("value", "")
                    try:
                        # Parse JSON actions
                        action_data = json.loads(value)
                        actions = action_data.get("actions", [])
                        for action in actions:
                            tool_name = action.get("name", "")
                            if tool_name and tool_name != "Terminate":
                                tools.append(tool_name)
                    except:
                        pass
            
            # Fallback to other formats
            if not tools and sample.get("expected_tools"):
                return sample["expected_tools"]
            if not tools:
                tc = sample.get("tool_calls", [])
                if isinstance(tc, list):
                    return [t.get("tool", "") for t in tc if isinstance(t, dict) and t.get("tool")]
            return tools

        expected_tools_1 = _extract_tools(s1)
        expected_tools_2 = _extract_tools(s2)

        # Build a follow-up that references turn1 context
        q2 = (
            f"基于刚才的分析，{q2_raw}"
            if not q2_raw.startswith("基于")
            else q2_raw
        )

        # Extract data context from s1 for the follow-up
        data_dir = s1.get("data_dir", "")
        data_files = s1.get("data_files", [])
        images = s1.get("images", [])

        pairs.append({
            "id": f"pair_{i:03d}",
            "turn1_question": q1,
            "turn1_data_dir": data_dir,
            "turn1_data_files": data_files[:5],
            "turn1_images": images,
            "turn1_expected_tools": expected_tools_1,
            "turn2_question": q2,
            "turn2_expected_tools": expected_tools_2,
        })

    return pairs


# ── Metrics ──────────────────────────────────────────────────────────────────

def tool_match_f1(called: list[str], expected: list[str]) -> float:
    """Compute F1 of tool set overlap."""
    if not expected:
        return 1.0 if not called else 0.0
    called_set = set(called)
    expected_set = set(expected)
    if not called_set:
        return 0.0
    precision = len(called_set & expected_set) / len(called_set)
    recall = len(called_set & expected_set) / len(expected_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def is_task_complete(final_answer: str, called_tools: list[str]) -> bool:
    """Heuristic: task is complete if agent called at least one tool and gave a non-error answer."""
    if not called_tools:
        return False
    error_signals = ["error", "failed", "timed out", "无法完成", "无法执行"]
    answer_lower = final_answer.lower()
    if any(sig in answer_lower for sig in error_signals) and len(final_answer) < 200:
        return False
    return True


# ── Agent runner with configurable memory ────────────────────────────────────

@dataclass
class MemoryConfig:
    """Controls how session history is managed."""
    enabled: bool = True
    summary_keep_recent: int = 5
    max_history_messages: int = 20
    summary_threshold: int = 15


class InMemorySessionStore:
    """Simulates AgentSession DB storage without touching the real database."""

    def __init__(self):
        self.sessions: dict[str, dict] = {}

    def get_or_create(self, session_id: str) -> dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "messages": [],       # list of LangChain message dicts
                "summary": "",        # compressed summary text
            }
        return self.sessions[session_id]

    def add_messages(self, session_id: str, new_msgs: list):
        sess = self.get_or_create(session_id)
        sess["messages"].extend(new_msgs)

    def truncate(self, session_id: str, keep_recent: int, max_messages: int, summary_threshold: int):
        """Apply memory policy: compress old messages, keep recent N."""
        sess = self.get_or_create(session_id)
        msgs = sess["messages"]

        if len(msgs) <= summary_threshold:
            return

        # Messages to compress (everything except the last keep_recent)
        to_compress = msgs[:-keep_recent]
        recent = msgs[-keep_recent:]

        # Build simple summary from compressed messages
        summary_parts = []
        for m in to_compress:
            role = m.get("type", "unknown")
            content = m.get("content", "")
            if content and len(content) > 10:
                summary_parts.append(f"[{role}] {content[:200]}")

        if summary_parts:
            existing = sess.get("summary", "")
            new_summary = existing + "\n" + "\n".join(summary_parts[-10:])
            sess["summary"] = new_summary[-3000:]  # cap summary length

        sess["messages"] = recent

    def get_context_messages(self, session_id: str) -> list[dict]:
        """Return messages + summary as context for next turn."""
        sess = self.get_or_create(session_id)
        context = []
        if sess.get("summary"):
            context.append({
                "type": "system",
                "content": f"[Terrabox conversation memory]\n{sess['summary']}",
            })
        context.extend(sess["messages"])
        return context


def run_agent_turn(
    llm,
    tools: list,
    session_store: InMemorySessionStore,
    session_id: str,
    user_message: str,
    mem_config: MemoryConfig,
) -> dict:
    """Run one agent turn with session memory context."""
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
    from langgraph.prebuilt import create_react_agent

    # Build context from session memory
    if mem_config.enabled:
        context_dicts = session_store.get_context_messages(session_id)
        context_msgs = []
        for d in context_dicts:
            t = d.get("type", "")
            c = d.get("content", "")
            if t == "system":
                context_msgs.append(SystemMessage(content=c))
            elif t == "human":
                context_msgs.append(HumanMessage(content=c))
            elif t == "ai":
                context_msgs.append(AIMessage(content=c))
            elif t == "tool":
                context_msgs.append(ToolMessage(content=c, tool_call_id=d.get("tool_call_id", "")))
        input_messages = context_msgs + [HumanMessage(content=user_message)]
    else:
        input_messages = [HumanMessage(content=user_message)]

    # Run agent
    system_prompt = (
        "You are a geospatial analysis assistant with access to various tools. "
        "Use tools to answer the user's question. Provide a clear final answer.\n"
    )
    agent = create_react_agent(llm, tools, prompt=system_prompt)

    try:
        result = agent.invoke(
            {"messages": input_messages},
            config={"recursion_limit": 12},
        )
    except Exception as exc:
        return {
            "tools_called": [],
            "final_answer": f"Agent error: {exc}",
            "messages_dicts": [],
            "turn_count": 0,
        }

    # Extract results — collect tool names from all AI message tool_calls
    tools_called = []
    for msg in result.get("messages", []):
        for tc in getattr(msg, "tool_calls", []):
            # LangChain tool_calls use double-underscore for dots: geo_raster__calculate_index
            name = tc.get("name", "").replace("__", ".")
            if name:
                tools_called.append(name)

    final_answer = ""
    msgs = result.get("messages", [])
    if msgs and hasattr(msgs[-1], "content"):
        final_answer = str(msgs[-1].content)

    # Count turns (AI messages that aren't just tool calls)
    turn_count = sum(
        1 for m in msgs
        if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None)
    )

    # Store new messages to session
    new_msgs = []
    for m in msgs[len(input_messages):]:
        msg_dict = {"type": m.__class__.__name__.lower().replace("message", "")}
        if hasattr(m, "content") and m.content:
            msg_dict["content"] = str(m.content)[:500]
        for tc in getattr(m, "tool_calls", []):
            msg_dict.setdefault("tool_calls", []).append({
                "name": tc.get("name", "").replace("__", "."),
            })
        new_msgs.append(msg_dict)

    if mem_config.enabled:
        session_store.add_messages(session_id, new_msgs)
        session_store.truncate(
            session_id,
            keep_recent=mem_config.summary_keep_recent,
            max_messages=mem_config.max_history_messages,
            summary_threshold=mem_config.summary_threshold,
        )

    return {
        "tools_called": tools_called,
        "final_answer": final_answer,
        "messages_dicts": new_msgs,
        "turn_count": turn_count,
    }


# ── Experiment runner ────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    config_name: str
    task_completion_rate: float = 0.0
    tool_match_f1_avg: float = 0.0
    avg_turns: float = 0.0
    per_task_details: list = field(default_factory=list)


def run_experiment(
    pairs: list[dict],
    mem_config: MemoryConfig,
    config_name: str,
    llm,
    tools: list,
) -> ExperimentResult:
    """Run all pairs with the given memory config and collect metrics."""
    result = ExperimentResult(config_name=config_name)
    f1_scores = []
    completion_flags = []
    turn_counts = []

    for i, pair in enumerate(pairs):
        session_id = f"{config_name}_{pair['id']}_{uuid.uuid4().hex[:8]}"
        session_store = InMemorySessionStore()

        # Turn 1: context-setting question
        turn1_msg = pair["turn1_question"]
        if pair.get("turn1_data_dir"):
            turn1_msg += f"\n\n[Data directory: {pair['turn1_data_dir']}]"
        if pair.get("turn1_data_files"):
            turn1_msg += f"\n[Data files: {', '.join(pair['turn1_data_files'])}]"

        t1 = run_agent_turn(llm, tools, session_store, session_id, turn1_msg, mem_config)

        # Turn 2: follow-up that references turn1 context
        turn2_msg = pair["turn2_question"]
        t2 = run_agent_turn(llm, tools, session_store, session_id, turn2_msg, mem_config)

        # Evaluate on turn2 (the one that needs memory)
        expected = pair.get("turn2_expected_tools", [])
        f1 = tool_match_f1(t2["tools_called"], expected)
        complete = is_task_complete(t2["final_answer"], t2["tools_called"])

        f1_scores.append(f1)
        completion_flags.append(1 if complete else 0)
        turn_counts.append(t2.get("turn_count", 1))

        result.per_task_details.append({
            "pair_id": pair["id"],
            "turn2_tools_called": t2["tools_called"],
            "turn2_expected_tools": expected,
            "f1": round(f1, 3),
            "complete": complete,
            "turn2_answer_preview": t2["final_answer"][:200],
        })

        if (i + 1) % 10 == 0:
            print(f"  [{config_name}] {i+1}/{len(pairs)} done", file=sys.stderr)

    result.task_completion_rate = sum(completion_flags) / len(completion_flags) if completion_flags else 0
    result.tool_match_f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    result.avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0

    return result


# ── P-value calculation ──────────────────────────────────────────────────────

def compute_p_value(baseline_rates: list[float], treatment_rates: list[float]) -> float:
    """Two-proportion z-test for comparing task completion rates.

    P-value answers: "How likely is it that the observed difference happened by chance?"
    - p < 0.05: the difference is statistically significant (likely real)
    - p >= 0.05: the difference could be random noise
    """
    import math

    n1 = len(baseline_rates)
    n2 = len(treatment_rates)
    p1 = sum(baseline_rates) / n1 if n1 else 0
    p2 = sum(treatment_rates) / n2 if n2 else 0

    p_pool = (sum(baseline_rates) + sum(treatment_rates)) / (n1 + n2) if (n1 + n2) else 0

    if p_pool == 0 or p_pool == 1:
        return 1.0

    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 1.0

    z = (p2 - p1) / se

    # Approximate two-tailed p-value from normal CDF
    # Using the approximation: P(|Z| > |z|)
    abs_z = abs(z)
    # Abramowitz & Stegun approximation for erfc
    t = 1.0 / (1.0 + 0.2316419 * abs_z)
    d = 0.3989422804014327
    p_val = 2 * d * math.exp(-0.5 * abs_z * abs_z) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    )
    return min(max(p_val, 0.0), 1.0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Short-term Memory Ablation Study")
    print("=" * 70)

    # Load data
    samples = load_sft_data()
    print(f"\nLoaded {len(samples)} SFT samples")

    pairs = build_multi_turn_pairs(samples, n=10)  # Reduced for faster evaluation
    print(f"Built {len(pairs)} multi-turn conversation pairs")

    # Initialize agent infrastructure
    from terrabox.extensions import load_builtin_toolkits
    load_builtin_toolkits()

    from terrabox.agent.config import load_config
    from terrabox.agent.llm import get_llm
    from terrabox.agent.tools import build_langchain_tools
    from terrabox.evolution.shared.mock_user import MockUser

    config = load_config()
    llm = get_llm(config)
    user = MockUser()
    tools = build_langchain_tools(user)
    print(f"Loaded {len(tools)} tools, LLM ready")

    # Define experiment configs
    configs = [
        ("no_memory", MemoryConfig(enabled=False)),
        ("keep_recent_5", MemoryConfig(enabled=True, summary_keep_recent=5)),
        ("keep_recent_10", MemoryConfig(enabled=True, summary_keep_recent=10)),
        ("keep_recent_15", MemoryConfig(enabled=True, summary_keep_recent=15)),
    ]

    # Run experiments
    results: list[ExperimentResult] = []
    for name, mem_cfg in configs:
        print(f"\n{'─' * 50}")
        print(f"Running: {name} (keep_recent={mem_cfg.summary_keep_recent}, enabled={mem_cfg.enabled})")
        t0 = time.time()
        exp_result = run_experiment(pairs, mem_cfg, name, llm, tools)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        # Print intermediate result immediately
        print(f"  -> Completion: {exp_result.task_completion_rate*100:.1f}%, Tool F1: {exp_result.tool_match_f1_avg:.3f}, Avg Turns: {exp_result.avg_turns:.1f}")
        results.append(exp_result)

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Completion%':>12} {'Tool F1':>10} {'Avg Turns':>10}")
    print("-" * 55)
    for r in results:
        print(
            f"{r.config_name:<20} "
            f"{r.task_completion_rate * 100:>11.1f}% "
            f"{r.tool_match_f1_avg:>10.3f} "
            f"{r.avg_turns:>10.1f}"
        )

    # ── P-values vs no_memory baseline ───────────────────────────────────────
    baseline = results[0]  # no_memory
    baseline_completions = [1 if d["complete"] else 0 for d in baseline.per_task_details]

    print(f"\n{'─' * 55}")
    print("P-values (vs no_memory baseline):")
    print(f"{'Config':<20} {'p-value':>10} {'Significant':>12}")
    print("-" * 45)
    for r in results[1:]:
        treatment_completions = [1 if d["complete"] else 0 for d in r.per_task_details]
        p_val = compute_p_value(baseline_completions, treatment_completions)
        sig = "Yes" if p_val < 0.05 else "No"
        print(f"{r.config_name:<20} {p_val:>10.4f} {sig:>12}")

    # ── Per-task detail (sample) ─────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print("Sample per-task details (first 5 tasks, keep_recent_10):")
    r10 = next(r for r in results if r.config_name == "keep_recent_10")
    for d in r10.per_task_details[:5]:
        print(f"  {d['pair_id']}: F1={d['f1']:.3f} complete={d['complete']} "
              f"called={d['turn2_tools_called'][:3]} expected={d['turn2_expected_tools'][:3]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
