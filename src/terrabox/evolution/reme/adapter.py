import json

from ..shared.frozen_memory import (GENERALIZE, UpstreamPrompts, digest, parse_json,
                                    unit_score, write_json)

DEFAULT_SOURCE = "/data1/yuhongjie2/ReMe-procedural"
COMMIT = "2f37a159b72a04ac1885a7db7f1a663a833e7791"
PREFIX = "reme/extension/procedural_memory/"
PROMPTS = {
    "segment": ("summary/trajectory_segmentation.yaml", "step_segmentation_prompt"),
    "success": ("summary/success_extraction.yaml", "success_step_task_memory_prompt"),
    "failure": ("summary/failure_extraction.yaml", "failure_step_task_memory_prompt"),
    "compare": ("summary/comparative_extraction.yaml", "soft_comparative_step_task_memory_prompt"),
    "validate": ("summary/memory_validation.yaml", "task_memory_validation_prompt"),
    "rerank": ("retrieve/rerank_memory.yaml", "memory_rerank_prompt"),
    "rewrite": ("retrieve/rewrite_memory.yaml", "memory_rewrite_prompt"),
}


class ReMeAdapter:
    def __init__(self, upstream=DEFAULT_SOURCE):
        self.source = UpstreamPrompts(upstream)
        self.prompts = {name: self.source.get(PREFIX + path, key) for name, (path, key) in PROMPTS.items()}

    def provenance(self):
        return {"repository": "https://github.com/agentscope-ai/ReMe", "reference_commit": COMMIT,
                "source_path": str(self.source.root), "prompt_sha256": self.source.files,
                "implementation": "official_prompt_adapted_frozen_procedural",
                "differences": ["OEA conversation schema and shared LLM client",
                                "visible-only self-judge replaces benchmark scores",
                                "success boolean routes extraction; scalar score used for comparison",
                                "soft comparison within identical visible queries only",
                                "exact content dedup, not upstream optional embedding dedup",
                                "frozen eval: no utility updates/deletion or online memory addition",
                                "strict JSON parsing, inclusive zero-based segment ends per prompt",
                                "semantic recall then LLM rerank; optional rewriting disabled by default"]}

    def ask(self, client, name, **values):
        return parse_json(client.call(self.prompts[name].format(**values), system=GENERALIZE, max_tokens=4096))

    def validate(self, client, items, source):
        if not isinstance(items, list):
            raise ValueError("ReMe extraction must be a JSON list")
        records = []
        for item in items:
            condition, experience = item.get("when_to_use"), item.get("experience")
            if not isinstance(condition, str) or not condition.strip() or not isinstance(experience, str) or not experience.strip():
                raise ValueError("ReMe memory lacks condition/experience")
            confidence = unit_score(item["confidence"])
            result = self.ask(client, "validate", condition=condition, task_memory_content=experience)
            if type(result.get("is_valid")) is not bool:
                raise ValueError("ReMe validation lacks boolean is_valid")
            score = unit_score(result["score"])
            if result["is_valid"] and score >= 0.5:
                records.append({"when_to_use": condition, "content": experience,
                                "confidence": confidence, "validation_score": score,
                                "source": source, "retrieval_text": condition + " " + experience})
        return records

    def extract(self, client, trajectory, outcome):
        messages = trajectory["messages"]
        result = self.ask(client, "segment", query=trajectory["query"], total_steps=len(messages),
                          trajectory_content=json.dumps(messages, ensure_ascii=False))
        points = result.get("segment_points")
        if (not isinstance(points, list) or any(type(i) is not int or not 0 <= i < len(messages) for i in points)
                or points != sorted(set(points))):
            raise ValueError("Invalid ReMe segmentation boundaries")
        ends = sorted(set([i + 1 for i in points] + [len(messages)]))
        start, records = 0, []
        name = "success" if outcome["success"] else "failure"
        for end in ends:
            items = self.ask(client, name, query=trajectory["query"],
                             step_sequence=json.dumps(messages[start:end], ensure_ascii=False),
                             context=outcome["reason"], outcome=name)
            if not isinstance(items, list) or len(items) > 3:
                raise ValueError("ReMe segment returned more than 3 memories")
            records.extend(self.validate(client, items, name))
            start = end
        return records

    def finish(self, client, trajectories, entries, store):
        records = [record for entry in entries for record in entry["records"]]
        groups = {}
        for trajectory, entry in zip(trajectories, entries):
            groups.setdefault(trajectory["query"], []).append((trajectory, entry))
        for group in groups.values():
            ordered = sorted(group, key=lambda x: x[1]["outcome"]["score"])
            low, high = ordered[0], ordered[-1]
            if low[1]["outcome"]["score"] == high[1]["outcome"]["score"]:
                continue
            cache = store / "comparison_cache" / (digest([low[1], high[1]]) + ".json")
            if cache.exists():
                extra = json.loads(cache.read_text())
            else:
                items = self.ask(client, "compare", higher_steps=json.dumps(high[0], ensure_ascii=False),
                                 lower_steps=json.dumps(low[0], ensure_ascii=False),
                                 higher_score=high[1]["outcome"]["score"], lower_score=low[1]["outcome"]["score"])
                if not isinstance(items, list) or len(items) > 2:
                    raise ValueError("ReMe comparison returned more than 2 memories")
                extra = self.validate(client, items, "comparative")
                write_json(cache, extra)
            records.extend(extra)
        # Exact dedup is deterministic; no quality/label-derived retrieval bonus.
        return list({digest(r["retrieval_text"]): r for r in records}.values())
