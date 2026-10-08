"""Keep success/failure induction and prior-query-to-memory retrieval distinct."""
import json
import re

from ..shared.frozen_memory import GENERALIZE, UpstreamPrompts

DEFAULT_SOURCE = "/data1/yuhongjie2/reasoning-bank"
COMMIT = "ed80611788292ea739f1effd31f16c53823b8a0d"


class ReasoningBankAdapter:
    def __init__(self, upstream=DEFAULT_SOURCE):
        self.source = UpstreamPrompts(upstream)
        self.prompts = {key: self.source.get("WebArena/prompts/memory_instruction.py", key)
                        for key in ("SUCCESSFUL_SI", "FAILED_SI")}

    def provenance(self):
        return {"repository": "https://github.com/google-research/reasoning-bank",
                "reference_commit": COMMIT, "source_path": str(self.source.root),
                "prompt_sha256": self.source.files, "implementation": "official_prompt_adapted",
                "differences": ["OEA harness and domain wording instead of WebArena",
                                "visible-trajectory self-judge instead of browser autoeval",
                                "explicit OpenAI-compatible embedding model instead of Gemini",
                                "train-only frozen bank, no eval insertion or MaTTS"]}

    def extract(self, client, trajectory, outcome):
        system = self.prompts["SUCCESSFUL_SI" if outcome["success"] else "FAILED_SI"]
        system = system.replace("web navigation", "geospatial tool use") + "\n" + GENERALIZE
        text = client.call(json.dumps(trajectory, ensure_ascii=False) +
                           "\nSelf-evaluation: " + json.dumps(outcome), system=system, max_tokens=2048)
        blocks = re.split(r"(?m)^# Memory Item\s+\d+[^\n]*\n", text)[1:]
        if not 1 <= len(blocks) <= 3:
            raise ValueError("ReasoningBank must return 1–3 Markdown memory items")
        memories = []
        for block in blocks:
            item = {}
            for key in ("Title", "Description", "Content"):
                match = re.search(r"(?ms)^## " + key + r"[ \t]*\n?(.*?)(?=^## |\Z)", block)
                if not match or not match[1].strip().strip('`'):
                    raise ValueError(f"Missing ReasoningBank {key}")
                item[key.lower()] = match[1].strip().rstrip('`').strip()
            memories.append(item)
        return [{"retrieval_text": trajectory["query"], "memories": memories}]

    def finish(self, client, trajectories, entries, store):
        return [record for entry in entries for record in entry["records"]]
