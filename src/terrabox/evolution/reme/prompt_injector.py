import os

from ..shared.prompt_builder import PromptAugmenter
from ..shared.frozen_memory import BoundedClient, FrozenIndex, parse_json
from .adapter import ReMeAdapter, DEFAULT_SOURCE


class ReMePromptInjector(PromptAugmenter):
    strict_augmentation = True

    def __init__(self, store_dir, top_k=5, embedding=None, client=None, upstream=None,
                 recall_k=None, rewrite=False):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.index = FrozenIndex(store_dir, "reme", embedding)
        self.adapter = ReMeAdapter(upstream or os.environ.get("TERRABOX_REME_SOURCE", DEFAULT_SOURCE))
        recorded = self.index.bank["config"]["upstream"]["prompt_sha256"]
        if self.adapter.source.files != recorded:
            raise ValueError("ReMe prompts changed since bank build")
        if client is None:
            from terrabox.agent.llm_provider import make_llm_client
            client = make_llm_client(os.environ.get("TERRABOX_REME_PROVIDER", "local"))
        self.client, self.top_k = BoundedClient(client, 100000), top_k
        self.recall_k = recall_k or max(10, top_k)
        if self.recall_k < top_k:
            raise ValueError("recall_k must be >= top_k")
        self.rewrite = rewrite
        self.last_retrieval = []

    def augment(self, user_query, **kwargs):
        candidates = self.index.search(user_query, self.recall_k)
        text = "\n---\n".join(f"Candidate {i}:\nCondition: {m['when_to_use']}\nExperience: {m['content']}"
                                for i, m in enumerate(candidates))
        prompt = self.adapter.prompts["rerank"].format(query=user_query, candidates=text, num_candidates=len(candidates))
        order = parse_json(self.client.call(prompt, max_tokens=2048))["ranked_indices"]
        if (not isinstance(order, list) or any(type(i) is not int for i in order)
                or sorted(order) != list(range(len(candidates)))):
            raise ValueError("ReMe reranker must return a full permutation; no silent fallback")
        self.last_retrieval = [candidates[i] for i in order[:self.top_k]]
        content = "\n".join(f"Memory {i}:\nWhen to use: {m['when_to_use']}\nContent: {m['content']}"
                             for i, m in enumerate(self.last_retrieval, 1))
        if self.rewrite:
            prompt = self.adapter.prompts["rewrite"].format(current_query=user_query,
                                                            current_context="", original_context=content)
            content = parse_json(self.client.call(prompt, max_tokens=2048))["rewritten_context"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty rewritten ReMe context")
        return self.BASE_SYSTEM + "\n\n[ReMe: reusable guidance, not current evidence]\n" + content
