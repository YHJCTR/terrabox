from ..shared.prompt_builder import PromptAugmenter
from ..shared.frozen_memory import FrozenIndex


class ReasoningBankPromptInjector(PromptAugmenter):
    strict_augmentation = True

    def __init__(self, store_dir, top_k=5, embedding=None):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.index = FrozenIndex(store_dir, "reasoningbank", embedding)
        self.top_k = top_k
        self.last_retrieval = []

    def augment(self, user_query, **kwargs):
        # Dataset labels in kwargs deliberately do not participate.
        query = ("Instruct: Given prior geospatial tool-use queries, analyze the current query's "
                 "intent and select relevant prior queries that could help resolve it.\nQuery: " + user_query)
        self.last_retrieval = self.index.search(query, self.top_k)
        memories = [m for record in self.last_retrieval for m in record["memories"]]
        content = "\n\n".join(f"Title: {m['title']}\nDescription: {m['description']}\nContent: {m['content']}"
                               for m in memories)
        return self.BASE_SYSTEM + "\n\n[ReasoningBank: reusable guidance, not current evidence]\n" + content
