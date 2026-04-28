"""GraphSkillEvo: Relational Skill Graphs with Graph-Based Routing.

Extends SkillRL by adding explicit skill-to-skill relationships and graph-based
retrieval instead of pure keyword matching. Automatically detects conflicts and
prerequisite orderings.

Novel contributions:
1. SkillGraph: NetworkX-based skill graph with typed relations
2. LLMRelationExtractor: Automatic relation discovery via LLM
3. Graph-based retrieval: BFS traversal + ranking instead of BM25

Paper idea: "GraphSkillEvo: Relational Skill Graphs with Graph Neural Network Routing
for Self-Evolving Tool-Calling Agents"
"""
from .relation_extractor import LLMRelationExtractor
from .skill_graph import SkillGraph
from .prompt_injector import GraphSkillEvoPromptInjector

__all__ = ["SkillGraph", "LLMRelationExtractor", "GraphSkillEvoPromptInjector"]
