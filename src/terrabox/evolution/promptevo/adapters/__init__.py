"""Optional adapters for using promptevo with external agent projects.

Adapters in this package should stay thin: translate project-specific prompts,
rollout traces, and metrics into the Protocols defined in
``terrabox.evolution.promptevo.interfaces`` without importing the external
project when simple file parsing is enough.
"""

__all__ = []
