from .adapter import DEFAULT_SOURCE, ReasoningBankAdapter
from ..shared.frozen_memory import run_cli

if __name__ == "__main__":
    run_cli("reasoningbank", ReasoningBankAdapter, DEFAULT_SOURCE)
