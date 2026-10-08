from .adapter import DEFAULT_SOURCE, ReMeAdapter
from ..shared.frozen_memory import run_cli

if __name__ == "__main__":
    run_cli("reme", ReMeAdapter, DEFAULT_SOURCE)
