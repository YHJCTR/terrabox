"""Pure GRPO baseline wrapper around the shared agent_rl runner."""
from __future__ import annotations

import sys

from terrabox.evolution.agent_rl.runner import main as agent_rl_main


def main() -> None:
    # Keep this method directory as the user-facing entry while reusing the
    # shared Swift/veRL adapters.  If the caller does not pass --method, default
    # to the pure GRPO method namespace.
    if "--method" not in sys.argv and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sys.argv[2:2] = ["--method", "rl_grpo"]
    agent_rl_main()


if __name__ == "__main__":
    main()
