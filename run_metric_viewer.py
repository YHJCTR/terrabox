from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start the Terrabox metric viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    src = str(repo_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.chdir(repo_root)
    try:
        import uvicorn

        uvicorn.run("metricViewer.app:app", host=args.host, port=args.port, reload=args.reload)
    except ModuleNotFoundError:
        if args.reload:
            print("--reload requires uvicorn; falling back to a non-reload stdlib server.")
        from metricViewer.simple_server import run

        run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
