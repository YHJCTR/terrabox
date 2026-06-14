#!/usr/bin/env python
"""Pre-warm the web-search SQLite cache with the gold GoogleSearch queries of a
dataset, so later rollouts hit the cache (free) instead of spending Serper
credits on every run.

It walks the OpenEarth-style ``conversation`` trajectories, collects every
``GoogleSearch`` / ``bing_search.search`` action's ``query`` argument, de-dupes
them, and runs each MISS once through Serper via
``terrabox.toolkits.bing_search.search_with_cache`` (which stores the result in
the de-duplicated SQLite cache). Already-cached queries cost nothing, so the
script is safe to re-run / resume.

Usage:
  export SERPER_API_KEY=...        # or pass --api-key
  python scripts/prewarm_search_cache.py \
      --task-files data/openearth/test.json data/openearth/train.json \
      [--db-path ~/.verl_cache/search_cache.db] [--sleep 0.5] [--dry-run]

Credit cost == number of queries actually fetched live (printed as a plan first).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make the package importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from terrabox.toolkits.bing_search import (
    SearchCache,
    search_with_cache,
    _normalize_query,
    _ensure_env_loaded,
)

SEARCH_TOOL_NAMES = {"GoogleSearch", "bing_search.search", "bing_search", "Bing"}


def extract_queries(path: str) -> list[str]:
    d = json.load(open(path, encoding="utf-8"))
    tasks = d if isinstance(d, list) else d.get("tasks", d)
    out: list[str] = []
    for t in tasks:
        for turn in (t.get("conversation") or []):
            if turn.get("from") != "gpt":
                continue
            try:
                obj = json.loads(turn.get("value", ""))
            except Exception:
                continue
            for a in (obj.get("actions") or []):
                nm = str(a.get("name") or a.get("tool") or a.get("function_name") or "")
                if nm in SEARCH_TOOL_NAMES or "oogleSearch" in nm:
                    q = (a.get("arguments") or {}).get("query")
                    if q and str(q).strip():
                        out.append(str(q).strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-files", nargs="+", required=True,
                    help="OpenEarth-style JSON files with gold conversations.")
    ap.add_argument("--db-path", default=None,
                    help="Cache DB path (default: BING_SEARCH_CACHE_DB or ~/.verl_cache/search_cache.db).")
    ap.add_argument("--api-key", default=None, help="Serper key (else SERPER_API_KEY env).")
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds between live calls.")
    ap.add_argument("--limit", type=int, default=0, help="Cap number of live fetches (0=all).")
    ap.add_argument("--dry-run", action="store_true", help="Only report what WOULD be fetched.")
    args = ap.parse_args()

    if not args.api_key:
        _ensure_env_loaded()  # pick up SERPER_API_KEY from repo-root .env by default
    api_key = args.api_key or os.getenv("SERPER_API_KEY")

    # Collect + de-dupe (by the same normalisation the cache uses).
    raw: list[str] = []
    for f in args.task_files:
        qs = extract_queries(f)
        print(f"  {f}: {len(qs)} gold search calls")
        raw.extend(qs)
    by_norm: dict[str, str] = {}
    for q in raw:
        by_norm.setdefault(_normalize_query(q), q)  # keep first surface form
    uniques = list(by_norm.values())

    cache = SearchCache(args.db_path)
    already = [q for q in uniques if cache.get(q) is not None]
    todo = [q for q in uniques if cache.get(q) is None]
    if args.limit:
        todo = todo[: args.limit]

    print(f"\nTotal gold calls: {len(raw)}")
    print(f"Unique queries:   {len(uniques)}")
    print(f"Already cached:   {len(already)}  (free)")
    print(f"To fetch live:    {len(todo)}  (~{len(todo)} Serper credits)")
    print(f"Cache DB:         {cache.db_path}  (rows now: {cache.count()})")

    if args.dry_run:
        print("\n[dry-run] no live calls made.")
        for q in todo[:20]:
            print("   would fetch:", q)
        return
    if not todo:
        print("\nNothing to fetch — cache already warm.")
        return
    if not api_key:
        print("\nERROR: queries need fetching but no SERPER_API_KEY / --api-key given.")
        sys.exit(1)

    fetched = failed = 0
    for i, q in enumerate(todo, 1):
        res = search_with_cache(q, api_key=api_key)
        if res.get("success"):
            fetched += 1
            tag = "cached" if res.get("newly_cached") else "dup"
            print(f"  [{i}/{len(todo)}] OK ({tag}): {q[:70]}")
        else:
            failed += 1
            print(f"  [{i}/{len(todo)}] FAIL: {q[:70]}  -> {res.get('error')}")
        time.sleep(args.sleep)

    print(f"\nDone. fetched={fetched} failed={failed}. Cache rows now: {cache.count()}")


if __name__ == "__main__":
    main()
