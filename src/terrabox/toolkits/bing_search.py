"""Web search toolkit (slug ``bing_search.search``) backed by the Serper API.

Faithful to OpenEarthAgent's ``GoogleSearch`` worker: queries Google through
``https://google.serper.dev/search`` (header ``X-API-KEY``) and parses the
``answerBox`` / ``knowledgeGraph`` / ``organic`` blocks into the same readable
text format the SFT trajectories were generated with.

Persistent SQLite cache (``.db``) with de-duplication:
  - lookup is by a *normalised* query (lowercased, quotes removed, whitespace
    collapsed) used as the table PRIMARY KEY, so semantically-identical queries
    map to one row and never get stored twice;
  - on a cache HIT the result is returned from the DB with NO network call and
    NO API credit spent;
  - on a MISS we query Serper live, then ``INSERT OR IGNORE`` the
    (query, k, result, raw_json) row so the next identical call is free.

Configuration:
  - API key:  ``SERPER_API_KEY`` env var, or per-call ``api_key`` argument.
  - Cache DB: ``BING_SEARCH_CACHE_DB`` env var, else ``~/.verl_cache/search_cache.db``.
"""

import os
import re
import json
import time
import sqlite3
import pathlib
from typing import Any, Dict, Optional

import requests

from ..core.registry import ToolSpec

SERPER_ENDPOINT = "https://google.serper.dev"
DEFAULT_MAX_OUT_LEN = 1500  # matches OpenEarthAgent GoogleSearch worker


# ---------------------------------------------------------------------------
# Cache (SQLite, de-duplicated by normalised query)
# ---------------------------------------------------------------------------
def _default_db_path() -> str:
    env = os.environ.get("BING_SEARCH_CACHE_DB", "").strip()
    if env:
        return env
    cache_dir = pathlib.Path.home() / ".verl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / "search_cache.db")


def _normalize_query(query: str) -> str:
    """Normalise a query for de-duplication: drop quotes, lowercase, collapse
    whitespace. Two queries that differ only in case/spacing/quotes collapse to
    the same cache key (and thus the same stored row)."""
    q = (query or "").replace('"', "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


class SearchCache:
    """Thin SQLite wrapper. One connection per operation keeps it safe across
    threads/processes during parallel rollouts."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    qnorm   TEXT PRIMARY KEY,
                    query   TEXT NOT NULL,
                    k       INTEGER,
                    result  TEXT NOT NULL,
                    raw_json TEXT,
                    ts      REAL
                )
                """
            )

    def get(self, query: str) -> Optional[str]:
        qnorm = _normalize_query(query)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result FROM search_cache WHERE qnorm = ?", (qnorm,)
            ).fetchone()
        return row[0] if row else None

    def put(self, query: str, k: int, result: str, raw_json: str = "") -> bool:
        """Insert if absent; returns True if a new row was stored, False if the
        normalised query was already present (de-dup, no overwrite)."""
        qnorm = _normalize_query(query)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO search_cache (qnorm, query, k, result, raw_json, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (qnorm, query, int(k), result, raw_json, time.time()),
            )
            return cur.rowcount > 0

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM search_cache").fetchone()[0]


# ---------------------------------------------------------------------------
# Serper query + OEA-faithful result parsing
# ---------------------------------------------------------------------------
def _parse_results(data: Dict[str, Any], k: int, max_out_len: int = DEFAULT_MAX_OUT_LEN) -> str:
    """Port of OpenEarthAgent's GoogleSearch _parse_results (answerBox /
    knowledgeGraph / organic -> '1 - ...\\n\\n2 - ...')."""
    snippets = []

    answer_box = data.get("answerBox", {}) or {}
    if answer_box:
        if answer_box.get("answer"):
            snippets.append(f"Answer box: {answer_box['answer']}")
        elif answer_box.get("snippet"):
            snippets.append(f"Answer box: {answer_box['snippet']}")

    kg = data.get("knowledgeGraph", {}) or {}
    if kg:
        desc = f"{kg.get('title', '')} knowledge graph: {kg.get('type', '')}. {kg.get('description', '')}"
        if kg.get("attributes"):
            attrs = ", ".join(f"{kk}: {vv}" for kk, vv in kg["attributes"].items())
            desc += f" ({attrs})"
        snippets.append(desc)

    for item in (data.get("organic", []) or [])[:k]:
        content = ""
        if item.get("title"):
            content += item["title"] + ": "
        if item.get("snippet"):
            content += item["snippet"]
        if content:
            snippets.append(content)

    if not snippets:
        return "No good Google Search result found."

    out = ""
    for idx, item in enumerate(snippets):
        out += f"{idx + 1} - {item.strip().replace(chr(10), ' ')}\n\n"
    return out[:max_out_len]


def _serper_request(query: str, api_key: str, timeout: int = 30) -> Dict[str, Any]:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    resp = requests.post(
        f"{SERPER_ENDPOINT}/search",
        headers=headers,
        json={"q": query},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Serper API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def search_with_cache(
    query: str,
    k: int = 10,
    api_key: Optional[str] = None,
    db_path: Optional[str] = None,
    timeout: int = 30,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Cache-first Google search. Returns
    {success, query, results, from_cache, newly_cached, result_count}.

    HIT  -> served from SQLite, no API credit spent.
    MISS -> live Serper query, parsed, then de-dup inserted into the cache.
    """
    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "Search query is required", "query": query}

    cache = SearchCache(db_path)

    if not force_refresh:
        cached = cache.get(query)
        if cached is not None:
            return {
                "success": True,
                "query": query,
                "results": cached,
                "text": cached,
                "from_cache": True,
                "newly_cached": False,
                "result_count": cached.count("\n\n"),
            }

    key = api_key or os.getenv("SERPER_API_KEY")
    if not key:
        return {
            "success": False,
            "error": "Cache miss and no SERPER_API_KEY available (set env var or pass api_key).",
            "query": query,
            "from_cache": False,
        }

    try:
        data = _serper_request(query, key, timeout=timeout)
        result_text = _parse_results(data, k)
        newly = cache.put(query, k, result_text, raw_json=json.dumps(data, ensure_ascii=False))
        return {
            "success": True,
            "query": query,
            "results": result_text,
            "text": result_text,
            "from_cache": False,
            "newly_cached": newly,
            "result_count": result_text.count("\n\n"),
        }
    except Exception as e:  # surface error text like the other tools
        return {"success": False, "error": f"Search failed: {e}", "query": query, "from_cache": False}


# ---------------------------------------------------------------------------
# Handler + registration
# ---------------------------------------------------------------------------
def bing_search_handler(arguments: dict, context: dict, account=None) -> Dict[str, Any]:
    """Web search handler (Serper-backed, cache-first)."""
    query = arguments.get("query", "")
    # accept both OEA's `k` and the legacy `max_results`
    k = int(arguments.get("k", arguments.get("max_results", 10)))
    api_key = arguments.get("api_key")
    timeout = int(arguments.get("timeout", 30))
    return search_with_cache(query, k=k, api_key=api_key, timeout=timeout)


def setup(registrar):
    """Register the web-search toolkit (Serper-backed GoogleSearch + SQLite cache)."""
    registrar.toolkit(
        name="bing_search",
        description="Web search toolkit (Google via Serper API) with a persistent, de-duplicated SQLite cache.",
        version="2.0.0",
    )

    bing_search_spec = ToolSpec(
        slug="bing_search.search",
        name="Bing Search",
        description=(
            "Search the web for a factual query (e.g. a region's area in km², a "
            "unit price, an object's function) and return the top result snippets. "
            "Results are cached, so repeated identical queries are free."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query string."},
                "k": {
                    "type": "integer",
                    "description": "Number of top organic results to include (default: 10).",
                    "default": 10,
                },
                "api_key": {
                    "type": "string",
                    "description": "Serper API key (optional; defaults to the SERPER_API_KEY env var).",
                },
            },
            "required": ["query"],
        },
        requires_connection=True,
    )
    registrar.tool(bing_search_spec, bing_search_handler)
