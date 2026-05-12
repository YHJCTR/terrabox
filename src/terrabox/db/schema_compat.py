"""Small schema compatibility patches for development SQLite databases."""
from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def ensure_sqlite_schema_compat(engine: Engine) -> None:
    """Apply lightweight SQLite schema fixes that create_all() cannot perform."""
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    if "agent_sessions" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("agent_sessions")}
    if "summary_json" in columns:
        return

    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE agent_sessions "
            "ADD COLUMN summary_json TEXT NOT NULL DEFAULT '[]'"
        ))
