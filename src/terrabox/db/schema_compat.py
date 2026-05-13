"""Small schema compatibility patches for development SQLite databases."""
from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def ensure_sqlite_schema_compat(engine: Engine) -> None:
    """Apply lightweight SQLite schema fixes that create_all() cannot perform."""
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "agent_sessions" not in table_names:
        return

    columns = {column["name"] for column in inspector.get_columns("agent_sessions")}
    with engine.begin() as conn:
        if "summary_json" not in columns:
            conn.execute(text(
                "ALTER TABLE agent_sessions "
                "ADD COLUMN summary_json TEXT NOT NULL DEFAULT '[]'"
            ))
        if "agent_approvals" in table_names:
            approval_columns = {column["name"] for column in inspector.get_columns("agent_approvals")}
            if "metadata_json" not in approval_columns:
                conn.execute(text(
                    "ALTER TABLE agent_approvals "
                    "ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                ))
