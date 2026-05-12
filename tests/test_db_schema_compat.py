from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from terrabox.db.schema_compat import ensure_sqlite_schema_compat


def test_ensure_sqlite_schema_compat_adds_agent_session_summary_json():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE agent_sessions (
                id VARCHAR(36) PRIMARY KEY,
                user_id_fk VARCHAR(36) NOT NULL,
                messages_json TEXT NOT NULL DEFAULT '[]',
                created_at DATETIME,
                updated_at DATETIME
            )
            """
        ))
        conn.execute(text(
            """
            INSERT INTO agent_sessions (id, user_id_fk, messages_json)
            VALUES ('session-1', 'user-1', '[]')
            """
        ))

    ensure_sqlite_schema_compat(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("agent_sessions")}
    assert "summary_json" in columns
    with engine.connect() as conn:
        value = conn.execute(text(
            "SELECT summary_json FROM agent_sessions WHERE id = 'session-1'"
        )).scalar_one()
    assert value == "[]"
