"""Mock user object for offline experiment runs without a live DB session."""
from __future__ import annotations

import uuid


class MockUser:
    """Minimal user stub compatible with build_langchain_tools(user).

    Tools handlers receive this as the `account` parameter.
    """

    def __init__(self, user_id: str = "evolution-offline"):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.email = f"{user_id}@evolution.local"
        self.is_active = True
        self.api_keys = []

    def __repr__(self) -> str:
        return f"MockUser(user_id={self.user_id!r})"
