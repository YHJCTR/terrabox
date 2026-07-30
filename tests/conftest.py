from __future__ import annotations

from typing import Optional


class MockRegistrar:
    """Minimal registrar used by toolkit smoke tests."""

    def __init__(self):
        self.toolkits = {}
        self.tools = {}
        self.handlers = {}

    def toolkit(self, name: str, description: str = "", version: str = ""):
        self.toolkits[name] = {"description": description, "version": version}

    def tool(self, toolspec, handler):
        self.tools[toolspec.slug] = toolspec
        self.handlers[toolspec.slug] = handler

    def call(self, slug: str, arguments: dict, context: Optional[dict] = None, account=None):
        if context is None:
            context = {}
        handler = self.handlers.get(slug)
        if handler is None:
            raise KeyError(f"Tool not registered: {slug}. Available: {list(self.handlers.keys())}")
        return handler(arguments, context, account)


class NullQuery:
    def filter_by(self, **_kwargs):
        return self

    def first(self):
        return None


class NullDb:
    def add(self, _record):
        pass

    def flush(self):
        pass

    def query(self, *_args, **_kwargs):
        return NullQuery()


class ObjectQuery(NullQuery):
    def first(self):
        return object()


class ObjectDb:
    def query(self, *_args, **_kwargs):
        return ObjectQuery()


class RecordQuery:
    def __init__(self, record):
        self.record = record
        self.filters = {}

    def filter_by(self, **kwargs):
        self.filters.update(kwargs)
        return self

    def first(self):
        if self.record and self.filters.get("id") == self.record.id:
            return self.record
        return None


class RecordDb:
    def __init__(self, record=None):
        self.record = record
        self.committed = False

    def query(self, *_args, **_kwargs):
        return RecordQuery(self.record)

    def commit(self):
        self.committed = True
