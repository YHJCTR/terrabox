"""Small thread-safe bulkhead manager for isolating tool groups."""
from __future__ import annotations

import threading
from collections import defaultdict


class _BulkheadState:
    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self.semaphore = threading.BoundedSemaphore(self.capacity)
        self.active = 0
        self.rejections = 0
        self.lock = threading.Lock()


class BulkheadManager:
    def __init__(self, capacities: dict[str, int] | None = None):
        self._capacities = capacities or {}
        self._states: dict[str, _BulkheadState] = {}
        self._lock = threading.Lock()

    def _state_for(self, bucket: str) -> _BulkheadState:
        with self._lock:
            if bucket not in self._states:
                self._states[bucket] = _BulkheadState(self._capacities.get(bucket, 8))
            return self._states[bucket]

    def acquire(self, bucket: str) -> bool:
        state = self._state_for(bucket)
        ok = state.semaphore.acquire(blocking=False)
        with state.lock:
            if ok:
                state.active += 1
            else:
                state.rejections += 1
        return ok

    def release(self, bucket: str) -> None:
        state = self._state_for(bucket)
        with state.lock:
            if state.active > 0:
                state.active -= 1
                state.semaphore.release()

    def snapshot(self) -> dict[str, dict[str, int]]:
        data: dict[str, dict[str, int]] = {}
        with self._lock:
            items = list(self._states.items())
        for bucket, state in items:
            with state.lock:
                data[bucket] = {
                    "capacity": state.capacity,
                    "active": state.active,
                    "rejections": state.rejections,
                }
        return data
