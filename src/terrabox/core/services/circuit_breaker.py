"""Lightweight in-process circuit breaker utilities for agent runtime guards."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class CircuitBreakerSnapshot:
    bucket: str
    state: str
    failures: int
    opened_at: float | None
    last_error_type: str | None


class SimpleCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: float = 30.0):
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout_s = max(1.0, recovery_timeout_s)
        self._state = "closed"
        self._failures = 0
        self._opened_at: float | None = None
        self._last_error_type: str | None = None
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                now = time.time()
                if self._opened_at and now - self._opened_at >= self.recovery_timeout_s:
                    self._state = "half_open"
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failures = 0
            self._opened_at = None
            self._last_error_type = None

    def record_failure(self, error_type: str | None = None) -> None:
        with self._lock:
            self._failures += 1
            self._last_error_type = error_type
            if self._state == "half_open" or self._failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.time()

    def snapshot(self, bucket: str) -> CircuitBreakerSnapshot:
        with self._lock:
            return CircuitBreakerSnapshot(
                bucket=bucket,
                state=self._state,
                failures=self._failures,
                opened_at=self._opened_at,
                last_error_type=self._last_error_type,
            )
