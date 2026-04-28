"""In-process runtime guards for agent runs and tool execution."""
from __future__ import annotations

import threading
from collections import defaultdict

from ..core.services.bulkhead import BulkheadManager
from ..core.services.circuit_breaker import SimpleCircuitBreaker


class AgentRuntime:
    def __init__(self):
        self._lock = threading.Lock()
        self._initialized = False
        self._run_slots = threading.BoundedSemaphore(64)
        self._run_capacity = 64
        self._active_runs = 0
        self._rejected_runs = 0
        self._metrics = defaultdict(int)
        self._bulkhead = BulkheadManager({})
        self._breakers: dict[str, SimpleCircuitBreaker] = {}
        self._breaker_threshold = 5
        self._breaker_recovery_s = 30.0
        self._breaker_overrides: dict[str, dict] = {}

    def configure(self, config) -> None:
        with self._lock:
            if self._initialized:
                return
            self._run_capacity = max(1, config.max_concurrent_agent_runs)
            self._run_slots = threading.BoundedSemaphore(self._run_capacity)
            capacities = {
                "default": config.default_tool_bulkhead,
                "perception": config.perception_tool_bulkhead,
                "compute": config.compute_tool_bulkhead,
                "network": config.network_tool_bulkhead,
                "risky": config.risky_tool_bulkhead,
            }
            self._bulkhead = BulkheadManager(capacities)
            self._breaker_threshold = config.breaker_failure_threshold
            self._breaker_recovery_s = config.breaker_recovery_timeout_s
            self._breaker_overrides = getattr(config, "circuit_breaker_overrides", {}) or {}
            self._initialized = True

    def _breaker(self, bucket: str) -> SimpleCircuitBreaker:
        with self._lock:
            if bucket not in self._breakers:
                override = self._breaker_overrides.get(bucket, {})
                self._breakers[bucket] = SimpleCircuitBreaker(
                    failure_threshold=override.get("threshold", self._breaker_threshold),
                    recovery_timeout_s=override.get("recovery_s", self._breaker_recovery_s),
                )
            return self._breakers[bucket]

    def acquire_run(self) -> bool:
        ok = self._run_slots.acquire(blocking=False)
        with self._lock:
            if ok:
                self._active_runs += 1
            else:
                self._rejected_runs += 1
        return ok

    def release_run(self) -> None:
        with self._lock:
            if self._active_runs > 0:
                self._active_runs -= 1
                self._run_slots.release()

    def start_tool(self, bucket: str) -> tuple[bool, str | None]:
        breaker = self._breaker(bucket)
        if not breaker.allow_request():
            with self._lock:
                self._metrics["breaker_rejections"] += 1
            return False, "circuit_open"
        if not self._bulkhead.acquire(bucket):
            with self._lock:
                self._metrics["bulkhead_rejections"] += 1
            return False, "bulkhead_full"
        with self._lock:
            self._metrics["tool_starts"] += 1
        return True, None

    def finish_tool(self, bucket: str, success: bool, error_type: str | None = None) -> None:
        self._bulkhead.release(bucket)
        breaker = self._breaker(bucket)
        if success:
            breaker.record_success()
            with self._lock:
                self._metrics["tool_successes"] += 1
        else:
            breaker.record_failure(error_type)
            with self._lock:
                self._metrics["tool_failures"] += 1

    def metrics(self) -> dict:
        with self._lock:
            base = {
                "run_capacity": self._run_capacity,
                "active_runs": self._active_runs,
                "rejected_runs": self._rejected_runs,
                **dict(self._metrics),
            }
        base["bulkheads"] = self._bulkhead.snapshot()
        base["breakers"] = {
            bucket: vars(breaker.snapshot(bucket))
            for bucket, breaker in list(self._breakers.items())
        }
        return base


_RUNTIME = AgentRuntime()


def get_runtime() -> AgentRuntime:
    return _RUNTIME
