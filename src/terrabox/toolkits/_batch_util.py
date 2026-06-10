"""auto_batch: make single-image tools accept EarthBench-style batch calls.

EarthBench batched a time series into ONE call by passing parallel LISTS of file
paths (and a list ``output_path``). Terrabox tools are single-image. This wrapper
detects the batch convention (``output_path`` is a list) and runs the wrapped
handler once per element, broadcasting non-list args, then returns the list of
results. When ``output_path`` is a single string it is a plain single call and
the wrapper is a transparent pass-through — so it is always safe to apply.
"""
from __future__ import annotations

from typing import Any, Callable, Dict


def auto_batch(handler: Callable) -> Callable:
    def _wrapped(arguments: Dict[str, Any], context: Any, account: Any):
        args = dict(arguments or {})
        out = args.get("output_path")
        # Batch convention: a list of output paths. Otherwise pass through.
        if not isinstance(out, list):
            return handler(args, context, account)
        n = len(out)
        if n == 0:
            return handler(args, context, account)
        # output_path is a list → batch. Iterate EVERY list-valued arg by index
        # (clamp to the last element if a list is shorter — the source data
        # occasionally has mismatched list lengths). This guarantees the wrapped
        # single-image handler never receives a list for a path argument.
        batch_keys = [k for k, v in args.items() if isinstance(v, list)]
        results = []
        outputs = []
        for i in range(n):
            sub = dict(args)
            for k in batch_keys:
                lst = args[k]
                sub[k] = lst[i] if i < len(lst) else (lst[-1] if lst else None)
            r = handler(sub, context, account)
            results.append(r)
            if isinstance(r, dict) and r.get("output_path"):
                outputs.append(r["output_path"])
        return {"status": "success", "batch": True, "count": n,
                "output_paths": outputs, "results": results}
    return _wrapped
