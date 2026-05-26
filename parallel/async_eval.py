"""
V3 — Asyncio cooperative concurrency.

Designed for asymmetric evaluation scenarios where each fitness call
involves I/O latency (e.g. querying a simulation service, reading
from a database, network call).  We simulate this with asyncio.sleep.

For pure CPU-bound functions asyncio offers no benefit (single thread).
The point is to demonstrate real speedup when evaluation has I/O waits.
"""
from __future__ import annotations
import asyncio
import time
import numpy as np


def make_latent_fn(fn, latency: float = 0.01):
    """Wrap fn to simulate I/O latency per evaluation."""
    def wrapper(x):
        time.sleep(latency)
        return fn(x)
    wrapper.__name__ = f"{fn.__name__}_latent"
    return wrapper


class AsyncEvaluator:
    """
    V3 — Evaluate fitness asynchronously using asyncio.gather.

    Each fn(x) call is dispatched to a thread via run_in_executor,
    allowing concurrent I/O waits.  A semaphore limits concurrency.
    """
    def __init__(self, max_concurrency: int = 64):
        self.max_concurrency = max_concurrency

    def __call__(self, fn, positions):
        try:
            loop = asyncio.get_running_loop()
            # Already inside an event loop — use nest_asyncio pattern
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                results = list(pool.map(fn, positions))
            return np.array(results)
        except RuntimeError:
            return asyncio.run(self._run(fn, positions))

    async def _run(self, fn, positions):
        sem = asyncio.Semaphore(self.max_concurrency)
        loop = asyncio.get_event_loop()
        async def _one(x):
            async with sem:
                return await loop.run_in_executor(None, fn, x)
        results = await asyncio.gather(*[_one(x) for x in positions])
        return np.array(results)
