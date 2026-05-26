"""V2 — ProcessPoolExecutor with batching to amortise IPC."""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def _eval_batch(args):
    fn, batch = args
    return [fn(x) for x in batch]

class ProcessEvaluator:
    def __init__(self, max_workers=None, batch_size=None):
        self.max_workers = max_workers or os.cpu_count() or 1
        self.batch_size = batch_size
    def __call__(self, fn, positions):
        n = len(positions)
        bs = self.batch_size or max(1, n // self.max_workers)
        batches = [positions[i:i+bs] for i in range(0, n, bs)]
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            results = pool.map(_eval_batch, [(fn, b) for b in batches])
        flat = []
        for r in results: flat.extend(r)
        return np.array(flat)
