"""V1 — ThreadPoolExecutor. Limited by GIL for CPU-bound work."""
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class ThreadEvaluator:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1)+4)
    def __call__(self, fn, positions):
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return np.array(list(pool.map(fn, positions)))
