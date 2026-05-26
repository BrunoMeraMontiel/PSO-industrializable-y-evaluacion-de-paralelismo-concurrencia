"""
V4 — NumPy vectorised evaluation (implicit SIMD parallelism).

Instead of looping fn(x) per particle, uses fn_vec(X) which operates
on the full (n, d) matrix via NumPy broadcasting.  Zero IPC overhead,
best cache locality, typically fastest for analytical functions.
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VectorisedEvaluator:
    """
    V4 — Evaluate using a vectorised batch function.

    Falls back to sequential if no fn_vec is provided.
    """
    def __init__(self, fn_vec=None):
        self.fn_vec = fn_vec

    def __call__(self, fn, positions):
        if self.fn_vec is not None:
            return self.fn_vec(positions)
        logger.warning("VectorisedEvaluator: no fn_vec, falling back to sequential")
        return np.array([fn(x) for x in positions])
