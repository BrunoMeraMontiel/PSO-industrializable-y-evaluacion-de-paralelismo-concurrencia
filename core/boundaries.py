"""Boundary handling policies: clamp (default) and reflect."""
from __future__ import annotations
import numpy as np

def clamp(positions: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(positions, lo, hi)

def reflect(positions: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    result = positions.copy()
    for dim in range(positions.shape[1]):
        while True:
            below = result[:, dim] < lo[dim]
            above = result[:, dim] > hi[dim]
            if not (below.any() or above.any()): break
            result[below, dim] = 2 * lo[dim] - result[below, dim]
            result[above, dim] = 2 * hi[dim] - result[above, dim]
    return result

BOUNDARY_POLICIES = {"clamp": clamp, "reflect": reflect}
