"""V0 — Sequential evaluation (baseline)."""
import numpy as np

def sequential_evaluator(fn, positions):
    return np.array([fn(x) for x in positions])
