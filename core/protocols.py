"""Protocols (interfaces) that decouple the PSO loop from strategies."""
from __future__ import annotations
from typing import Protocol
import numpy as np

class ObjectiveFn(Protocol):
    def __call__(self, x: np.ndarray) -> float: ...

class Evaluator(Protocol):
    def __call__(self, fn: ObjectiveFn, positions: np.ndarray) -> np.ndarray: ...
