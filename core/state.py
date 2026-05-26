"""Data classes for PSO iteration records and results."""
from __future__ import annotations
import dataclasses
import numpy as np
from core.config import PSOConfig

@dataclasses.dataclass
class IterationRecord:
    iteration: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_position: np.ndarray
    elapsed_eval: float
    elapsed_update: float
    positions: np.ndarray | None = None

@dataclasses.dataclass
class PSOResult:
    best_position: np.ndarray
    best_fitness: float
    history: list[IterationRecord]
    total_time: float
    stop_reason: str
    config: PSOConfig
