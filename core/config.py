"""Immutable PSO configuration."""
from __future__ import annotations
import dataclasses

@dataclasses.dataclass(frozen=True)
class PSOConfig:
    n_particles: int = 30
    n_iterations: int = 200
    w: float = 0.7298
    c1: float = 1.49618
    c2: float = 1.49618
    topology: str = "global"
    boundary: str = "clamp"
    tol: float = 1e-12
    stagnation: int = 50
    seed: int | None = None
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
