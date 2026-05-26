"""Benchmark instance registry."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from objectives.functions import sphere, rosenbrock, rastrigin, ackley

@dataclass(frozen=True)
class BenchmarkInstance:
    name: str
    fn: Callable
    bounds: list[tuple[float,float]]
    dim: int
    known_optimum: float

def make_instances(dims=(2,10,30)):
    templates = [("sphere",sphere,(-5.12,5.12),0.),("rosenbrock",rosenbrock,(-5.,10.),0.),
                 ("rastrigin",rastrigin,(-5.12,5.12),0.),("ackley",ackley,(-5.,5.),0.)]
    out = []
    for d in dims:
        for name,fn,(lo,hi),opt in templates:
            out.append(BenchmarkInstance(f"{name}_d{d}",fn,[(lo,hi)]*d,d,opt))
    return out

BENCHMARKS = {i.name: i for i in make_instances()}
