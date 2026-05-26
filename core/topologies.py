"""Neighbourhood topologies: global-best (star) and ring."""
from __future__ import annotations
import numpy as np

def global_best_topology(pbest_fit: np.ndarray, pbest_pos: np.ndarray) -> np.ndarray:
    return np.tile(pbest_pos[np.argmin(pbest_fit)], (len(pbest_fit), 1))

def ring_topology(pbest_fit: np.ndarray, pbest_pos: np.ndarray) -> np.ndarray:
    n = len(pbest_fit)
    nbest = np.empty_like(pbest_pos)
    for i in range(n):
        ns = [(i-1) % n, i, (i+1) % n]
        nbest[i] = pbest_pos[min(ns, key=lambda j: pbest_fit[j])]
    return nbest

TOPOLOGIES = {"global": global_best_topology, "ring": ring_topology}
