"""PSO engine — strategy-agnostic canonical loop."""
from __future__ import annotations
import logging, time
from typing import Sequence
import numpy as np
from core.boundaries import BOUNDARY_POLICIES
from core.config import PSOConfig
from core.protocols import Evaluator, ObjectiveFn
from core.state import IterationRecord, PSOResult
from core.topologies import TOPOLOGIES

logger = logging.getLogger(__name__)

def default_evaluator(fn: ObjectiveFn, positions: np.ndarray) -> np.ndarray:
    return np.array([fn(x) for x in positions])

def run_pso(fn, bounds, config=None, evaluator=None, store_positions=False):
    config = config or PSOConfig()
    evaluator = evaluator or default_evaluator
    bounds_arr = np.array(bounds, dtype=np.float64)
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
    d = len(bounds)
    rng = np.random.default_rng(config.seed)

    positions = rng.uniform(lo, hi, size=(config.n_particles, d))
    velocities = rng.uniform(-(hi-lo), (hi-lo), size=(config.n_particles, d))
    boundary_fn = BOUNDARY_POLICIES[config.boundary]
    topo_fn = TOPOLOGIES[config.topology]

    t0_eval = time.perf_counter()
    fitness = evaluator(fn, positions)
    t_eval = time.perf_counter() - t0_eval

    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()
    gi = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[gi].copy()
    gbest_fit = float(pbest_fit[gi])

    history, stag = [], 0
    stop_reason = "max_iterations"
    t_start = time.perf_counter()
    logger.info("PSO start dim=%d particles=%d iters=%d seed=%s", d, config.n_particles, config.n_iterations, config.seed)

    for it in range(config.n_iterations):
        t0_u = time.perf_counter()
        r1 = rng.random((config.n_particles, d))
        r2 = rng.random((config.n_particles, d))
        nbest = topo_fn(pbest_fit, pbest_pos)
        velocities = config.w*velocities + config.c1*r1*(pbest_pos-positions) + config.c2*r2*(nbest-positions)
        positions = boundary_fn(positions + velocities, lo, hi)
        t_upd = time.perf_counter() - t0_u

        t0_e = time.perf_counter()
        fitness = evaluator(fn, positions)
        t_ev = time.perf_counter() - t0_e

        imp = fitness < pbest_fit
        pbest_pos[imp] = positions[imp]
        pbest_fit[imp] = fitness[imp]
        ci = int(np.argmin(pbest_fit))
        cf = float(pbest_fit[ci])
        if cf < gbest_fit:
            gbest_fit, gbest_pos, stag = cf, pbest_pos[ci].copy(), 0
        else:
            stag += 1

        history.append(IterationRecord(it, gbest_fit, float(np.mean(fitness)), float(np.std(fitness)),
                                       gbest_pos.copy(), t_ev, t_upd,
                                       positions.copy() if store_positions else None))
        if it % max(1, config.n_iterations//10) == 0:
            logger.info("iter=%4d best=%.6e eval=%.4fs", it, gbest_fit, t_ev)
        if 0 <= gbest_fit <= config.tol:
            stop_reason = "tolerance"; break
        if stag >= config.stagnation:
            stop_reason = "stagnation"; break

    total = time.perf_counter() - t_start
    logger.info("PSO done reason=%s best=%.6e time=%.3fs", stop_reason, gbest_fit, total)
    return PSOResult(gbest_pos, gbest_fit, history, total, stop_reason, config)
