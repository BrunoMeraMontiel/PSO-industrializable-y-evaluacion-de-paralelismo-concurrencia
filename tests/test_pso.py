"""Unit tests — 15+ checks covering core, objectives, strategies V0-V4, I/O."""
from __future__ import annotations
import sys, tempfile, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import PSOConfig
from core.engine import run_pso
from core.boundaries import clamp, reflect
from objectives.functions import sphere, rosenbrock, rastrigin, ackley, sphere_vec, VEC_REGISTRY
from parallel.sequential import sequential_evaluator
from parallel.threading_eval import ThreadEvaluator
from parallel.multiprocessing_eval import ProcessEvaluator
from parallel.vectorised_eval import VectorisedEvaluator
import persistence

class TestObjectives:
    def test_sphere(self): assert abs(sphere(np.zeros(5))) < 1e-15
    def test_rosenbrock(self): assert abs(rosenbrock(np.ones(5))) < 1e-15
    def test_rastrigin(self): assert abs(rastrigin(np.zeros(5))) < 1e-15
    def test_ackley(self): assert abs(ackley(np.zeros(5))) < 1e-10
    def test_vec_match(self):
        X = np.random.default_rng(42).uniform(-5,5,(20,4))
        np.testing.assert_allclose(sphere_vec(X), [sphere(x) for x in X])

class TestBoundaries:
    def test_clamp(self):
        assert np.array_equal(clamp(np.array([[-10.,6],[3,-7]]), np.array([-5.,-5]), np.array([5.,5])), [[-5,5],[3,-5]])
    def test_reflect(self):
        r = reflect(np.array([[-10.,12]]), np.array([-5.,-5]), np.array([5.,5]))
        assert np.all(r >= -5) and np.all(r <= 5)

class TestPSOCore:
    def test_reproducibility(self):
        cfg = PSOConfig(n_particles=20, n_iterations=50, seed=12345)
        r1 = run_pso(sphere, [(-5,5)]*3, cfg); r2 = run_pso(sphere, [(-5,5)]*3, cfg)
        assert abs(r1.best_fitness - r2.best_fitness) < 1e-15
    def test_monotonic(self):
        res = run_pso(sphere, [(-5,5)]*5, PSOConfig(n_particles=30, n_iterations=100, seed=42))
        fits = [r.best_fitness for r in res.history]
        assert all(fits[i] <= fits[i-1]+1e-15 for i in range(1, len(fits)))
    def test_converges(self):
        assert run_pso(sphere, [(-5,5)]*2, PSOConfig(n_particles=40, n_iterations=500, seed=42)).best_fitness < 1e-4
    def test_early_stop(self):
        res = run_pso(sphere, [(-5,5)]*2, PSOConfig(n_particles=20, n_iterations=1000, seed=42, stagnation=20, tol=1e-20))
        assert len(res.history) < 1000

class TestStrategies:
    def test_v0_v1(self):
        cfg = PSOConfig(n_particles=15, n_iterations=30, seed=99)
        r0 = run_pso(sphere, [(-5,5)]*3, cfg, evaluator=sequential_evaluator)
        r1 = run_pso(sphere, [(-5,5)]*3, cfg, evaluator=ThreadEvaluator(max_workers=2))
        assert abs(r0.best_fitness - r1.best_fitness) < 1e-10
    def test_v0_v4(self):
        cfg = PSOConfig(n_particles=15, n_iterations=30, seed=99)
        r0 = run_pso(sphere, [(-5,5)]*3, cfg, evaluator=sequential_evaluator)
        r4 = run_pso(sphere, [(-5,5)]*3, cfg, evaluator=VectorisedEvaluator(fn_vec=sphere_vec))
        assert abs(r0.best_fitness - r4.best_fitness) < 1e-10
    def test_v2_runs(self):
        assert run_pso(sphere, [(-5,5)]*3, PSOConfig(n_particles=15, n_iterations=30, seed=99),
                       evaluator=ProcessEvaluator(max_workers=2)).best_fitness < 100

class TestPortfolioCase:
    def test_portfolio_objective(self):
        from objectives.portfolio import make_portfolio_objective, decode_weights, portfolio_metrics
        fn, bounds, info = make_portfolio_objective(n_assets=5, seed=42)
        x = np.random.default_rng(0).uniform(0, 1, size=5)
        assert fn(x) < 0  # negative Sharpe (good portfolio)
        assert len(bounds) == 5
        w = decode_weights(x)
        assert abs(w.sum() - 1.0) < 1e-10
        m = portfolio_metrics(w, info["expected_returns"], info["cov_matrix"])
        assert m["sharpe"] > 0

class TestIO:
    def test_roundtrip(self):
        res = run_pso(sphere, [(-5,5)]*2, PSOConfig(n_particles=10, n_iterations=20, seed=1))
        with tempfile.TemporaryDirectory() as td:
            persistence.save_result(res, os.path.join(td,"t"), benchmark_name="sphere_d2", strategy_name="V0")
            meta = persistence.load_meta(os.path.join(td,"t"))
            assert meta["benchmark"] == "sphere_d2"
            assert abs(meta["best_fitness"] - res.best_fitness) < 1e-15
            assert len(persistence.load_history(os.path.join(td,"t"))) == len(res.history)
