"""run_benchmarks.py — Full benchmark suite across strategies."""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import PSOConfig, run_pso
from objectives import make_instances
from objectives.functions import VEC_REGISTRY
from parallel import get_evaluator
import persistence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s", datefmt="%H:%M:%S")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", nargs="+", type=int, default=[2,10,30])
    p.add_argument("--strategies", nargs="+", default=["V0_sequential","V1_threading","V4_vectorised"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42,123,7])
    p.add_argument("--n-particles", type=int, default=40)
    p.add_argument("--n-iterations", type=int, default=300)
    p.add_argument("--outdir", default="results/benchmarks")
    args = p.parse_args()
    instances = make_instances(tuple(args.dims))
    for inst in instances:
        for strat in args.strategies:
            for seed in args.seeds:
                cfg = PSOConfig(n_particles=args.n_particles, n_iterations=args.n_iterations, seed=seed)
                extra = {}
                if strat == "V4_vectorised": extra["fn_vec"] = VEC_REGISTRY.get(inst.fn)
                ev = get_evaluator(strat, **extra)
                logging.info(">> %s | %s | seed=%d", inst.name, strat, seed)
                result = run_pso(inst.fn, inst.bounds, cfg, evaluator=ev)
                persistence.save_result(result, Path(args.outdir)/f"{inst.name}/{strat}/seed_{seed}",
                                        benchmark_name=inst.name, strategy_name=strat)

if __name__ == "__main__": main()
