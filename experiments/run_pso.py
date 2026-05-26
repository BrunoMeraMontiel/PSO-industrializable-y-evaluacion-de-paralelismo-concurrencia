"""run_pso.py — Single PSO execution with CLI."""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import PSOConfig, run_pso
from objectives import BENCHMARKS
from objectives.functions import VEC_REGISTRY
from parallel import get_evaluator
import persistence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s", datefmt="%H:%M:%S")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="sphere_d2")
    p.add_argument("--strategy", default="V0_sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-particles", type=int, default=30)
    p.add_argument("--n-iterations", type=int, default=200)
    p.add_argument("--w", type=float, default=0.7298)
    p.add_argument("--c1", type=float, default=1.49618)
    p.add_argument("--c2", type=float, default=1.49618)
    p.add_argument("--topology", default="global")
    p.add_argument("--outdir", default="results/single")
    p.add_argument("--save-trajectories", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    if args.config:
        with open(args.config) as f:
            for k,v in yaml.safe_load(f).items():
                if hasattr(args, k): setattr(args, k, v)
    bench = BENCHMARKS[args.benchmark]
    cfg = PSOConfig(n_particles=args.n_particles, n_iterations=args.n_iterations,
                    w=args.w, c1=args.c1, c2=args.c2, topology=args.topology, seed=args.seed)
    extra = {}
    if args.strategy == "V4_vectorised":
        extra["fn_vec"] = VEC_REGISTRY.get(bench.fn)
    ev = get_evaluator(args.strategy, **extra)
    result = run_pso(bench.fn, bench.bounds, cfg, evaluator=ev, store_positions=args.save_trajectories)
    tag = f"{bench.name}_{args.strategy}_s{args.seed}"
    persistence.save_result(result, Path(args.outdir)/tag, benchmark_name=bench.name, strategy_name=args.strategy,
                            save_trajectories=args.save_trajectories)

if __name__ == "__main__": main()
