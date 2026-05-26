"""make_viz.py — Generate plots and animations."""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import PSOConfig, run_pso
from objectives import BENCHMARKS
from parallel import get_evaluator
from viz import animate_2d, plot_2d_contour, plot_3d_surface, plot_convergence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="sphere_d2")
    p.add_argument("--strategy", default="V0_sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--animate", action="store_true")
    p.add_argument("--outdir", default="results/viz")
    args = p.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    bench = BENCHMARKS[args.benchmark]
    result = run_pso(bench.fn, bench.bounds, PSOConfig(seed=args.seed, n_iterations=200, n_particles=30),
                     evaluator=get_evaluator(args.strategy), store_positions=(bench.dim <= 3))
    plot_convergence({args.strategy: result}, title=f"Convergence -- {bench.name}",
                     outfile=outdir/f"convergence_{bench.name}.png")
    if bench.dim == 2:
        last = result.history[-1].positions if result.history[-1].positions is not None else None
        plot_2d_contour(bench.fn, bench.bounds, positions=last, best=result.best_position,
                        title=f"{bench.name}", outfile=outdir/f"contour_{bench.name}.png")
        plot_3d_surface(bench.fn, bench.bounds, positions=last, title=bench.name,
                        outfile=outdir/f"surface_{bench.name}.png")
        if args.animate:
            animate_2d(bench.fn, bench.bounds, result, outfile=outdir/f"anim_{bench.name}.gif", fps=8, every=2)

if __name__ == "__main__": main()
