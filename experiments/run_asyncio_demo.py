"""
V3 Asyncio Demo — asymmetric evaluation with simulated I/O latency.

Compares V0 (sequential) vs V1 (threads) vs V3 (asyncio) on sphere_d10
where each evaluation incurs an artificial I/O delay, mimicking a scenario
where fitness queries hit a remote service (database, simulation API, etc.).

Usage:
    python -m experiments.run_asyncio_demo [--latency 0.01] [--n-particles 40]
"""
from __future__ import annotations
import argparse, json, logging, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from core.config import PSOConfig
from core.engine import run_pso
from objectives.functions import sphere
from parallel.sequential import sequential_evaluator
from parallel.threading_eval import ThreadEvaluator
from parallel.async_eval import AsyncEvaluator, make_latent_fn

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

BOUNDS = [(-5.0, 5.0)] * 10


def run_one(strategy_name, evaluator, latent_fn, n_particles, n_iterations, seed):
    cfg = PSOConfig(n_particles=n_particles, n_iterations=n_iterations, seed=seed,
                    stagnation=n_iterations)  # don't stop early
    t0 = time.perf_counter()
    result = run_pso(latent_fn, BOUNDS, config=cfg, evaluator=evaluator)
    wall = time.perf_counter() - t0
    return {"strategy": strategy_name, "wall_time": wall,
            "n_iters": len(result.history), "best_fitness": result.best_fitness}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency", type=float, default=0.005,
                        help="Simulated I/O latency per evaluation (seconds)")
    parser.add_argument("--n-particles", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--outdir", default="results/asyncio_demo")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    latent_fn = make_latent_fn(sphere, latency=args.latency)

    strategies = [
        ("V0_sequential", lambda fn, pos: sequential_evaluator(fn, pos)),
        ("V1_threading",  ThreadEvaluator(max_workers=args.n_particles)),
        ("V3_asyncio",    AsyncEvaluator(max_concurrency=args.n_particles)),
    ]

    print(f"\nAsyncio Demo — latency={args.latency*1000:.1f}ms/eval, "
          f"particles={args.n_particles}, iters={args.n_iterations}")
    print(f"Expected seq time/iter ≈ {args.latency*1000*args.n_particles:.0f}ms\n")

    records = []
    for seed in args.seeds:
        for name, ev in strategies:
            r = run_one(name, ev, latent_fn, args.n_particles, args.n_iterations, seed)
            r["seed"] = seed
            records.append(r)
            print(f"  {name:20s} seed={seed}  wall={r['wall_time']:.3f}s  "
                  f"best={r['best_fitness']:.4e}")

    # Aggregate by strategy
    from collections import defaultdict
    times = defaultdict(list)
    for r in records:
        times[r["strategy"]].append(r["wall_time"])
    mean_times = {k: np.mean(v) for k, v in times.items()}
    baseline = mean_times["V0_sequential"]

    print(f"\n{'Strategy':<22} {'Mean time':>10} {'Speedup':>8}")
    print("-" * 42)
    for name in ["V0_sequential", "V1_threading", "V3_asyncio"]:
        t = mean_times[name]
        sp = baseline / t
        print(f"  {name:<20} {t:>10.3f}s {sp:>7.2f}x")

    # Save JSON summary
    summary = {
        "latency_ms": args.latency * 1000,
        "n_particles": args.n_particles,
        "n_iterations": args.n_iterations,
        "seeds": args.seeds,
        "mean_times": mean_times,
        "speedups": {k: baseline / v for k, v in mean_times.items()},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Bar chart: speedup
    names = list(mean_times.keys())
    speedups = [baseline / mean_times[n] for n in names]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, speedups, color=colors, width=0.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="baseline (1×)")
    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{sp:.2f}×", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Speedup vs V0 sequential")
    ax.set_title(f"Asyncio concurrency speedup under I/O latency\n"
                 f"({args.latency*1000:.1f}ms/eval, {args.n_particles} particles)")
    ax.set_ylim(0, max(speedups) * 1.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "speedup_bar.png", dpi=150)
    plt.close(fig)

    # Line chart: wall time across seeds
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in zip(["V0_sequential", "V1_threading", "V3_asyncio"], colors):
        seed_times = [r["wall_time"] for r in records if r["strategy"] == name]
        ax.plot(args.seeds, seed_times, marker="o", label=name, color=color)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall time per seed by strategy (I/O latency scenario)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "wall_time_per_seed.png", dpi=150)
    plt.close(fig)

    print(f"\nResults saved to {outdir}/")


if __name__ == "__main__":
    main()
