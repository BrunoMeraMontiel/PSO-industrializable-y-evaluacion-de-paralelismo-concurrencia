"""
run_extras.py — GIF animation + cProfile real profiling results.

Usage:
    python -m experiments.run_extras
"""
from __future__ import annotations
import cProfile, pstats, io, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import PSOConfig, run_pso
from objectives.functions import sphere, rastrigin
from objectives.suite import BENCHMARKS
from parallel import get_evaluator
from viz.animation import animate_2d

def _bounds(key: str):
    return BENCHMARKS[key].bounds

OUTDIR = Path("results/extras")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ── 1. GIFs ──────────────────────────────────────────────────────────────────

def make_gifs():
    configs = [
        ("sphere",    sphere,    _bounds("sphere_d2"),    "sphere_d2.gif"),
        ("rastrigin", rastrigin, _bounds("rastrigin_d2"), "rastrigin_d2.gif"),
    ]
    cfg = PSOConfig(n_particles=40, n_iterations=200, seed=42,
                    w=0.7298, c1=1.49618, c2=1.49618, stagnation=80)
    evaluator = get_evaluator("V0_sequential")

    for name, fn, bounds, fname in configs:
        print(f"Generating GIF: {fname} ...")
        result = run_pso(fn, bounds, cfg, evaluator=evaluator, store_positions=True)
        outpath = OUTDIR / fname
        animate_2d(fn, bounds, result, outfile=outpath, fps=12, every=2)
        print(f"  → {outpath}  ({outpath.stat().st_size // 1024} KB)")


# ── 2. cProfile ──────────────────────────────────────────────────────────────

def run_profile(strategy: str, fn_name: str, d: int, n_iter: int = 300) -> dict:
    from objectives.functions import sphere, rastrigin, ackley, rosenbrock

    fns = {"sphere": sphere, "rastrigin": rastrigin, "ackley": ackley, "rosenbrock": rosenbrock}
    fn  = fns[fn_name]
    bounds = _bounds(f"{fn_name}_d{d}")

    cfg = PSOConfig(n_particles=40, n_iterations=n_iter, seed=42,
                    w=0.7298, c1=1.49618, c2=1.49618, stagnation=50)
    evaluator = get_evaluator(strategy)

    pr = cProfile.Profile()
    pr.enable()
    run_pso(fn, bounds, cfg, evaluator=evaluator)
    pr.disable()

    s  = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    profile_text = s.getvalue()

    # extract top functions
    s2  = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(10)

    # total time
    total = sum(v[3] for v in pr.getstats())

    return {
        "strategy":     strategy,
        "function":     fn_name,
        "dim":          d,
        "total_time_s": round(total, 4),
        "profile_text": profile_text,
        "top_tottime":  s2.getvalue(),
    }


def make_profiles():
    cases = [
        ("V0_sequential", "sphere",    10),
        ("V1_threading",  "sphere",    10),
        ("V4_vectorised", "sphere",    10),
        ("V0_sequential", "rastrigin", 30),
        ("V4_vectorised", "rastrigin", 30),
    ]
    results = []
    for strategy, fn_name, d in cases:
        print(f"Profiling {strategy} / {fn_name} d={d} ...")
        r = run_profile(strategy, fn_name, d)
        results.append(r)
        print(f"  total={r['total_time_s']:.4f}s")

    out = OUTDIR / "cprofile_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved → {out}")

    # human-readable summary
    txt_out = OUTDIR / "cprofile_summary.txt"
    lines = []
    for r in results:
        lines.append(f"\n{'='*70}")
        lines.append(f"Strategy={r['strategy']}  fn={r['function']}  d={r['dim']}  total={r['total_time_s']}s")
        lines.append(r["profile_text"])
    txt_out.write_text("\n".join(lines))
    print(f"Saved → {txt_out}")

    return results


if __name__ == "__main__":
    make_gifs()
    make_profiles()
    print("\nDone — results in results/extras/")
