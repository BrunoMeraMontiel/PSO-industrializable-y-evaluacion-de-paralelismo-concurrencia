"""analyse_results.py — Load results, generate comparison tables and plots."""
from __future__ import annotations
import argparse, csv, logging, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import persistence
from viz import plot_boxplots, plot_speedup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/benchmarks")
    p.add_argument("--outdir", default="results/analysis")
    args = p.parse_args()
    root, outdir = Path(args.results_dir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = [persistence.load_meta(mp.parent) for mp in root.rglob("meta.json")]
    if not records: logging.warning("No results in %s", root); return
    gf = defaultdict(lambda: defaultdict(list)); gt = defaultdict(lambda: defaultdict(list))
    for r in records:
        gf[r["benchmark"]][r["strategy"]].append(r["best_fitness"])
        gt[r["benchmark"]][r["strategy"]].append(r["total_time"])
    for bm, strats in gf.items():
        plot_boxplots(strats, title=f"Fitness -- {bm}", outfile=outdir/f"boxplot_{bm}.png")
    all_t = defaultdict(list)
    for strats in gt.values():
        for st, ts in strats.items(): all_t[st].extend(ts)
    plot_speedup({st: np.mean(ts) for st, ts in all_t.items()}, outfile=outdir/"speedup.png")
    print("\n" + "="*70 + "\nRESULTS SUMMARY\n" + "="*70)
    for bm in sorted(gf):
        print(f"\n-- {bm} --")
        for st in sorted(gf[bm]):
            print(f"  {st:25s} fitness={np.mean(gf[bm][st]):.4e} time={np.mean(gt[bm][st]):.3f}s")

if __name__ == "__main__": main()
