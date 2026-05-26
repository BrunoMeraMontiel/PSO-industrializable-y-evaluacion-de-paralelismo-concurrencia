"""run_grid_search.py — Grid search over PSO hyper-parameters."""
from __future__ import annotations
import argparse, csv, logging, sys
from itertools import product
from pathlib import Path
import yaml, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import PSOConfig, run_pso
from objectives import BENCHMARKS
from parallel import get_evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

DEFAULT = {"grid": {"w":[0.4,0.7298,0.9],"c1":[1.0,1.49618,2.0],"c2":[1.0,1.49618,2.0],
                     "n_particles":[30],"n_iterations":[300]},
           "seeds":[42,123,7,99,314], "benchmarks":["sphere_d2","rastrigin_d10","ackley_d30"],
           "strategies":["V0_sequential"]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None); p.add_argument("--outdir", default="results/grid_search")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config)) if args.config else DEFAULT
    grid = cfg["grid"]; pn = list(grid.keys()); combos = list(product(*[grid[k] for k in pn]))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bn in cfg["benchmarks"]:
        bench = BENCHMARKS[bn]
        for sn in cfg["strategies"]:
            for combo in combos:
                params = dict(zip(pn, combo))
                for seed in cfg["seeds"]:
                    pc = PSOConfig(n_particles=int(params.get("n_particles",30)),
                                   n_iterations=int(params.get("n_iterations",300)),
                                   w=float(params.get("w",0.7298)),c1=float(params.get("c1",1.49618)),
                                   c2=float(params.get("c2",1.49618)),seed=seed)
                    res = run_pso(bench.fn, bench.bounds, pc, evaluator=get_evaluator(sn))
                    rows.append({"benchmark":bn,"strategy":sn,"seed":seed,**{k:params[k] for k in pn},
                                 "best_fitness":res.best_fitness,"total_time":res.total_time,
                                 "stop_reason":res.stop_reason})
    with open(outdir/"summary.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    logging.info("Grid search done: %d rows -> %s", len(rows), outdir/"summary.csv")

if __name__ == "__main__": main()
