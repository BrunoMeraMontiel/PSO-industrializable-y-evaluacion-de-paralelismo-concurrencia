"""Save PSO results to disk (JSON + CSV + optional NPZ)."""
from __future__ import annotations
import csv, json, os, platform, sys, time
from pathlib import Path
from typing import Any
import numpy as np
from core.state import PSOResult

def _hw():
    return {"platform": platform.platform(), "python": sys.version,
            "cpu": platform.processor() or "unknown", "cpu_count": os.cpu_count()}

def save_result(result, directory, *, benchmark_name="unknown", strategy_name="unknown",
                extra_meta=None, save_trajectories=False):
    d = Path(directory); d.mkdir(parents=True, exist_ok=True)
    meta = {"benchmark": benchmark_name, "strategy": strategy_name,
            "config": result.config.to_dict(), "best_fitness": result.best_fitness,
            "best_position": result.best_position.tolist(), "total_time": result.total_time,
            "stop_reason": result.stop_reason, "n_iterations_actual": len(result.history),
            "hardware": _hw(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra_meta: meta.update(extra_meta)
    with open(d/"meta.json","w") as f: json.dump(meta, f, indent=2)
    with open(d/"history.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration","best_fitness","mean_fitness","std_fitness","elapsed_eval","elapsed_update"])
        for r in result.history:
            w.writerow([r.iteration,r.best_fitness,r.mean_fitness,r.std_fitness,r.elapsed_eval,r.elapsed_update])
    if save_trajectories:
        traj = {f"iter_{r.iteration}": r.positions for r in result.history if r.positions is not None}
        if traj: np.savez_compressed(d/"trajectories.npz", **traj)
    return d
