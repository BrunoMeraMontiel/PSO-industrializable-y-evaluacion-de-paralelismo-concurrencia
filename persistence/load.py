"""Load persisted PSO results."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

def load_meta(directory):
    with open(Path(directory)/"meta.json") as f: return json.load(f)

def load_history(directory):
    rows = []
    with open(Path(directory)/"history.csv") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if k != "iteration" else int(v) for k,v in row.items()})
    return rows

def load_trajectories(directory):
    p = Path(directory)/"trajectories.npz"
    if not p.exists(): return None
    d = np.load(p)
    return {int(k.split("_")[1]): d[k] for k in d.files}
