import matplotlib; matplotlib.use("Agg")
import numpy as np

def eval_grid(fn, grid):
    return np.array([fn(x) for x in grid])
