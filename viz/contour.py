import matplotlib.pyplot as plt
import numpy as np
from viz.helpers import eval_grid

def plot_2d_contour(fn, bounds, positions=None, best=None, title="", outfile=None):
    x = np.linspace(*bounds[0], 200); y = np.linspace(*bounds[1], 200)
    X, Y = np.meshgrid(x, y); grid = np.column_stack([X.ravel(), Y.ravel()])
    Z = eval_grid(fn, grid).reshape(X.shape)
    fig, ax = plt.subplots(figsize=(8,7))
    ax.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.8)
    if positions is not None:
        ax.scatter(positions[:,0], positions[:,1], c="white", s=15, edgecolors="black", lw=0.5, zorder=5)
    if best is not None:
        ax.scatter([best[0]], [best[1]], c="red", s=100, marker="*", zorder=6)
    ax.set_title(title); fig.tight_layout()
    if outfile: fig.savefig(str(outfile), dpi=150)
    plt.close(fig); return fig
