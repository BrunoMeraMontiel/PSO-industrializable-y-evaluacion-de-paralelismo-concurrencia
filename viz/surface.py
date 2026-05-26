import matplotlib.pyplot as plt
import numpy as np
from viz.helpers import eval_grid

def plot_3d_surface(fn, bounds, positions=None, title="", outfile=None):
    x = np.linspace(*bounds[0], 80); y = np.linspace(*bounds[1], 80)
    X, Y = np.meshgrid(x, y); grid = np.column_stack([X.ravel(), Y.ravel()])
    Z = eval_grid(fn, grid).reshape(X.shape)
    fig = plt.figure(figsize=(10,8)); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6, linewidth=0)
    if positions is not None:
        zp = eval_grid(fn, positions[:,:2])
        ax.scatter(positions[:,0], positions[:,1], zp, c="red", s=25, zorder=5)
    ax.set_title(title); fig.tight_layout()
    if outfile: fig.savefig(str(outfile), dpi=150)
    plt.close(fig); return fig
