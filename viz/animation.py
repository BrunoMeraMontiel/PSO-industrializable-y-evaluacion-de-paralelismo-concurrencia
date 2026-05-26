import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np, logging
from viz.helpers import eval_grid
logger = logging.getLogger(__name__)

def animate_2d(fn, bounds, result, outfile="swarm.gif", fps=10, every=1):
    frames = [r for r in result.history if r.positions is not None][::every]
    if not frames: logger.warning("No positions to animate"); return
    x = np.linspace(*bounds[0],150); y = np.linspace(*bounds[1],150)
    X, Y = np.meshgrid(x, y); Z = eval_grid(fn, np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    fig, ax = plt.subplots(figsize=(7,6)); ax.contourf(X, Y, Z, 40, cmap="viridis", alpha=.75)
    scat = ax.scatter([],[], c="white", s=18, edgecolors="black", lw=.4)
    star = ax.scatter([],[], c="red", s=120, marker="*", zorder=6); ttl = ax.set_title("")
    def upd(i):
        r = frames[i]; scat.set_offsets(r.positions[:,:2]); star.set_offsets([r.best_position[:2]])
        ttl.set_text(f"iter {r.iteration}  best={r.best_fitness:.4e}"); return scat, star
    ani = animation.FuncAnimation(fig, upd, frames=len(frames), blit=False, interval=1000//fps)
    ani.save(str(outfile), writer="pillow", fps=fps); logger.info("Saved %s", outfile); plt.close(fig)
