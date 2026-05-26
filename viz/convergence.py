import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

def plot_convergence(results, title="Convergence", outfile=None, log_scale=True):
    fig, ax = plt.subplots(figsize=(10,6))
    for label, res in results.items():
        ax.plot([r.iteration for r in res.history], [r.best_fitness for r in res.history], label=label, lw=1.5)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best Fitness")
    if log_scale: ax.set_yscale("log")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if outfile: fig.savefig(str(outfile), dpi=150); logger.info("Saved %s", outfile)
    plt.close(fig); return fig
