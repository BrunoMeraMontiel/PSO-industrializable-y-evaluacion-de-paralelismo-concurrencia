import matplotlib.pyplot as plt

def plot_speedup(strategy_times, baseline_key="V0_sequential", outfile=None):
    base = strategy_times.get(baseline_key, 1.0)
    names = list(strategy_times.keys()); speedups = [base/strategy_times[k] for k in names]
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(names, speedups, color=plt.cm.Set2.colors[:len(names)])
    ax.axhline(1., color="gray", ls="--", lw=.8); ax.set_ylabel("Speedup vs baseline"); ax.set_title("Strategy Speedup")
    for b, s in zip(bars, speedups): ax.text(b.get_x()+b.get_width()/2, b.get_height()+.02, f"{s:.2f}x", ha="center", fontsize=9)
    fig.tight_layout()
    if outfile: fig.savefig(str(outfile), dpi=150)
    plt.close(fig); return fig

def plot_boxplots(data, ylabel="Final Fitness", title="Fitness Distribution", outfile=None):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.boxplot([data[k] for k in data], labels=list(data.keys()), patch_artist=True)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.tick_params(axis="x", rotation=30); fig.tight_layout()
    if outfile: fig.savefig(str(outfile), dpi=150)
    plt.close(fig); return fig
