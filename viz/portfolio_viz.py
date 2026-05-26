"""Visualisation helpers for the portfolio case study."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_weights(weights: np.ndarray, asset_names: list[str] | None = None,
                 title: str = "Portfolio Allocation", outfile=None):
    """Pie chart of portfolio weights."""
    n = len(weights)
    labels = asset_names or [f"Asset {i+1}" for i in range(n)]
    # Filter out negligible weights for cleaner chart
    mask = weights > 0.01
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(weights[mask], labels=[labels[i] for i in range(n) if mask[i]],
           autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    fig.tight_layout()
    if outfile:
        fig.savefig(str(outfile), dpi=150)
    plt.close(fig)
    return fig


def plot_efficient_frontier(mu: np.ndarray, cov: np.ndarray,
                            optimal_weights: np.ndarray | None = None,
                            rf: float = 0.02, n_portfolios: int = 5000,
                            outfile=None):
    """
    Monte Carlo efficient frontier with the PSO-optimal portfolio marked.
    """
    rng = np.random.default_rng(0)
    n_assets = len(mu)
    rets, vols = [], []

    for _ in range(n_portfolios):
        w = rng.dirichlet(np.ones(n_assets))
        rets.append(w @ mu)
        vols.append(np.sqrt(w @ cov @ w))

    rets = np.array(rets)
    vols = np.array(vols)
    sharpes = (rets - rf) / vols

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(vols, rets, c=sharpes, cmap="viridis", s=5, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Sharpe Ratio")

    if optimal_weights is not None:
        opt_ret = optimal_weights @ mu
        opt_vol = np.sqrt(optimal_weights @ cov @ optimal_weights)
        ax.scatter([opt_vol], [opt_ret], c="red", s=200, marker="*",
                   zorder=5, label=f"PSO Optimal (Sharpe={((opt_ret-rf)/opt_vol):.3f})")
        ax.legend(fontsize=10)

    ax.set_xlabel("Volatility (annual)")
    ax.set_ylabel("Expected Return (annual)")
    ax.set_title("Efficient Frontier — PSO Portfolio Optimisation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if outfile:
        fig.savefig(str(outfile), dpi=150)
    plt.close(fig)
    return fig


def plot_risk_return_bar(weights, mu, cov, asset_names=None, outfile=None):
    """Bar chart showing each asset's weighted return and standalone risk contribution."""
    n = len(weights)
    labels = asset_names or [f"Asset {i+1}" for i in range(n)]
    contrib_ret  = weights * mu                           # weighted return (always >= 0)
    indiv_vol    = np.sqrt(np.diag(cov))                  # individual volatility per asset
    contrib_risk = weights * indiv_vol                    # standalone risk (always >= 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(n)

    ax1.bar(x, contrib_ret * 100, color="#4C72B0")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("Weighted Return Contribution (%)"); ax1.set_title("Return Decomposition by Asset")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    ax2.bar(x, contrib_risk * 100, color="#DD8452")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("Standalone Risk Contribution (%)"); ax2.set_title("Risk Decomposition by Asset")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    fig.tight_layout()
    if outfile:
        fig.savefig(str(outfile), dpi=150)
    plt.close(fig)
    return fig


def plot_sharpe_convergence(result, outfile=None):
    """Plot Sharpe ratio (positive) over iterations on linear scale."""
    iters  = [r.iteration   for r in result.history]
    sharpe = [-r.best_fitness for r in result.history]   # negate: PSO stores -Sharpe

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, sharpe, color="#1565c0", lw=1.8, label="PSO best Sharpe")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Portfolio Optimisation — Sharpe Ratio Convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if outfile:
        fig.savefig(str(outfile), dpi=150)
    plt.close(fig)
    return fig
