"""
run_portfolio_case.py — Case study: PSO portfolio optimisation.

Finds the portfolio weights that maximise the Sharpe ratio across
N assets using Particle Swarm Optimisation.

Usage:
    python -m experiments.run_portfolio_case
    python -m experiments.run_portfolio_case --n-assets 15 --strategy V4_vectorised
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import PSOConfig, run_pso
from objectives.portfolio import (
    make_portfolio_objective, decode_weights, portfolio_metrics,
)
from parallel import get_evaluator
from viz.portfolio_viz import (plot_weights, plot_efficient_frontier,
                               plot_risk_return_bar, plot_sharpe_convergence)
import persistence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("portfolio_case")


def main():
    p = argparse.ArgumentParser(description="PSO portfolio optimisation")
    p.add_argument("--n-assets", type=int, default=10)
    p.add_argument("--risk-free-rate", type=float, default=0.02)
    p.add_argument("--n-particles", type=int, default=50)
    p.add_argument("--n-iterations", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategy", default="V0_sequential")
    p.add_argument("--outdir", default="results/portfolio_case")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Building portfolio objective: %d assets, rf=%.2f%%",
                args.n_assets, args.risk_free_rate * 100)
    fn, bounds, info = make_portfolio_objective(
        n_assets=args.n_assets,
        risk_free_rate=args.risk_free_rate,
        seed=args.seed,
    )

    config = PSOConfig(
        n_particles=args.n_particles,
        n_iterations=args.n_iterations,
        seed=args.seed,
        w=0.7298, c1=1.49618, c2=1.49618,
        stagnation=100,
    )

    evaluator = get_evaluator(args.strategy)

    logger.info("Running PSO: %d particles, %d iters, strategy=%s",
                config.n_particles, config.n_iterations, args.strategy)
    result = run_pso(fn, bounds, config, evaluator=evaluator)

    # Decode solution
    weights = decode_weights(result.best_position)
    metrics = portfolio_metrics(
        weights, info["expected_returns"], info["cov_matrix"], info["risk_free_rate"]
    )

    logger.info("=== RESULTS ===")
    logger.info("Sharpe Ratio:  %.4f", metrics["sharpe"])
    logger.info("Annual Return: %.2f%%", metrics["return"] * 100)
    logger.info("Annual Vol:    %.2f%%", metrics["volatility"] * 100)
    logger.info("Portfolio weights:")
    for i, w in enumerate(weights):
        if w > 0.01:
            logger.info("  Asset %2d: %.1f%%", i + 1, w * 100)

    # Persist
    persistence.save_result(
        result, outdir / "pso_result",
        benchmark_name=f"portfolio_{args.n_assets}assets",
        strategy_name=args.strategy,
        extra_meta={
            "n_assets": args.n_assets,
            "risk_free_rate": args.risk_free_rate,
            "sharpe_ratio": metrics["sharpe"],
            "annual_return": metrics["return"],
            "annual_volatility": metrics["volatility"],
            "weights": weights.tolist(),
        },
    )

    # Visualisations
    plot_sharpe_convergence(result, outfile=outdir / "convergence.png")

    plot_weights(weights, title=f"Optimal Portfolio ({args.n_assets} assets)",
                 outfile=outdir / "weights.png")

    plot_efficient_frontier(
        info["expected_returns"], info["cov_matrix"],
        optimal_weights=weights, rf=info["risk_free_rate"],
        outfile=outdir / "efficient_frontier.png",
    )

    plot_risk_return_bar(
        weights, info["expected_returns"], info["cov_matrix"],
        outfile=outdir / "risk_return.png",
    )

    logger.info("Results saved to %s/", outdir)


if __name__ == "__main__":
    main()
