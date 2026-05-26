"""
Case study: Portfolio optimisation using PSO.

Problem
-------
An investor wants to allocate capital across N assets to maximise the
risk-adjusted return (Sharpe ratio).  Each particle encodes a vector of
N weights in [0, 1]; the weights are normalised so they sum to 1 before
evaluation.

The objective function is  -Sharpe  (negated because PSO minimises):

    Sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

where:
    portfolio_return    = w^T · mu
    portfolio_volatility = sqrt(w^T · Cov · w)

This is a non-convex problem when additional constraints are present
(cardinality, min/max allocation), and even the unconstrained version
has a curved Pareto front that PSO explores well.

Dimension = N (number of assets), bounds = [0, 1] per weight.
"""
from __future__ import annotations

import numpy as np


def generate_market_data(
    n_assets: int = 10,
    n_days: int = 252,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic daily returns for n_assets over n_days.

    Returns
    -------
    expected_returns : (n_assets,) annualised mean returns
    cov_matrix : (n_assets, n_assets) annualised covariance
    """
    rng = np.random.default_rng(seed)
    # Simulate realistic annual returns (5%-25%) and volatilities (10%-40%)
    annual_mu = rng.uniform(0.05, 0.25, n_assets)
    daily_mu = annual_mu / 252

    # Random correlation structure
    A = rng.normal(0, 1, (n_assets, n_assets))
    corr = A @ A.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    daily_vol = rng.uniform(0.10, 0.40, n_assets) / np.sqrt(252)
    daily_cov = corr * np.outer(daily_vol, daily_vol)

    # Annualise
    expected_returns = annual_mu
    cov_matrix = daily_cov * 252

    return expected_returns, cov_matrix


def make_portfolio_objective(
    n_assets: int = 10,
    risk_free_rate: float = 0.02,
    seed: int = 42,
):
    """
    Build a PSO-compatible objective for portfolio optimisation.

    Parameters
    ----------
    n_assets : number of assets
    risk_free_rate : annual risk-free rate (e.g. 0.02 = 2%)
    seed : for reproducible market data

    Returns
    -------
    fn : callable f(x) -> float  (negative Sharpe ratio)
    bounds : list of (0, 1) per asset weight
    info : dict with expected_returns, cov_matrix, risk_free_rate
    """
    mu, cov = generate_market_data(n_assets, seed=seed)

    def objective(x: np.ndarray) -> float:
        # Normalise weights to sum to 1
        w = np.abs(x)
        s = w.sum()
        if s < 1e-12:
            return 1e6  # degenerate: all zeros
        w = w / s

        port_return = w @ mu
        port_vol = np.sqrt(w @ cov @ w)

        if port_vol < 1e-12:
            return 1e6  # degenerate volatility

        sharpe = (port_return - risk_free_rate) / port_vol
        return -sharpe  # negate because PSO minimises

    bounds = [(0.0, 1.0)] * n_assets
    info = {
        "expected_returns": mu,
        "cov_matrix": cov,
        "risk_free_rate": risk_free_rate,
        "n_assets": n_assets,
    }
    return objective, bounds, info


def decode_weights(x: np.ndarray) -> np.ndarray:
    """Normalise a raw PSO solution vector to valid portfolio weights."""
    w = np.abs(x)
    s = w.sum()
    if s < 1e-12:
        return np.ones_like(w) / len(w)
    return w / s


def portfolio_metrics(weights: np.ndarray, mu: np.ndarray,
                      cov: np.ndarray, rf: float = 0.02) -> dict:
    """Compute portfolio return, volatility, and Sharpe ratio."""
    ret = weights @ mu
    vol = np.sqrt(weights @ cov @ weights)
    sharpe = (ret - rf) / vol if vol > 1e-12 else 0.0
    return {"return": ret, "volatility": vol, "sharpe": sharpe, "weights": weights}
