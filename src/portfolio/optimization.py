"""
Portfolio Optimization
======================
Implements three portfolio strategies:
  1. Equal Weight (1/N)
  2. Global Minimum Variance (GMV)
  3. Maximum Sharpe Ratio (MSR)

All use Ledoit-Wolf covariance shrinkage for stability.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


# ─────────────────────────────────────────────
# Covariance Estimation
# ─────────────────────────────────────────────

def ledoit_wolf_covariance(returns_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Ledoit-Wolf shrinkage covariance from a (T x N) returns matrix.

    Args:
        returns_matrix: (T x N) array of daily returns

    Returns:
        (N x N) shrunk covariance matrix
    """
    lw = LedoitWolf()
    lw.fit(returns_matrix)
    return lw.covariance_


# ─────────────────────────────────────────────
# Portfolio Strategies
# ─────────────────────────────────────────────

def equal_weight(n_assets: int) -> np.ndarray:
    """Naive 1/N portfolio."""
    return np.ones(n_assets) / n_assets


def global_minimum_variance(cov: np.ndarray) -> np.ndarray:
    """
    Minimum variance portfolio (ignores expected returns).

    min  w^T Σ w
    s.t. sum(w) = 1, w >= 0
    """
    n = cov.shape[0]
    w0 = np.ones(n) / n

    def portfolio_variance(w):
        return w @ cov @ w

    def grad_variance(w):
        return 2 * cov @ w

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n

    result = minimize(
        portfolio_variance,
        x0=w0,
        jac=grad_variance,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    return result.x if result.success else w0


def maximum_sharpe_ratio(mu: np.ndarray, cov: np.ndarray,
                          rf: float = 0.04 / 252) -> np.ndarray:
    """
    Maximum Sharpe Ratio portfolio.

    max  (w^T μ - rf) / sqrt(w^T Σ w)
    s.t. sum(w) = 1, w >= 0

    Args:
        mu: (N,) expected returns vector
        cov: (N, N) covariance matrix
        rf: Daily risk-free rate (default: 4% annual / 252)

    Returns:
        (N,) optimal weights
    """
    n = cov.shape[0]
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w + 1e-12)
        return -(ret - rf) / vol

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n

    result = minimize(
        neg_sharpe,
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    return result.x if result.success else w0


# ─────────────────────────────────────────────
# Expected Returns from Forecasts
# ─────────────────────────────────────────────

def forecasts_to_expected_returns(current_prices: dict,
                                   predicted_prices: dict) -> np.ndarray:
    """
    Compute 1-day momentum signal from price forecasts.

    μ_i = (P̂_{t+1} - P_t) / P_t

    Args:
        current_prices: {ticker: current_close}
        predicted_prices: {ticker: predicted_next_close}

    Returns:
        (N,) array of expected returns (same order as sorted tickers)
    """
    tickers = sorted(current_prices.keys())
    mu = np.array([
        (predicted_prices[t] - current_prices[t]) / current_prices[t]
        for t in tickers
    ])
    return mu


# ─────────────────────────────────────────────
# Transaction Cost Calculation
# ─────────────────────────────────────────────

def transaction_costs(w_new: np.ndarray, w_old: np.ndarray,
                       portfolio_value: float,
                       cost_bps: float = 15.0) -> float:
    """
    Proportional transaction costs on turnover.

    Cost = sum(|w_new - w_old|) * portfolio_value * (bps / 10000)

    Args:
        w_new: New target weights
        w_old: Previous weights (drift-adjusted)
        portfolio_value: Current portfolio value
        cost_bps: One-way transaction cost in basis points

    Returns:
        Dollar transaction cost
    """
    turnover = np.sum(np.abs(w_new - w_old))
    return turnover * portfolio_value * (cost_bps / 10_000)
