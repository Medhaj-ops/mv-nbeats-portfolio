"""
Walk-Forward Backtesting Engine
================================
Implements expanding-window quarterly rebalancing backtest.

Protocol:
  - Initial training window: Jan 2010 – Dec 2020 (3+ years)
  - Rebalancing: Quarterly (Mar 31, Jun 30, Sep 30, Dec 31)
  - At each date: retrain all models → forecast → optimize → hold ~60 days
  - Transaction costs: 15 bps on turnover
  - No look-ahead bias: scaler fitted on training data only
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.portfolio.optimization import (
    ledoit_wolf_covariance,
    equal_weight,
    global_minimum_variance,
    maximum_sharpe_ratio,
    forecasts_to_expected_returns,
    transaction_costs
)

TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA',
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'MA', 'HD', 'DIS',
    'BAC', 'NFLX', 'ADBE', 'CRM', 'CSCO', 'PEP', 'INTC',
    'CMCSA', 'PFE', 'ABT', 'TMO'
]

REBALANCE_DATES = pd.date_range(start='2021-01-01', end='2021-12-31', freq='QE')
TRANSACTION_COST_BPS = 15
RISK_FREE_ANNUAL = 0.04
COV_LOOKBACK_DAYS = 252
INITIAL_CAPITAL = 100_000


def get_quarterly_rebalance_dates(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq='QE')


def compute_portfolio_metrics(returns: pd.Series,
                               rf_daily: float = RISK_FREE_ANNUAL / 252) -> dict:
    """Compute annualized performance metrics from daily return series."""
    ann_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - RISK_FREE_ANNUAL) / (ann_vol + 1e-12)

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        'Ann_Return': ann_return,
        'Ann_Vol': ann_vol,
        'Sharpe': sharpe,
        'Max_DD': max_dd
    }


def run_backtest(data_df: pd.DataFrame,
                 model_type: str = 'nbeats',
                 strategy: str = 'max_sharpe',
                 rebalance_dates=None,
                 initial_capital: float = INITIAL_CAPITAL,
                 verbose: bool = True) -> dict:
    """
    Main walk-forward backtest loop.

    Args:
        data_df: Full preprocessed dataset (long format)
        model_type: 'nbeats', 'lstm', 'gru', or 'arima'
        strategy: 'equal_weight', 'min_variance', or 'max_sharpe'
        rebalance_dates: pd.DatetimeIndex of rebalancing dates
        initial_capital: Starting portfolio value
        verbose: Print progress

    Returns:
        dict with 'equity_curve', 'weights_history', 'metrics', 'returns'
    """
    if rebalance_dates is None:
        rebalance_dates = get_quarterly_rebalance_dates('2021-01-01', '2021-12-31')

    tickers = sorted(TICKERS)
    n = len(tickers)
    portfolio_value = initial_capital
    weights = np.ones(n) / n  # Start equal weight

    equity_curve = []
    weights_history = []
    all_returns = []

    for i, rebal_date in enumerate(rebalance_dates):
        if verbose:
            print(f"\n[{i+1}/{len(rebalance_dates)}] Rebalancing: {rebal_date.date()}")

        # ── Training data: all available up to rebal_date ──────────────
        train_data = data_df[data_df['Date'] <= rebal_date].copy()

        # ── Forecast next-day prices ────────────────────────────────────
        current_prices = {}
        predicted_prices = {}

        for ticker in tickers:
            ticker_train = train_data[train_data['Ticker'] == ticker].sort_values('Date')
            if len(ticker_train) < 60:
                current_prices[ticker] = 100.0
                predicted_prices[ticker] = 100.0
                continue

            current_prices[ticker] = ticker_train['Close'].iloc[-1]

            if model_type == 'nbeats':
                pred = _forecast_nbeats(ticker_train, ticker)
            elif model_type == 'lstm':
                pred = _forecast_lstm(ticker_train, ticker)
            elif model_type == 'gru':
                pred = _forecast_gru(ticker_train, ticker)
            elif model_type == 'arima':
                pred = _forecast_arima(ticker_train, ticker)
            else:
                raise ValueError(f"Unknown model: {model_type}")

            predicted_prices[ticker] = pred

        # ── Expected returns ────────────────────────────────────────────
        mu = forecasts_to_expected_returns(current_prices, predicted_prices)

        # ── Covariance (trailing 252 days) ──────────────────────────────
        cov_data = train_data[train_data['Date'] > train_data['Date'].max() - pd.Timedelta(days=365)]
        returns_pivot = cov_data.pivot(index='Date', columns='Ticker', values='Daily_Return')
        returns_pivot = returns_pivot[tickers].dropna()
        cov = ledoit_wolf_covariance(returns_pivot.values)

        # ── Optimize weights ────────────────────────────────────────────
        if strategy == 'equal_weight':
            new_weights = equal_weight(n)
        elif strategy == 'min_variance':
            new_weights = global_minimum_variance(cov)
        elif strategy == 'max_sharpe':
            new_weights = maximum_sharpe_ratio(mu, cov)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # ── Transaction costs ───────────────────────────────────────────
        tc = transaction_costs(new_weights, weights, portfolio_value, TRANSACTION_COST_BPS)
        portfolio_value -= tc
        weights = new_weights

        if verbose:
            print(f"  Transaction cost: ${tc:.2f} | Portfolio: ${portfolio_value:,.0f}")
            top3 = sorted(zip(tickers, weights), key=lambda x: -x[1])[:3]
            print(f"  Top holdings: {[(t, f'{w:.1%}') for t, w in top3]}")

        # ── Simulate holding period ─────────────────────────────────────
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else rebal_date + pd.DateOffset(months=3)
        holding_data = data_df[
            (data_df['Date'] > rebal_date) & (data_df['Date'] <= next_date)
        ]
        holding_dates = sorted(holding_data['Date'].unique())

        for date in holding_dates:
            day_returns = holding_data[holding_data['Date'] == date].set_index('Ticker')['Daily_Return']
            r_vec = np.array([day_returns.get(t, 0.0) for t in tickers])
            port_return = weights @ r_vec
            portfolio_value *= (1 + port_return)
            all_returns.append({'Date': date, 'Return': port_return})
            equity_curve.append({'Date': date, 'Value': portfolio_value})

        # Drift-adjust weights
        price_changes = np.array([
            holding_data[holding_data['Ticker'] == t]['Close'].iloc[-1] /
            holding_data[holding_data['Ticker'] == t]['Close'].iloc[0]
            if len(holding_data[holding_data['Ticker'] == t]) > 0 else 1.0
            for t in tickers
        ])
        drifted = weights * price_changes
        weights = drifted / drifted.sum()

        weights_history.append({'Date': rebal_date, 'Weights': dict(zip(tickers, new_weights))})

    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    returns_df = pd.DataFrame(all_returns).set_index('Date')['Return']
    metrics = compute_portfolio_metrics(returns_df)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Model: {model_type.upper()} | Strategy: {strategy}")
        print(f"  Ann. Return : {metrics['Ann_Return']:.2%}")
        print(f"  Ann. Vol    : {metrics['Ann_Vol']:.2%}")
        print(f"  Sharpe      : {metrics['Sharpe']:.3f}")
        print(f"  Max DD      : {metrics['Max_DD']:.2%}")
        print(f"{'='*50}")

    return {
        'equity_curve': equity_df,
        'weights_history': weights_history,
        'metrics': metrics,
        'returns': returns_df
    }


# ─────────────────────────────────────────────
# Model dispatch stubs (import actual model wrappers)
# ─────────────────────────────────────────────

def _forecast_nbeats(ticker_train: pd.DataFrame, ticker: str) -> float:
    from src.models.nbeats import (
        MvNBeats, prepare_multivariate_datasets, train_nbeats,
        SELECTED_FEATURES, LOOKBACK_WINDOW
    )
    import torch
    datasets = prepare_multivariate_datasets(
        ticker_train[ticker_train['Ticker'] == ticker] if 'Ticker' in ticker_train.columns else ticker_train,
        lookback_window=LOOKBACK_WINDOW
    )
    if ticker not in datasets or len(datasets[ticker]['X_train']) < 30:
        return ticker_train['Close'].iloc[-1]
    _, preds, _ = train_nbeats(ticker, datasets[ticker])
    return float(preds[-1])


def _forecast_arima(ticker_train: pd.DataFrame, ticker: str) -> float:
    from statsmodels.tsa.arima.model import ARIMA
    close = ticker_train['Close'].values
    try:
        model = ARIMA(close, order=(2, 1, 2))
        fit = model.fit()
        return float(fit.forecast(steps=1)[0])
    except Exception:
        return float(close[-1])


def _forecast_lstm(ticker_train: pd.DataFrame, ticker: str) -> float:
    # Placeholder — see src/models/lstm_gru.py for full implementation
    return float(ticker_train['Close'].iloc[-1])


def _forecast_gru(ticker_train: pd.DataFrame, ticker: str) -> float:
    # Placeholder — see src/models/lstm_gru.py for full implementation
    return float(ticker_train['Close'].iloc[-1])
