# Mv-N-BEATS: Interpretable Multivariate Forecasting for Portfolio Optimization

> **N-BEATS for Portfolio Construction: Integrating Interpretable Forecasting with Mean-Variance Optimization**  
> Medhaj Dubey, M V Akshay Reddy, Gaurav Malkani — Manipal Institute of Technology

---

## Overview

This repository contains the full implementation of **Mv-N-BEATS** (Multivariate-adapted Neural Basis Expansion Analysis for Time Series), a novel architecture that extends the original N-BEATS model to handle multivariate financial data, and integrates it into a rigorous mean-variance portfolio optimization pipeline.

The core question this project answers: **does better stock price forecasting actually translate into better portfolio performance?** The answer, empirically validated across 27 large-cap U.S. equities from 2010–2021, is yes.

---

## Key Results

| Model | Strategy | Ann. Return | Ann. Vol | Sharpe Ratio | Max Drawdown |
|-------|----------|-------------|----------|--------------|--------------|
| **Mv-N-BEATS** | **Max Sharpe** | **37.21%** | **23.10%** | **1.71** | **-16.76%** |
| Mv-N-BEATS | Equal Weight | 30.20% | 15.54% | 1.671 | -16.32% |
| Mv-N-BEATS | Min Variance | 28.31% | 15.61% | 1.557 | -16.32% |
| GRU | Max Sharpe | 28.6% | 26.86% | 1.41 | -24.78% |
| LSTM | Max Sharpe | 31.34% | 27.0% | 1.34 | -21.8% |
| ARIMA | Equal Weight | 14.1% | 23.2% | 0.83 | -32.81% |

**Forecasting accuracy (average across all 27 stocks):**

| Model | RMSE ($) | MAE ($) | MAPE (%) | R² |
|-------|----------|---------|----------|-----|
| **Mv-N-BEATS** | **318.7** | **286.2** | **3.82** | **0.95** |
| GRU | 373.38 | 295.30 | 3.19 | 0.934 |
| LSTM | 769.81 | 640.16 | 6.19 | 0.771 |
| ARIMA | 2401.97 | 2068.21 | 21.24 | -1.036 |

---

## Architecture

### Mv-N-BEATS

The original N-BEATS model operates on univariate time series. Our key modification: instead of a 1D lookback window of length *L*, we construct a **multivariate input tensor** of shape *(L × D)* — 20 days × 10 features — flattened into a 200-dimensional vector. This preserves temporal structure while encoding cross-feature dependencies, while keeping the interpretable trend/seasonality basis functions unchanged.

```
Input: 200-dim vector (20 days × 10 features, flattened)
  ↓
Trend Stack (3 blocks) — Polynomial basis, degree 3
  ↓
Seasonality Stack (3 blocks) — Fourier basis, 10 harmonics
  ↓
Output: Next-day price forecast
```

Each block contains:
- 4 fully-connected layers (256 units, ReLU)
- Two theta-projection MLPs ([128, 128]) for backcast/forecast coefficients
- Basis function expansion into time-domain signals

### Portfolio Pipeline

```
Data Ingestion (Yahoo Finance, 27 stocks, 2010–2021)
  → Feature Engineering (23 candidates)
  → Feature Selection (XGBoost + Mutual Information → top 10)
  → Forecasting Engine (Mv-N-BEATS / LSTM / GRU / ARIMA)
  → Portfolio Optimization (1/N, GMV, MSR with Ledoit-Wolf covariance)
  → Walk-Forward Backtest (quarterly rebalancing, 15 bps transaction costs)
```

---

## Selected Features

| Feature | XGB Score | MI Score | Rationale |
|---------|-----------|----------|-----------|
| BB Percent | 0.142 | 0.089 | Mean reversion signals |
| RSI 14 | 0.128 | 0.082 | Momentum exhaustion |
| Volume | 0.115 | 0.078 | Conviction/liquidity |
| MACD | 0.108 | 0.074 | Trend strength |
| MACD Histogram | 0.102 | 0.071 | Momentum acceleration |
| Rolling Vol 20D | 0.098 | 0.068 | Time-varying risk |
| Momentum 10D | 0.095 | 0.065 | Short-term persistence |
| EMA 200 | 0.087 | 0.061 | Long-term trend |
| S&P500 Close | 0.081 | 0.058 | Systematic risk factor |
| VIX Close | 0.074 | 0.052 | Market sentiment/fear |

---

## Repository Structure

```
mv-nbeats-portfolio/
│
├── notebooks/
│   └── stock_prediction_full.ipynb     # Full end-to-end pipeline notebook
│
├── src/
│   ├── data_preprocessing.py           # Data ingestion & feature engineering
│   ├── feature_selection.py            # XGBoost + Mutual Information selection
│   ├── models/
│   │   ├── nbeats.py                   # Mv-N-BEATS architecture (PyTorch)
│   │   ├── lstm_gru.py                 # LSTM & GRU baselines (Keras/TF)
│   │   └── arima_baseline.py           # ARIMA baseline (statsmodels)
│   ├── portfolio/
│   │   ├── optimization.py             # MVO: 1/N, GMV, MSR with SLSQP
│   │   └── backtest.py                 # Walk-forward backtesting engine
│   └── utils/
│       └── metrics.py                  # RMSE, MAE, MAPE, R², Sharpe, MDD
│
├── results/                            # Figures, tables, equity curves
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for N-BEATS training)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/mv-nbeats-portfolio.git
cd mv-nbeats-portfolio
pip install -r requirements.txt
```

### Running the Pipeline

The recommended entry point is the notebook:

```bash
jupyter notebook notebooks/stock_prediction_full.ipynb
```

Or run individual components:

```bash
# Step 1: Preprocess data
python src/data_preprocessing.py

# Step 2: Feature selection
python src/feature_selection.py

# Step 3: Train Mv-N-BEATS and run backtest
python src/portfolio/backtest.py
```

**Note:** The notebook was originally developed on Kaggle. The data loading cell references `/kaggle/input/findata/`. To run locally, either download the data from Yahoo Finance using `yfinance` (the preprocessing script handles this automatically) or update the path accordingly.

---

## Data

- **Universe:** 27 large-cap U.S. equities (AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA, JPM, V, JNJ, WMT, PG, UNH, MA, HD, DIS, BAC, NFLX, ADBE, CRM, CSCO, PEP, INTC, CMCSA, PFE, ABT, TMO)
- **Period:** December 31, 2009 – December 31, 2021
- **Source:** Yahoo Finance (adjusted closing prices via `yfinance`)
- **Observations:** ~68,094 (1,970 trading days × 27 stocks)

---

## Methodology Notes

### Walk-Forward Backtesting

All models use an **expanding training window** — at each quarterly rebalancing date, models are retrained on all data from inception to that date. This strictly prevents look-ahead bias. The backtest period covers January 2021 onwards with ~19 rebalancing events.

### Prediction-Holding Horizon

Models produce **1-day-ahead forecasts** used as momentum signals for **~60-day quarterly holding periods**. This is standard practice in quantitative portfolio management — short-horizon predictions provide directional bias exploiting momentum persistence (Jegadeesh & Titman, 1993).

### Covariance Estimation

Ledoit-Wolf shrinkage is used throughout, applied to trailing 252-day return windows. This guarantees positive-definiteness and stability with N=27 assets.

---

## Citation

If you use this code or paper in your work, please cite:

```bibtex
@article{dubey2024mvnbeats,
  title     = {N-BEATS for Portfolio Construction: Integrating Interpretable Forecasting with Mean-Variance Optimization},
  author    = {Dubey, Medhaj and Reddy, M V Akshay and Malkani, Gaurav},
  year      = {2024},
  institution = {Manipal Institute of Technology}
}
```

---

## References

1. Chaweewanchon & Chaysiri (2022). Markowitz Mean-Variance Portfolio Optimization with Predictive Stock Selection. *IJFS.*
2. Oreshkin et al. (2020). N-BEATS: Neural Basis Expansion Analysis For Interpretable Time Series Forecasting. *ICLR 2020.*
3. Ledoit & Wolf (2004). A well-conditioned estimator for large-dimensional covariance matrices. *JMVA.*
4. Jegadeesh & Titman (1993). Returns to buying winners and selling losers. *Journal of Finance.*
5. Markowitz (1952). Portfolio Selection. *Journal of Finance.*

---

## License

MIT License. See `LICENSE` for details.
