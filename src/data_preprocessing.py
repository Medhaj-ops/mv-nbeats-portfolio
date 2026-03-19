"""
Data Preprocessing & Feature Engineering
=========================================
Downloads OHLCV data via yfinance and computes all 23 candidate features
across 5 categories: price/volume, returns/volatility, moving averages,
technical oscillators, and market context.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Universe
TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA',
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'MA', 'HD', 'DIS',
    'BAC', 'NFLX', 'ADBE', 'CRM', 'CSCO', 'PEP', 'INTC',
    'CMCSA', 'PFE', 'ABT', 'TMO'
]

START_DATE = '2009-12-31'
END_DATE = '2021-12-31'


def download_data(tickers: list = TICKERS,
                  start: str = START_DATE,
                  end: str = END_DATE) -> pd.DataFrame:
    """
    Download adjusted OHLCV data from Yahoo Finance.

    Returns:
        Long-format DataFrame with columns: Date, Ticker, Open, High, Low, Close, Volume
    """
    print(f"Downloading data for {len(tickers)} tickers ({start} to {end})...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    dfs = []
    for ticker in tickers:
        df = pd.DataFrame({
            'Date':   raw.index,
            'Ticker': ticker,
            'Open':   raw['Open'][ticker].values,
            'High':   raw['High'][ticker].values,
            'Low':    raw['Low'][ticker].values,
            'Close':  raw['Close'][ticker].values,
            'Volume': raw['Volume'][ticker].values,
        })
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data.sort_values(['Ticker', 'Date']).reset_index(drop=True)


def add_market_context(data: pd.DataFrame) -> pd.DataFrame:
    """Download and merge S&P500 and VIX context features."""
    print("Downloading market context (SPY, ^VIX)...")
    spy = yf.download('SPY', start=START_DATE, end=END_DATE,
                       auto_adjust=True, progress=False)[['Close', 'Volume']]
    spy.columns = ['S&P500_Close', 'S&P500_Volume']
    spy.index = pd.to_datetime(spy.index)

    vix = yf.download('^VIX', start=START_DATE, end=END_DATE,
                       auto_adjust=True, progress=False)[['Close']]
    vix.columns = ['VIX_Close']
    vix.index = pd.to_datetime(vix.index)

    context = spy.join(vix, how='left').reset_index().rename(columns={'Date': 'Date'})
    context['Date'] = pd.to_datetime(context['Date'])

    return data.merge(context, on='Date', how='left')


def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 23 candidate features per ticker.

    Categories:
      1. Price/Volume: Open, High, Low, Close, Volume, OBV
      2. Returns & Volatility: Daily_Return, Rolling_Vol_20D, Momentum_10D, BB_Percent
      3. Moving Averages: SMA_10, SMA_20, SMA_50, EMA_200
      4. Technical Oscillators: RSI_14, MACD, MACD_Signal, MACD_Histogram
      5. Market Context: S&P500_Close, S&P500_Volume, VIX_Close
    """
    all_dfs = []

    for ticker in data['Ticker'].unique():
        df = data[data['Ticker'] == ticker].copy().sort_values('Date').reset_index(drop=True)

        # ── Returns & Volatility ────────────────────────
        df['Daily_Return'] = df['Close'].pct_change()
        df['Rolling_Vol_20D'] = df['Daily_Return'].rolling(20).std()
        df['Momentum_10D'] = df['Close'].pct_change(10)

        # ── On-Balance Volume ───────────────────────────
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv

        # ── Moving Averages ─────────────────────────────
        df['SMA_10']  = df['Close'].rolling(10).mean()
        df['SMA_20']  = df['Close'].rolling(20).mean()
        df['SMA_50']  = df['Close'].rolling(50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

        # ── Bollinger Bands ─────────────────────────────
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        df['BB_Percent'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)

        # ── RSI (14-day Wilder smoothing) ───────────────
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # ── MACD ────────────────────────────────────────
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def build_dataset(save_path: str = None) -> pd.DataFrame:
    """Full pipeline: download → market context → features."""
    data = download_data()
    data = add_market_context(data)
    data = compute_features(data)
    data = data.dropna().reset_index(drop=True)

    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data['Date'].min()} → {data['Date'].max()}")
    print(f"Tickers: {data['Ticker'].nunique()}")

    if save_path:
        data.to_parquet(save_path, index=False)
        print(f"Saved to {save_path}")

    return data


if __name__ == '__main__':
    df = build_dataset(save_path='data/level2_data.parquet')
    print(df.head())
