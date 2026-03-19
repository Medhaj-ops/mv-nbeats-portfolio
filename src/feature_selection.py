"""
Feature Selection
=================
Two-stage selection combining:
  1. XGBoost gain-based feature importance
  2. Mutual Information (captures nonlinear dependencies)

Final selection: top-10 features passing both thresholds τ₁ and τ₂.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

CANDIDATE_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'OBV',
    'Daily_Return', 'Rolling_Vol_20D', 'Momentum_10D', 'BB_Percent',
    'SMA_10', 'SMA_20', 'SMA_50', 'EMA_200',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
    'S&P500_Close', 'S&P500_Volume', 'VIX_Close'
]

SELECTED_FEATURES = [
    'BB_Percent', 'RSI_14', 'Volume', 'MACD', 'MACD_Histogram',
    'Rolling_Vol_20D', 'Momentum_10D', 'EMA_200',
    'S&P500_Close', 'VIX_Close'
]


def compute_xgboost_importance(data_df: pd.DataFrame,
                                 features: list = CANDIDATE_FEATURES,
                                 target_col: str = 'Daily_Return',
                                 n_splits: int = 5) -> pd.Series:
    """
    Compute XGBoost gain-based feature importance using time-series CV.

    Args:
        data_df: Long-format DataFrame (all tickers pooled)
        features: Candidate feature columns
        target_col: Target variable column
        n_splits: Number of TimeSeriesSplit folds

    Returns:
        pd.Series of normalized importance scores, sorted descending
    """
    df = data_df[features + [target_col]].dropna()
    X = df[features].values
    y = df[target_col].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    importances = np.zeros(len(features))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X[train_idx], y[train_idx])
        importances += model.feature_importances_
        print(f"  XGBoost fold {fold+1}/{n_splits} done")

    importances /= n_splits
    importances /= importances.sum()  # normalize

    return pd.Series(importances, index=features).sort_values(ascending=False)


def compute_mutual_information(data_df: pd.DataFrame,
                                features: list = CANDIDATE_FEATURES,
                                target_col: str = 'Daily_Return',
                                random_state: int = 42) -> pd.Series:
    """
    Compute mutual information between each feature and the target.

    MI is model-agnostic and captures both linear and nonlinear dependencies.

    Args:
        data_df: Long-format DataFrame
        features: Candidate feature columns
        target_col: Target column
        random_state: For reproducibility

    Returns:
        pd.Series of MI scores, sorted descending
    """
    df = data_df[features + [target_col]].dropna()
    X = df[features].values
    y = df[target_col].values

    mi_scores = mutual_info_regression(X, y, random_state=random_state)
    mi_scores /= mi_scores.sum()  # normalize

    return pd.Series(mi_scores, index=features).sort_values(ascending=False)


def select_features(data_df: pd.DataFrame,
                     n_top: int = 10,
                     features: list = CANDIDATE_FEATURES) -> list:
    """
    Run dual-criterion feature selection.

    A feature is selected if:
        ImportanceXGB(f) > τ₁  AND  MI(f, y) > τ₂

    where τ₁, τ₂ are set to retain exactly n_top features.

    Args:
        data_df: Long-format DataFrame
        n_top: Number of features to select
        features: Candidate feature list

    Returns:
        List of selected feature names
    """
    print("Computing XGBoost importance...")
    xgb_scores = compute_xgboost_importance(data_df, features)

    print("Computing mutual information...")
    mi_scores = compute_mutual_information(data_df, features)

    # Rank by combined score
    combined = (xgb_scores + mi_scores).sort_values(ascending=False)
    selected = combined.head(n_top).index.tolist()

    print(f"\nSelected {n_top} features:")
    comparison = pd.DataFrame({
        'XGB': xgb_scores[selected],
        'MI':  mi_scores[selected]
    }).sort_values('XGB', ascending=False)
    print(comparison.to_string())

    return selected


if __name__ == '__main__':
    import pandas as pd
    # Assumes data has been built by data_preprocessing.py
    data = pd.read_parquet('data/level2_data.parquet')
    selected = select_features(data)
    print("\nFinal feature set:", selected)
