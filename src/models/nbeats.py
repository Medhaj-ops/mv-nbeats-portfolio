"""
Mv-N-BEATS: Multivariate-Adapted N-BEATS Architecture
======================================================
Extends the original N-BEATS (Oreshkin et al., ICLR 2020) to handle
multivariate financial feature inputs via a flattened (L x D) input tensor.

Architecture:
- Input: 200-dim vector (20 days x 10 features, flattened)
- 2 stacks: Trend (polynomial basis) + Seasonality (Fourier basis)
- 3 blocks per stack, each with 4 FC layers (256 units, ReLU)
- Theta projection: 2-layer MLP [128, 128]
- Output: 1-day-ahead price forecast
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SELECTED_FEATURES = [
    'BB_Percent', 'RSI_14', 'Volume', 'MACD', 'MACD_Histogram',
    'Rolling_Vol_20D', 'Momentum_10D', 'EMA_200',
    'S&P500_Close', 'VIX_Close'
]
LOOKBACK_WINDOW = 20
TEST_SIZE = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# N-BEATS Hyperparameters
STACK_TYPES = ['trend', 'seasonality']
NUM_BLOCKS_PER_STACK = 3
HIDDEN_LAYER_UNITS = 256
THETA_HIDDEN_UNITS = [128, 128]
EPOCHS = 25
BATCH_SIZE = 32


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# Basis Functions
# ─────────────────────────────────────────────

class TrendBasis(nn.Module):
    """Polynomial trend basis of specified degree."""

    def __init__(self, degree: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.degree = degree
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor, is_backcast: bool) -> torch.Tensor:
        T = self.backcast_size if is_backcast else self.forecast_size
        t = torch.linspace(0, 1, T, device=theta.device)
        basis = torch.stack([t ** i for i in range(self.degree + 1)], dim=0)  # (degree+1, T)
        return torch.einsum('bd,dt->bt', theta, basis)


class SeasonalityBasis(nn.Module):
    """Fourier seasonality basis with H harmonics."""

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.harmonics = harmonics
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor, is_backcast: bool) -> torch.Tensor:
        T = self.backcast_size if is_backcast else self.forecast_size
        t = torch.linspace(0, 1, T, device=theta.device)
        cos_terms = [torch.cos(2 * np.pi * i * t) for i in range(1, self.harmonics + 1)]
        sin_terms = [torch.sin(2 * np.pi * i * t) for i in range(1, self.harmonics + 1)]
        basis = torch.stack(cos_terms + sin_terms, dim=0)  # (2*H, T)
        return torch.einsum('bd,dt->bt', theta, basis)


# ─────────────────────────────────────────────
# N-BEATS Block
# ─────────────────────────────────────────────

class NBeatsBlock(nn.Module):
    """Single N-BEATS block with FC stack + backcast/forecast projection."""

    def __init__(self, input_size: int, theta_size: int, basis_fn: nn.Module,
                 hidden_units: list):
        super().__init__()
        layers = []
        prev = input_size
        for units in hidden_units:
            layers += [nn.Linear(prev, units), nn.ReLU()]
            prev = units
        self.fc_stack = nn.Sequential(*layers)

        # Theta projection MLPs
        self.theta_b = nn.Sequential(
            nn.Linear(prev, THETA_HIDDEN_UNITS[0]), nn.ReLU(),
            nn.Linear(THETA_HIDDEN_UNITS[0], theta_size)
        )
        self.theta_f = nn.Sequential(
            nn.Linear(prev, THETA_HIDDEN_UNITS[0]), nn.ReLU(),
            nn.Linear(THETA_HIDDEN_UNITS[0], theta_size)
        )
        self.basis_fn = basis_fn

    def forward(self, x: torch.Tensor):
        h = self.fc_stack(x)
        backcast = self.basis_fn(self.theta_b(h), is_backcast=True)
        forecast = self.basis_fn(self.theta_f(h), is_backcast=False)
        return backcast, forecast


# ─────────────────────────────────────────────
# Mv-N-BEATS Model
# ─────────────────────────────────────────────

class MvNBeats(nn.Module):
    """
    Multivariate-adapted N-BEATS.

    Input: (batch, L*D) = (batch, 200) — flattened (20 days x 10 features)
    Output: (batch, 1) — next-day price forecast
    """

    def __init__(self,
                 input_size: int = 200,
                 backcast_size: int = 20,
                 forecast_size: int = 1,
                 stack_types: list = None,
                 num_blocks_per_stack: int = 3,
                 hidden_units: int = 256,
                 trend_degree: int = 3,
                 seasonality_harmonics: int = 10):
        super().__init__()

        if stack_types is None:
            stack_types = ['trend', 'seasonality']

        self.input_size = input_size
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

        self.stacks = nn.ModuleList()
        hidden = [hidden_units] * 4

        for stack_type in stack_types:
            blocks = nn.ModuleList()
            for _ in range(num_blocks_per_stack):
                if stack_type == 'trend':
                    theta_size = trend_degree + 1
                    basis_fn = TrendBasis(trend_degree, backcast_size, forecast_size)
                elif stack_type == 'seasonality':
                    theta_size = 2 * seasonality_harmonics
                    basis_fn = SeasonalityBasis(seasonality_harmonics, backcast_size, forecast_size)
                else:
                    raise ValueError(f"Unknown stack type: {stack_type}")

                blocks.append(NBeatsBlock(input_size, theta_size, basis_fn, hidden))
            self.stacks.append(blocks)

    def forward(self, x: torch.Tensor):
        residual = x  # (batch, input_size)
        forecast = torch.zeros(x.size(0), self.forecast_size, device=x.device)

        for stack in self.stacks:
            for block in stack:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast

        return forecast  # (batch, 1)


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

def prepare_multivariate_datasets(data_df, lookback_window=LOOKBACK_WINDOW,
                                   test_size=TEST_SIZE,
                                   selected_features=SELECTED_FEATURES):
    """
    Prepares per-ticker (X_train, X_test, y_train, y_test, scaler) dictionaries
    from a long-format DataFrame.

    Args:
        data_df: Long-format DataFrame with columns ['Date', 'Ticker', 'Close', ...features]
        lookback_window: Number of historical days per sample
        test_size: Fraction of data reserved for testing
        selected_features: List of feature column names

    Returns:
        dict: {ticker -> {'X_train', 'X_test', 'y_train', 'y_test', 'scaler', 'test_dates'}}
    """
    datasets = {}
    tickers = data_df['Ticker'].unique()

    for ticker in tickers:
        df = data_df[data_df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
        df = df.dropna(subset=selected_features + ['Close'])

        features = df[selected_features].values
        target = df['Close'].values

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        X, y = [], []
        for i in range(lookback_window, len(features_scaled)):
            window = features_scaled[i - lookback_window:i]   # (L, D)
            X.append(window.flatten())                          # (L*D,)
            y.append(target[i])

        X, y = np.array(X), np.array(y)
        split = int(len(X) * (1 - test_size))

        datasets[ticker] = {
            'X_train': X[:split],
            'X_test': X[split:],
            'y_train': y[:split],
            'y_test': y[split:],
            'scaler': scaler,
            'test_dates': df['Date'].values[lookback_window + split:]
        }

    return datasets


def train_nbeats(ticker: str, ticker_data: dict, epochs: int = EPOCHS,
                 batch_size: int = BATCH_SIZE) -> tuple:
    """Train a single Mv-N-BEATS model for one ticker."""
    X_train = ticker_data['X_train']
    y_train = ticker_data['y_train'].reshape(-1, 1)
    X_test = ticker_data['X_test']
    y_test = ticker_data['y_test']

    # Normalize targets
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)

    train_loader = DataLoader(
        StockDataset(X_train, y_train_norm),
        batch_size=batch_size, shuffle=True
    )

    model = MvNBeats(input_size=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(DEVICE)
        preds_norm = model(X_test_t).cpu().numpy().flatten()
        preds = preds_norm * y_std + y_mean

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-8))) * 100
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}
    print(f"  {ticker}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, R²={r2:.4f}")

    return model, preds, metrics
