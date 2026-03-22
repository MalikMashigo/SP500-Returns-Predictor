"""
data_loader.py
--------------
Loads S&P 500 OHLCV data, computes technical indicators,
and assembles rolling 30-day windows for the BiLSTM encoder.

Data source: Yahoo Finance via yfinance (ticker ^GSPC, 2005-01-01 to 2024-12-31)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ─── Technical Indicator Computation ──────────────────────────────────────────

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger_width(series: pd.Series, window: int = 20) -> pd.Series:
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (upper - lower) / (sma + 1e-9)  # normalized width


def compute_roc(series: pd.Series, window: int = 10) -> pd.Series:
    return series.pct_change(window)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = compute_rsi(df["Close"])
    df["macd"], df["macd_signal"] = compute_macd(df["Close"])
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["bb_width"] = compute_bollinger_width(df["Close"])
    df["roc_10"] = compute_roc(df["Close"], 10)
    df["log_volume"] = np.log1p(df["Volume"])
    # Log returns as the primary price feature
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    # Target: next-day log return
    df["target"] = df["log_return"].shift(-1)
    return df


# ─── Feature Columns ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    "log_return", "log_volume",
    "rsi", "macd_hist", "bb_width", "roc_10",
]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SP500WindowDataset(Dataset):
    """
    Returns (window_tensor, target) pairs.
    window_tensor shape: (seq_len, n_features)

    Normalization is fit on the training slice only and stored in
    self.mean_ / self.std_ so they can be applied to val/test slices.
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "train",         # "train" | "val" | "test"
        seq_len: int = 30,
        train_end: str = "2018-12-31",
        val_end: str = "2021-12-31",
        mean_: np.ndarray = None,     # pass pre-fit stats for val/test
        std_: np.ndarray = None,
    ):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = add_indicators(df).dropna()

        # Temporal splits — strictly chronological, no shuffling
        if split == "train":
            df = df[df.index <= train_end]
        elif split == "val":
            df = df[(df.index > train_end) & (df.index <= val_end)]
        elif split == "test":
            df = df[df.index > val_end]
        else:
            raise ValueError(f"Unknown split: {split}")

        features = df[FEATURE_COLS].values.astype(np.float32)
        targets = df["target"].values.astype(np.float32)

        # Fit normalization on training set only
        if mean_ is None:
            self.mean_ = features.mean(axis=0)
            self.std_ = features.std(axis=0) + 1e-8
        else:
            self.mean_ = mean_
            self.std_ = std_

        features = (features - self.mean_) / self.std_

        # Build rolling windows
        self.windows = []
        self.labels = []
        for i in range(seq_len, len(features)):
            self.windows.append(features[i - seq_len : i])
            self.labels.append(targets[i])

        self.seq_len = seq_len
        self.n_features = len(FEATURE_COLS)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)  # (seq_len, n_features)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
