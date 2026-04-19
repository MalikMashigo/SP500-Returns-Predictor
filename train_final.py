"""
train_final.py
--------------
Final training script for the S&P 500 direction predictor.

Uses a BiLSTM-only architecture trained on OHLCV + technical indicator
windows downloaded automatically via yfinance.

Produces:
  checkpoints/bilstm_final.pt    — model weights
  checkpoints/norm_stats.npz     — training-set normalization statistics
  checkpoints/metrics.txt        — train/val evaluation summary

Usage:
    python train_final.py
    python train_final.py --epochs 60 --seq_len 30 --hidden 128 --lr 3e-4
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_auc_score,
)

# ── Optional yfinance download ─────────────────────────────────────────────────
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA
# ═══════════════════════════════════════════════════════════════════════════════

def download_sp500(start="2005-01-01", end="2022-01-01", csv_path="data/sp500_ohlcv.csv"):
    """Download ^GSPC from Yahoo Finance and cache as CSV."""
    if os.path.exists(csv_path):
        print(f"[data] Using cached {csv_path}")
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if not HAS_YFINANCE:
        raise RuntimeError("yfinance not installed and no cached CSV found. "
                           "Run: pip install yfinance")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    print(f"[data] Downloading ^GSPC {start} → {end} …")
    df = yf.download("^GSPC", start=start, end=end, progress=False)
    # yfinance may return MultiIndex columns; flatten if so
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(csv_path)
    print(f"[data] Saved to {csv_path}")
    return df


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # histogram only


def compute_bb_width(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (4 * std) / (sma + 1e-9)   # (upper-lower)/sma = 4σ/sma


def add_features(df):
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_volume"]  = np.log1p(df["Volume"])
    df["rsi"]         = compute_rsi(df["Close"])
    df["macd_hist"]   = compute_macd(df["Close"])
    df["bb_width"]    = compute_bb_width(df["Close"])
    df["roc_10"]      = df["Close"].pct_change(10)
    df["target"]      = df["log_return"].shift(-1)   # next-day log return
    return df.dropna()


FEATURE_COLS = ["log_return", "log_volume", "rsi", "macd_hist", "bb_width", "roc_10"]


class WindowDataset(Dataset):
    """Rolling 30-day windows → (features, target) pairs."""

    def __init__(self, df, seq_len=30, mean_=None, std_=None, fit_norm=True):
        feats = df[FEATURE_COLS].values.astype(np.float32)
        targets = df["target"].values.astype(np.float32)

        if fit_norm:
            self.mean_ = feats.mean(axis=0)
            self.std_  = feats.std(axis=0) + 1e-8
        else:
            self.mean_ = mean_
            self.std_  = std_

        feats = (feats - self.mean_) / self.std_

        self.windows, self.labels = [], []
        for i in range(seq_len, len(feats)):
            self.windows.append(feats[i - seq_len: i])
            self.labels.append(targets[i])

        self.seq_len    = seq_len
        self.n_features = len(FEATURE_COLS)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],  dtype=torch.float32)
        return x, y


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class BiLSTMRegressor(nn.Module):
    """
    Bidirectional LSTM → dense head → scalar next-day log-return prediction.
    Direction (up/down) is inferred from the sign of the output.
    """

    def __init__(self, input_size=6, hidden_size=128, num_layers=2,
                 dropout=0.3, proj_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        fwd = h_n[-2]   # last layer forward
        bwd = h_n[-1]   # last layer backward
        h = torch.cat([fwd, bwd], dim=-1)
        return self.head(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TRAINING / EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_metrics(preds, targets, split_name):
    """Return a dict of regression + directional classification metrics."""
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)

    # Binary direction labels: 1 = up (return > 0), 0 = down/flat
    pred_dir  = (preds   > 0).astype(int)
    true_dir  = (targets > 0).astype(int)

    acc  = accuracy_score(true_dir, pred_dir)
    prec = precision_score(true_dir, pred_dir, zero_division=0)
    rec  = recall_score(true_dir, pred_dir, zero_division=0)
    f1   = f1_score(true_dir, pred_dir, zero_division=0)
    f1_macro = f1_score(true_dir, pred_dir, average="macro", zero_division=0)
    cm   = confusion_matrix(true_dir, pred_dir)

    # ROC-AUC using raw predicted return as the score
    try:
        auc = roc_auc_score(true_dir, preds)
    except Exception:
        auc = float("nan")

    metrics = {
        "split": split_name,
        "n_samples": len(preds),
        "mae": mae,
        "mse": mse,
        "directional_accuracy": acc,
        "precision_up": prec,
        "recall_up": rec,
        "f1_up": f1,
        "f1_macro": f1_macro,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def print_metrics(m):
    cm = np.array(m["confusion_matrix"])
    print(f"\n{'='*55}")
    print(f"  {m['split'].upper()} SET  ({m['n_samples']} samples)")
    print(f"{'='*55}")
    print(f"  MSE                  : {m['mse']:.6f}")
    print(f"  MAE                  : {m['mae']:.6f}")
    print(f"  Directional Accuracy : {m['directional_accuracy']*100:.2f}%")
    print(f"  Precision (Up class) : {m['precision_up']:.4f}")
    print(f"  Recall    (Up class) : {m['recall_up']:.4f}")
    print(f"  F1        (Up class) : {m['f1_up']:.4f}")
    print(f"  F1  (macro avg)      : {m['f1_macro']:.4f}")
    print(f"  ROC-AUC              : {m['roc_auc']:.4f}")
    print(f"  Confusion Matrix (TN FP / FN TP):")
    print(f"    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"    {cm[1,0]:5d}  {cm[1,1]:5d}")
    print(f"{'='*55}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv",   default="data/sp500_ohlcv.csv")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--seq_len",    type=int,   default=30)
    p.add_argument("--hidden",     type=int,   default=128)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--patience",   type=int,   default=8)
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--train_end",  default="2018-12-31")
    p.add_argument("--val_end",    default="2021-12-31")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    df_raw = download_sp500(csv_path=args.data_csv)
    df = add_features(df_raw)

    train_df = df[df.index <= args.train_end]
    val_df   = df[(df.index > args.train_end) & (df.index <= args.val_end)]

    print(f"[data] Train samples (before windowing): {len(train_df)}")
    print(f"[data] Val   samples (before windowing): {len(val_df)}")

    train_ds = WindowDataset(train_df, seq_len=args.seq_len, fit_norm=True)
    val_ds   = WindowDataset(val_df,   seq_len=args.seq_len, fit_norm=False,
                             mean_=train_ds.mean_, std_=train_ds.std_)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ───────────────────────────────────────────────────────────────
    model = BiLSTMRegressor(
        input_size=len(FEATURE_COLS),
        hidden_size=args.hidden,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    # ── Training loop ────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "bilstm_final.pt")
    norm_path = os.path.join(args.save_dir, "norm_stats.npz")

    best_val_loss = float("inf")
    patience_ctr  = 0

    criterion = nn.MSELoss()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # quick validation loss for early stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved checkpoint (val MSE {best_val_loss:.6f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ── Final evaluation with best checkpoint ───────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    train_preds, train_targets = collect_preds(model, train_loader, device)
    val_preds,   val_targets   = collect_preds(model, val_loader,   device)

    train_metrics = compute_metrics(train_preds, train_targets, "train")
    val_metrics   = compute_metrics(val_preds,   val_targets,   "validation")

    print_metrics(train_metrics)
    print_metrics(val_metrics)

    # Save normalization stats so predict.py can reuse them
    np.savez(norm_path, mean=train_ds.mean_, std=train_ds.std_)
    print(f"\n[save] Norm stats → {norm_path}")

    # Save human-readable metrics summary
    metrics_path = os.path.join(args.save_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        for m in [train_metrics, val_metrics]:
            f.write(f"\n{m['split'].upper()} SET ({m['n_samples']} samples)\n")
            f.write(f"  MSE                  : {m['mse']:.6f}\n")
            f.write(f"  MAE                  : {m['mae']:.6f}\n")
            f.write(f"  Directional Accuracy : {m['directional_accuracy']*100:.2f}%\n")
            f.write(f"  Precision (Up)       : {m['precision_up']:.4f}\n")
            f.write(f"  Recall    (Up)       : {m['recall_up']:.4f}\n")
            f.write(f"  F1        (Up)       : {m['f1_up']:.4f}\n")
            f.write(f"  F1  (macro)          : {m['f1_macro']:.4f}\n")
            f.write(f"  ROC-AUC              : {m['roc_auc']:.4f}\n")
            f.write(f"  Confusion Matrix:\n")
            cm = np.array(m["confusion_matrix"])
            f.write(f"    TN={cm[0,0]}  FP={cm[0,1]}\n")
            f.write(f"    FN={cm[1,0]}  TP={cm[1,1]}\n")
    print(f"[save] Metrics summary → {metrics_path}")

    # ── Save a single validation sample for predict.py ──────────────────────
    sample_dir = "sample_val"
    os.makedirs(sample_dir, exist_ok=True)
    # Pick the first window from the val set (date after 2018-12-31 + seq_len days)
    sample_window = val_ds.windows[0]   # shape (seq_len, n_features) — already normalized
    sample_target = val_ds.labels[0]
    # Determine the date of this sample
    val_dates = val_df.index[args.seq_len:]
    sample_date = str(val_dates[0].date()) if len(val_dates) > 0 else "unknown"

    np.savez(
        os.path.join(sample_dir, "sample_val.npz"),
        window=sample_window,                  # (30, 6) normalized feature window
        target=np.array([sample_target]),      # scalar: actual next-day log return
        norm_mean=train_ds.mean_,              # (6,) — for reference
        norm_std=train_ds.std_,                # (6,) — for reference
        feature_names=np.array(FEATURE_COLS),
        sample_date=np.array([sample_date]),
    )
    print(f"[save] Validation sample ({sample_date}) → {sample_dir}/sample_val.npz")
    print("\nDone. Run  python predict.py  to test inference on the saved sample.")


if __name__ == "__main__":
    main()
