"""
train.py
--------
Main training script for the multimodal S&P 500 return prediction model.

Current status (Part 3 interim):
  - BiLSTM + CNN branches running end-to-end.
  - Per-branch learning rates implemented to address training instability.
  - FinBERT and Sentiment branches not yet wired in (see model.py flags).
  - Early stopping based on validation MSE.

Usage:
    python train.py --epochs 50 --batch_size 32 --lr_lstm 3e-4 --lr_cnn 1e-5
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import SP500WindowDataset
from chart_generator import CandlestickDataset  # wraps image generation
from model import MultimodalFusionModel, train_one_epoch, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", type=str, default="data/sp500_ohlcv_2005_2024.csv")
    p.add_argument("--chart_dir", type=str, default="data/charts/")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--lr_lstm", type=float, default=3e-4)
    p.add_argument("--lr_cnn", type=float, default=1e-5)   # lower for pretrained ResNet
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=8)       # early stopping patience
    p.add_argument("--save_path", type=str, default="checkpoints/best_model.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ts = SP500WindowDataset(args.data_csv, split="train", seq_len=args.seq_len)
    val_ts   = SP500WindowDataset(
        args.data_csv, split="val", seq_len=args.seq_len,
        mean_=train_ts.mean_, std_=train_ts.std_,  # use training stats
    )

    # TODO: replace with real multimodal DataLoader that returns (ts, img, target)
    # For now, using placeholder DataLoader over TS data only
    train_loader = DataLoader(train_ts, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ts,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MultimodalFusionModel(
        dropout=args.dropout,
        use_text=False,        # flip once FinBERT integration is stable
        use_sentiment=False,
    ).to(device)

    # Per-branch learning rates to address training instability (Part 3, Challenge 3)
    param_groups = [
        {"params": model.ts_encoder.parameters(),     "lr": args.lr_lstm},
        {"params": model.img_encoder.parameters(),    "lr": args.lr_cnn},
        {"params": model.fusion_head.parameters(),    "lr": args.lr_lstm},
    ]
    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, verbose=True
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dir = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dir     = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_loss:.6f}  Dir: {train_dir:.3f} | "
            f"Val MSE:   {val_loss:.6f}  Dir: {val_dir:.3f}"
        )

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✓ Saved new best model (val MSE {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.6f}")
    print("Next: load best checkpoint and run ablation experiments.")


if __name__ == "__main__":
    main()
