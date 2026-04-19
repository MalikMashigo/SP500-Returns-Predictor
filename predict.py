"""
predict.py
----------
Run the trained BiLSTM model on a single validation sample.

Requirements:
  pip install torch numpy

Usage:
    python predict.py
    python predict.py --sample sample_val/sample_val.npz \
                      --checkpoint checkpoints/bilstm_final.pt

The script prints the predicted next-day log return and the implied
directional call (UP or DOWN), alongside the actual return stored in the
sample file so you can see whether the model was correct.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn


# ── Model definition (must match train_final.py exactly) ──────────────────────

class BiLSTMRegressor(nn.Module):
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
        fwd = h_n[-2]
        bwd = h_n[-1]
        h = torch.cat([fwd, bwd], dim=-1)
        return self.head(h).squeeze(-1)


# ── Inference ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample",     default="sample_val/sample_val.npz",
                   help="Path to the .npz validation sample")
    p.add_argument("--checkpoint", default="checkpoints/bilstm_final.pt",
                   help="Path to trained model weights")
    p.add_argument("--hidden",     type=int, default=128,
                   help="Must match the value used during training")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load sample ───────────────────────────────────────────────────────────
    if not os.path.exists(args.sample):
        print(f"[error] Sample file not found: {args.sample}")
        print("  Run  python train_final.py  first, or check the path.")
        sys.exit(1)

    data = np.load(args.sample, allow_pickle=True)
    window       = data["window"].astype(np.float32)   # (seq_len, n_features)
    actual_ret   = float(data["target"][0])
    sample_date  = str(data["sample_date"][0])
    feature_names = list(data["feature_names"])

    seq_len, n_features = window.shape
    print(f"\n{'─'*55}")
    print(f"  Sample date  : {sample_date}")
    print(f"  Window shape : {seq_len} days × {n_features} features")
    print(f"  Features     : {', '.join(feature_names)}")
    print(f"  Actual next-day log return : {actual_ret:+.6f}  "
          f"({'UP' if actual_ret > 0 else 'DOWN'})")
    print(f"{'─'*55}")

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"\n[error] Checkpoint not found: {args.checkpoint}")
        print("  Run  python train_final.py  to train the model first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMRegressor(input_size=n_features, hidden_size=args.hidden).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # ── Run inference ─────────────────────────────────────────────────────────
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 30, 6)
    with torch.no_grad():
        pred_ret = model(x).item()

    direction      = "UP"   if pred_ret > 0 else "DOWN"
    actual_dir     = "UP"   if actual_ret > 0 else "DOWN"
    correct        = direction == actual_dir

    print(f"\n  Predicted log return : {pred_ret:+.6f}  → {direction}")
    print(f"  Actual    log return : {actual_ret:+.6f}  → {actual_dir}")
    print(f"\n  Directional call     : {'✓ CORRECT' if correct else '✗ INCORRECT'}")

    # Convert log return to approximate percentage
    pct = (np.exp(pred_ret) - 1) * 100
    print(f"  Implied % move       : {pct:+.3f}%")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
