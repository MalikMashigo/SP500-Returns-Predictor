"""
chart_generator.py
------------------
Generates 224x224 candlestick chart images from OHLCV data
for the CNN encoder branch.

Each image represents a 30-trading-day window ending on the prediction date.
Images are saved to disk and loaded on demand by the dataset class.

Dependencies:
    pip install mplfinance pillow torch torchvision

Usage (batch generation):
    python chart_generator.py \
        --csv data/sp500_ohlcv_2005_2024.csv \
        --output_dir data/charts/ \
        --seq_len 30
"""

import os
import argparse
import io

import numpy as np
import pandas as pd
from PIL import Image
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ─── Image Generation ─────────────────────────────────────────────────────────

def generate_chart_image(ohlcv_window: pd.DataFrame, size: int = 224) -> Image.Image:
    """
    Renders a candlestick chart for a given OHLCV window and returns a PIL Image.
    No axis labels, no titles — CNN should learn from visual pattern only.
    """
    style = mpf.make_mpf_style(
        base_mpl_style="default",
        marketcolors=mpf.make_marketcolors(
            up="white", down="black",
            edge="black",
            wick="black",
            volume="gray",
        ),
        facecolor="white",
        gridstyle="",
    )

    buf = io.BytesIO()
    fig, _ = mpf.plot(
        ohlcv_window,
        type="candle",
        volume=True,
        style=style,
        axisoff=True,         # suppress all axis labels
        tight_layout=True,
        returnfig=True,
        figsize=(2.24, 2.24), # 100 dpi → 224x224 px
    )
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((size, size))
    return img


def pregenerate_charts(csv_path: str, output_dir: str, seq_len: int = 30):
    """
    Pre-generates chart images for all valid windows and saves them to disk.
    Skips dates that already have a saved image (idempotent).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # mplfinance expects specific column names
    df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low",
                             "Close": "Close", "Volume": "Volume"})

    dates = df.index[seq_len:]
    print(f"Generating {len(dates)} chart images...")

    for i, date in enumerate(dates):
        fname = os.path.join(output_dir, f"{date.strftime('%Y-%m-%d')}.png")
        if os.path.exists(fname):
            continue
        window = df.iloc[i : i + seq_len]
        img = generate_chart_image(window)
        img.save(fname)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(dates)} done")

    print("Chart generation complete.")


# ─── Dataset ──────────────────────────────────────────────────────────────────

# ImageNet normalization (for pretrained ResNet)
CHART_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class CandlestickDataset(Dataset):
    """
    Loads pre-generated candlestick chart images from disk.
    Paired with SP500WindowDataset by date index.
    """

    def __init__(self, chart_dir: str, dates: list, transform=CHART_TRANSFORM):
        self.chart_dir = chart_dir
        self.dates = dates
        self.transform = transform

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date_str = self.dates[idx].strftime("%Y-%m-%d")
        path = os.path.join(self.chart_dir, f"{date_str}.png")

        if not os.path.exists(path):
            # Fallback: return a blank image (shouldn't happen after pregeneration)
            img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        else:
            img = Image.open(path).convert("RGB")

        return self.transform(img)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=30)
    args = parser.parse_args()
    pregenerate_charts(args.csv, args.output_dir, args.seq_len)
