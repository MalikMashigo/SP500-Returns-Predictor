"""
model.py
--------
Late-fusion multimodal model combining all four encoder branches.
Currently functional with BiLSTM + CNN branches.
FinBERT and Sentiment branches are architecture-complete but not yet
connected in the training loop.

See Part 3 README for a description of training instability challenges.
"""

import torch
import torch.nn as nn
from encoders import BiLSTMEncoder, CNNEncoder, FinBERTEncoder, SentimentEncoder


class MultimodalFusionModel(nn.Module):
    """
    Late-fusion architecture:

      [BiLSTM] ─────────────────────┐
      [ResNet-18 CNN] ───────────────┤──► concat ──► FC layers ──► return prediction
      [FinBERT] ─────────────────────┤
      [Sentiment MLP] ───────────────┘

    Output: scalar next-day log return (regression, MSE loss)
    """

    def __init__(
        self,
        ts_output_dim: int = 256,
        img_output_dim: int = 128,
        text_output_dim: int = 128,
        sent_output_dim: int = 128,
        fusion_hidden: int = 256,
        dropout: float = 0.3,
        use_text: bool = False,       # flip to True once text branch is stable
        use_sentiment: bool = False,  # flip to True once sentiment pipeline done
    ):
        super().__init__()
        self.use_text = use_text
        self.use_sentiment = use_sentiment

        # Branch encoders
        self.ts_encoder = BiLSTMEncoder(output_dim=ts_output_dim)
        self.img_encoder = CNNEncoder(output_dim=img_output_dim)
        if use_text:
            self.text_encoder = FinBERTEncoder(output_dim=text_output_dim)
        if use_sentiment:
            self.sent_encoder = SentimentEncoder(output_dim=sent_output_dim)

        # Compute fusion input size based on active branches
        fusion_input = ts_output_dim + img_output_dim
        if use_text:
            fusion_input += text_output_dim
        if use_sentiment:
            fusion_input += sent_output_dim

        # Shared prediction head
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        ts_input: torch.Tensor,           # (B, seq_len, n_features)
        img_input: torch.Tensor,          # (B, 3, 224, 224)
        text_input=None,                  # list[list[str]] or None
        sentiment_input: torch.Tensor = None,  # (B, 5) or None
    ) -> torch.Tensor:
        device = ts_input.device

        ts_emb = self.ts_encoder(ts_input)       # (B, 256)
        img_emb = self.img_encoder(img_input)    # (B, 128)

        parts = [ts_emb, img_emb]

        if self.use_text and text_input is not None:
            text_emb = self.text_encoder(text_input, device)  # (B, 128)
            parts.append(text_emb)

        if self.use_sentiment and sentiment_input is not None:
            sent_emb = self.sent_encoder(sentiment_input)  # (B, 128)
            parts.append(sent_emb)

        fused = torch.cat(parts, dim=-1)         # (B, fusion_input)
        return self.fusion_head(fused).squeeze(-1)  # (B,)


# ─── Training Utilities ───────────────────────────────────────────────────────

def directional_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of samples where predicted sign matches target sign."""
    return ((preds > 0) == (targets > 0)).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_dir_acc = 0.0
    criterion = nn.MSELoss()

    for batch in loader:
        ts, img, targets = batch  # TODO: extend tuple when text/sentiment added
        ts = ts.to(device)
        img = img.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(ts_input=ts, img_input=img)
        loss = criterion(preds, targets)
        loss.backward()

        # Gradient clipping — important for LSTM stability
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_dir_acc += directional_accuracy(preds.detach(), targets)

    n = len(loader)
    return total_loss / n, total_dir_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_dir_acc = 0.0
    criterion = nn.MSELoss()

    for batch in loader:
        ts, img, targets = batch
        ts = ts.to(device)
        img = img.to(device)
        targets = targets.to(device)

        preds = model(ts_input=ts, img_input=img)
        loss = criterion(preds, targets)

        total_loss += loss.item()
        total_dir_acc += directional_accuracy(preds, targets)

    n = len(loader)
    return total_loss / n, total_dir_acc / n
