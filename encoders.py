"""
encoders.py
-----------
Encoder modules for all four modalities.

Implemented and working:
  - BiLSTMEncoder     (time series branch)
  - CNNEncoder        (candlestick chart image branch)
  - FinBERTEncoder    (financial news text branch)  ← partial integration

Not yet integrated:
  - SentimentEncoder  (Reddit/StockTwits sentiment branch)  ← TODO
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# ─── Time Series Encoder ──────────────────────────────────────────────────────

class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM over a (seq_len, n_features) window.
    Produces a fixed-size embedding from the final hidden state.

    NOTE (interim challenge): Currently overfitting with hidden_size=128, num_layers=2.
    Trying smaller configs. See Part 3 README for details.
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = 256,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project from 2 * hidden_size (bidirectional) to output_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch, output_dim)
        """
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, batch, hidden_size) — take last layer, both directions
        fwd = h_n[-2]   # forward direction, last layer
        bwd = h_n[-1]   # backward direction, last layer
        h = torch.cat([fwd, bwd], dim=-1)  # (batch, 2 * hidden_size)
        return self.proj(h)                 # (batch, output_dim)


# ─── Image Encoder (ResNet-18 backbone) ───────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Pretrained ResNet-18 with the final FC replaced by a projection head.
    All layers except the last two residual blocks are frozen.

    Input: (batch, 3, 224, 224) candlestick chart images (ImageNet normalized)
    Output: (batch, output_dim)
    """

    def __init__(self, output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze everything up to layer3 (keep layer4 trainable)
        for name, param in backbone.named_parameters():
            if not name.startswith(("layer4", "fc")):
                param.requires_grad = False

        # Replace the FC layer
        in_features = backbone.fc.in_features  # 512 for ResNet-18
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.proj = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 3, 224, 224)
        returns: (batch, output_dim)
        """
        feats = self.backbone(x)   # (batch, 512)
        return self.proj(feats)    # (batch, output_dim)


# ─── Text Encoder (FinBERT) ───────────────────────────────────────────────────

class FinBERTEncoder(nn.Module):
    """
    Encodes a list of financial news headlines using FinBERT.
    Aggregates multiple headlines per day via mean pooling of CLS tokens.

    KNOWN ISSUE (interim): Days with 0 headlines are imputed with a zero vector.
    Days with many headlines produce diluted pooled embeddings.
    Need to investigate attention-based pooling or top-k selection.
    See Part 3 README, Challenge 1.

    Model: ProsusAI/finbert (https://huggingface.co/ProsusAI/finbert)
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, output_dim: int = 128, dropout: float = 0.3, freeze_bert: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.bert = AutoModel.from_pretrained(self.MODEL_NAME)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_hidden = self.bert.config.hidden_size  # 768 for BERT-base

        self.proj = nn.Sequential(
            nn.Linear(bert_hidden, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def encode_headlines(self, headlines: list[str], device: torch.device) -> torch.Tensor:
        """
        Encodes a list of headlines and returns mean-pooled CLS embedding.
        Returns zero vector if headlines list is empty.
        """
        bert_hidden = self.bert.config.hidden_size

        if not headlines:
            return torch.zeros(bert_hidden, device=device)

        inputs = self.tokenizer(
            headlines,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad() if not self.bert.training else torch.enable_grad():
            outputs = self.bert(**inputs)

        cls_tokens = outputs.last_hidden_state[:, 0, :]  # (n_headlines, 768)
        return cls_tokens.mean(dim=0)                     # (768,)  ← mean pool

    def forward(self, batch_headlines: list[list[str]], device: torch.device) -> torch.Tensor:
        """
        batch_headlines: list of length batch_size, each element is a list of headlines for that day
        returns: (batch, output_dim)
        """
        embeddings = torch.stack(
            [self.encode_headlines(hl, device) for hl in batch_headlines]
        )  # (batch, 768)
        return self.proj(embeddings)  # (batch, output_dim)


# ─── Sentiment Encoder (TODO) ─────────────────────────────────────────────────

class SentimentEncoder(nn.Module):
    """
    Encodes daily Reddit/StockTwits sentiment scores into a fixed embedding.

    STATUS: Not yet integrated into training loop.
    Twitter Academic API deprecated → switching to StockTwits.
    See Part 3 README, Challenge 5.

    Input: (batch, n_sentiment_features)
      Expected features: [mean_sentiment, std_sentiment, post_count,
                          bullish_ratio, bearish_ratio]
    Output: (batch, output_dim)
    """

    def __init__(self, input_dim: int = 5, output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
