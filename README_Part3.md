# Semester Project: Multimodal Neural Network for S&P 500 Return Prediction
**CSE-60868 Neural Networks | University of Notre Dame | Spring 2026**

---

## Part 1: Problem Formulation

The goal of this project is to build a multimodal neural network that predicts the next-day return of the S&P 500 index by simultaneously processing four types of input: historical price and volume time series, financial news headlines, candlestick chart images, and social media sentiment from Reddit and StockTwits. This is framed as a regression task where the model outputs a continuous predicted log return, since regression preserves more information than bucketing returns into discrete categories and gives a cleaner training signal via mean squared error.

The motivation for multimodality is that no single data source fully explains why markets move. Price data is backward-looking by nature — it tells you what already happened. News introduces forward-looking information about earnings, policy decisions, and geopolitical events that can shift market direction before it appears in price. Candlestick chart images encode the same OHLCV data as the time series but in a visual format that human traders have used for decades to identify patterns like head-and-shoulders formations, double bottoms, and momentum divergence. Whether a convolutional neural network can learn to recognize these visual patterns the way experienced traders do is itself an interesting research question. Finally, social media sentiment from retail investors on platforms like Reddit's WallStreetBets captures crowd psychology and speculative momentum that fundamentals-based models tend to miss entirely, as the 2021 GameStop episode made clear. The hypothesis driving this project is that fusing all four modalities will produce a more robust predictor than any single modality alone, and ablation experiments planned for the final phase will test that directly.

The architecture is a late-fusion multimodal network where each modality is processed by a dedicated encoder before the representations are concatenated and passed through a shared regression head. The time series branch uses a Bidirectional LSTM over 30-day windows of OHLCV data plus engineered technical indicators (RSI, MACD histogram, Bollinger Band width, 10-day rate of change). The text branch uses FinBERT — a BERT model pretrained specifically on financial corpora — to encode daily news headlines, which are then mean-pooled into a single fixed-size embedding. The image branch processes programmatically generated 224×224 candlestick chart images through a pretrained ResNet-18 with the final layers replaced. The sentiment branch aggregates Reddit and StockTwits posts into daily sentiment scores using a pretrained RoBERTa-based financial sentiment model and passes them through a small dense encoder. All four embeddings are concatenated and passed through fully connected layers with dropout before a single linear output neuron. Evaluation metrics include MAE, MSE, and directional accuracy (fraction of days where the predicted return sign matches the actual sign), with ablation experiments testing each modality individually against the full fusion model.

The primary challenge in financial prediction tasks is overfitting and look-ahead bias. The temporal structure of the data means the train/validation/test split must be strictly chronological — no shuffling — because any leakage of future information into the training set produces misleadingly optimistic results that would not hold in real deployment. Regularization through dropout, weight decay, and early stopping will be essential given the relatively small number of training samples (approximately 3,500 trading days).

---

## Part 2: Datasets

For this project I need data that captures the behavior of the S&P 500 across a wide range of market conditions — bull runs, crashes, recoveries, and everything in between. My primary dataset is S&P 500 historical OHLCV data (open, high, low, close prices, and trading volume) downloaded through the `yfinance` Python library using the ticker `^GSPC`, covering January 2, 2005 through December 31, 2024, which comes to roughly 5,030 trading days. The yfinance library is available at [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/).

Alongside the price data I downloaded secondary time series: the CBOE Volatility Index (VIX, ticker `^VIX`) from Yahoo Finance and closing prices for five sector ETFs — XLK (Technology), XLF (Financials), XLV (Healthcare), XLE (Energy), and XLP (Consumer Staples). All OHLCV and VIX data has been physically downloaded and stored locally as CSV files. The candlestick chart images for the CNN branch are generated programmatically from this same price data using the `mplfinance` library, so no additional download is required for that modality.

For the news text branch, I am using the Alpaca News API, which provides timestamped financial headlines tagged to specific equity tickers and broad market indices. Headlines tagged to SPY or the broader market are collected for each trading day during U.S. market hours. The API documentation is available at [https://alpaca.markets/docs/api-references/market-data-api/news/](https://alpaca.markets/docs/api-references/market-data-api/news/). For the text encoder I am using FinBERT, a BERT model pretrained on financial corpora, available at [https://huggingface.co/ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert).

For the sentiment branch, I am using the Pushshift Reddit dataset filtered to r/wallstreetbets and r/investing, combined with posts from StockTwits via their public API. Daily sentiment scores are computed using a pretrained RoBERTa-based financial sentiment model available at [https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis). The Reddit data is available through Pushshift at [https://pushshift.io/](https://pushshift.io/).

The dataset must be split chronologically rather than randomly, since shuffling would introduce look-ahead bias. The training set covers January 2005 through December 2018 (approximately 3,520 trading days), the validation set covers January 2019 through December 2021 (approximately 756 days), and the test set covers January 2022 through December 2024 (approximately 754 days). These splits were chosen to separate qualitatively different market regimes: the training set contains the 2008–09 financial crisis, the 2010–15 recovery, and the 2016–18 expansion; the validation set contains the 2020 COVID crash and the 2020–21 bull market; and the test set contains the 2022 Fed rate-hiking cycle and 2023–24 recovery. This means the model will need to generalize across regime shifts it has never seen during training, which is a realistic challenge for any production forecasting system.

A representative training set sample (2008-09-15, the day of the Lehman Brothers collapse):
```
Date: 2008-09-15 | Close: 1192.7 | Volume: 9.1B | VIX: 31.7 | 10Y Yield: 3.47% | Next-day log return: -0.047
```
A representative validation set sample (2020-03-16, COVID crash):
```
Date: 2020-03-16 | Close: 2386.1 | Volume: 14.5B | VIX: 82.7 | 10Y Yield: 0.73% | Next-day log return: +0.062
```

This project is completed individually. All data acquisition, preprocessing design, and report writing were completed by the submitting student.

---

## Part 3: Interim Results and Current Challenges

### Overview

At this point in the semester, the project has progressed from conceptual design and data collection into active implementation. The core data pipeline is functional, three of the four modality encoders have been built and tested in isolation, and the late-fusion architecture skeleton is in place. However, several significant challenges have emerged during implementation that are shaping the direction of the remaining work. This section describes what has been built, what is working, what is not, and where I am most in need of guidance.

### What Has Been Implemented

The time series branch is the most complete component of the system. I have a working data loader (`data_loader.py`) that reads OHLCV data from a local CSV, computes a set of technical indicators — RSI (14-day), MACD histogram, Bollinger Band width, and a 10-day rate of change — and assembles rolling 30-day windows as input sequences. Each window is normalized using z-score statistics computed exclusively on the training set, which is critical for avoiding look-ahead bias. The BiLSTM encoder is implemented in PyTorch (`encoders.py`) with two bidirectional layers and a hidden size of 128 per direction, producing a 256-dimensional output embedding after passing the final hidden state through a linear projection. I verified that this encoder trains stably on the training set alone and produces reasonable loss curves, giving me a working single-modality baseline.

The candlestick image generation pipeline (`chart_generator.py`) is also functional. I am using `mplfinance` to render 30-day candlestick charts for each prediction date and saving them as 224×224 PNG files. The charts have axis labels removed so the CNN must learn from visual price pattern rather than scale information. The CNN encoder is a pretrained ResNet-18 with the final fully-connected layer replaced by a linear projection to 128 dimensions, with all layers except the last two residual blocks frozen during training. A basic training loop on the image branch alone confirms the pipeline is connected end to end, though the image branch alone performs only modestly better than predicting the mean return.

The FinBERT text encoder is partially implemented. I have a working script (`news_fetcher.py`) that queries the Alpaca News API, retrieves headlines for each trading day, tokenizes them using the FinBERT tokenizer, and produces CLS-token embeddings. For days with multiple headlines, I am currently using simple mean pooling across all headline embeddings to produce a single 768-dimensional text representation, projected down to 128 dimensions by a learned linear layer. The encoder produces outputs with correct dimensions, but it has not yet been integrated into a joint training loop because of the complications described below.

The overall fusion architecture (`model.py`) is also in place: all four branch embeddings (256 + 128 + 128 + 128 = 640 total) are concatenated and passed through two fully connected layers with ReLU activations and dropout at 0.3, ending in a single output neuron. A training script (`train.py`) runs the time series and image branches jointly. The text and sentiment branches are implemented as modules but gated behind flags (`use_text=False`, `use_sentiment=False`) until the issues below are resolved.

### Current Challenges

**Challenge 1: News data alignment and API reliability.** The most pressing practical problem is aligning financial news headlines to trading days consistently. The Alpaca News API does not always return results for every trading date, and the number of headlines per day varies enormously — some days return two or three articles, others return thirty or more. This creates two sub-problems. First, mean pooling over a variable number of embeddings produces representations of inconsistent quality: on high-news days the pooled embedding is diluted across many unrelated stories, while on low-news days it may reflect one idiosyncratic article. I have experimented with taking only the top-five headlines by Alpaca's relevance score, which helps somewhat, but I am not sure this is the right approach. I would like to discuss with Adam and the TA whether attention-based pooling over the headline set is worth the added complexity at this stage, or whether simple top-k selection is sufficient.

Second, roughly 80–100 training-set trading days have zero headlines in the Alpaca database, likely because coverage is sparse before 2018. I am currently imputing zero vectors for these days, which is a rough approximation. A smarter fallback might be to treat the absence of news as a valid signal — the model could learn that no-news days have a distinct character — or to source a secondary news feed for gap-filling. I would appreciate guidance on how to handle this properly before committing to one approach.

**Challenge 2: Overfitting in the time series branch.** Even with dropout and weight decay, the BiLSTM encoder begins overfitting after approximately 15–20 epochs: training loss decreases while validation loss flattens and then rises. The model has around 1.2M parameters in the LSTM layers, which is probably too many for roughly 3,500 training samples. Reducing hidden size from 256 to 128 and layers from 2 to 1 helps, but the train/validation gap is still uncomfortably large. I have tried increasing dropout to 0.5 and adding L2 regularization, but this slows convergence without eliminating the overfit. My current thinking is to constrain the model more aggressively — either a much smaller single-layer LSTM or a 1D CNN over the time series — but I am not confident which direction is better for this type of financial data, and I would like to discuss this tradeoff with Adam and the TA.

**Challenge 3: Multimodal training instability when branches are fused.** When I connect the time series and image branches into the joint fusion architecture and train them simultaneously, training becomes noticeably less stable than training either branch alone. The loss oscillates more, and on several runs it diverges within the first few epochs. My hypothesis is that the two branches learn at very different rates — the ResNet parameters (even partially frozen) respond to very different gradient magnitudes than the randomly initialized LSTM — and this mismatch causes instability. I have partially addressed this by using per-branch learning rates in the optimizer (3e-4 for the LSTM, 1e-5 for the ResNet), which improves stability but does not fully resolve it. I have read about gradient normalization and modality dropout as potential remedies but have not implemented either. Any guidance on standard approaches to this problem in multimodal training would be very helpful.

**Challenge 4: Evaluation metric interpretation.** The natural metrics for this regression task — MAE and MSE — are hard to interpret in isolation because daily returns are tiny in absolute terms (typically ±1–2%). An MAE of 0.003 sounds small but may be poor depending on the return distribution. I have therefore also been tracking directional accuracy (the fraction of days where the model correctly predicts the sign of the return), which is more interpretable: random guessing scores ~50%, and a naive majority classifier scores ~55% on the training set. My two-branch model currently achieves roughly 54–56% directional accuracy on the validation set, which is barely above the naive baseline. I want to discuss with Adam and the TA whether this is expected at this stage, and whether directional accuracy is a fair measure given that being right on large-move days might add more value than being right on small-move days. I am also considering a Sharpe-ratio-based evaluation on the test set as a more financially meaningful metric, but I am uncertain whether that is appropriate for this course context.

**Challenge 5: Sentiment pipeline — Twitter/X API access lost.** The fourth modality is the least complete part of the project. I have a working script that loads preprocessed Reddit posts from Pushshift and scores them with a fine-tuned RoBERTa sentiment model, but the Twitter data collection has stalled because the Academic API access I was relying on has been deprecated, and the current free tier severely limits historical data retrieval. I am planning to substitute StockTwits data as a replacement, since it is a more financially focused platform and its API allows historical post retrieval. This scope adjustment means the sentiment branch will be trained on Reddit plus StockTwits rather than Reddit plus Twitter, which may actually improve signal quality since both are more financially specialized than general Twitter. I would like to confirm with Adam that this substitution is acceptable before investing time in the StockTwits integration.

### Next Steps

In the remaining weeks of the semester, the priorities are as follows. First, resolve the news alignment problem and fully integrate the FinBERT branch into the joint training loop. Second, stabilize multimodal training, most likely through per-branch learning rate scheduling and possibly modality dropout, and potentially reducing LSTM capacity to address overfitting. Third, complete the StockTwits-based sentiment pipeline and wire in the fourth branch. Fourth, run ablation experiments comparing single-modality baselines against the full fusion model to measure each modality's contribution. Fifth, evaluate the final model on the held-out 2022–2024 test set that has not been touched during development.

### Individual Contributions

This is a solo project. All work — including data collection and preprocessing, architecture design, PyTorch implementation, training experiments, debugging, and report writing — was completed independently by the submitting student.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/                          # local data directory (not committed to git)
│   └── sp500_ohlcv.csv            # downloaded automatically by train_final.py
├── data_loader.py                 # OHLCV dataset + technical indicators
├── chart_generator.py             # candlestick image generation + dataset
├── encoders.py                    # BiLSTM, CNN, FinBERT, Sentiment encoders
├── model.py                       # multimodal fusion model + training utilities
├── train.py                       # interim multimodal training script (Part 3)
├── train_final.py                 # ← FINAL training script (Part 4)
├── predict.py                     # ← single-sample inference (Part 4)
├── news_fetcher.py                # Alpaca News API integration
├── sample_val/
│   └── sample_val.npz             # ← single validation sample (Part 4)
└── checkpoints/                   # saved model weights (not committed to git)
    ├── bilstm_final.pt
    └── norm_stats.npz
```

---

*Part 4 (Final Evaluation) is in [README_Part4.md](README_Part4.md).*

