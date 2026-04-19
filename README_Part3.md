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

## Part 4: Final Evaluation

### How to Run

**Step 1 — Create a virtual environment and install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install torch yfinance scikit-learn pandas numpy
```

**Step 2 — Train the model** (downloads S&P 500 data automatically, takes ~5–10 min on CPU)

```bash
python train_final.py
```

This script downloads `^GSPC` OHLCV data from Yahoo Finance via `yfinance`, trains a
BiLSTM regressor on the 2005–2018 training split, evaluates on the 2019–2021 validation
split, and saves `checkpoints/bilstm_final.pt`, `checkpoints/norm_stats.npz`, and
`checkpoints/metrics.txt`. It also saves the bundled validation sample to
`sample_val/sample_val.npz`.

**Step 3 — Run inference on the included single validation sample**

```bash
python predict.py
```

This requires no additional downloads. It loads `sample_val/sample_val.npz` (a
pre-normalized 30-day feature window for 2019-02-19) and the trained checkpoint,
prints the predicted next-day log return, the implied directional call (UP or DOWN),
and compares it to the actual return stored in the sample file.

Optional flags:

```bash
python predict.py --sample sample_val/sample_val.npz \
                  --checkpoint checkpoints/bilstm_final.pt
```

> **Note for macOS (Homebrew Python):** system pip is blocked by PEP 668.
> Always activate the venv (`source venv/bin/activate`) before running
> `pip install` or any of the scripts above.

---

### Model Architecture for the Final Submission

The final trained model is the **BiLSTM-only branch** of the larger multimodal
architecture described in Parts 1–3. The full multimodal fusion (BiLSTM + ResNet-18
CNN + FinBERT + SentimentMLP) remains the long-term design goal, but the CNN branch
depends on pre-generated candlestick chart images (not committed to the repo due to
size), and the FinBERT/sentiment branches had not stabilized by the final deadline as
documented in Part 3. The BiLSTM branch is the component that ran end-to-end and
produced reproducible results on both train and validation splits.

Architecture summary:

```
Input: (batch, 30, 6)          — 30-day rolling window, 6 features
  │
  ▼
BiLSTM (hidden=128, layers=2, bidirectional)
  │  concat [forward h_n, backward h_n]  →  (batch, 256)
  ▼
Linear(256 → 128) + ReLU + Dropout(0.3)
Linear(128 →  64) + ReLU + Dropout(0.3)
Linear( 64 →   1)         →  scalar predicted next-day log return
```

The six input features — daily log return, log-volume, 14-day RSI, MACD histogram,
Bollinger Band normalized width, and 10-day rate of change — are z-scored using
statistics fit exclusively on the training split and saved to `checkpoints/norm_stats.npz`
for consistent reuse at inference time. Trainable parameter count: ~492,000.

Training objective: MSE on the continuous next-day log return. The binary
directional call (UP / DOWN) is derived from the sign of the predicted value.

---

### Classification Accuracy

#### Baselines

Before reporting model accuracy, it is essential to establish baselines so the
numbers are meaningful in context:

| Baseline | Train Directional Acc | Val Directional Acc |
|---|---|---|
| Random (50/50 coin flip) | ~50.0% | ~50.0% |
| Majority class (always predict Up) | ~55.0% | ~56.9% |
| **This model (BiLSTM)** | **57.3%** | **54.1%** |

Roughly 55% of training-set trading days had positive returns, and roughly 57% of
validation-set days did (the 2019–2021 period was a strong bull run punctuated by the
2020 COVID recovery). A trivial majority classifier — always predicting Up — scores
55% on the training set and 57% on the validation set without learning anything.
The model beats the majority classifier on the training set but falls 2.8 points
short on the validation set. This is a sobering but honest comparison that raw
accuracy alone would hide.

#### Training Set (2005-01-01 to 2018-12-31 — 3,490 windows)

| Metric | Value |
|---|---|
| MSE | 3.41 × 10⁻⁴ |
| MAE | 0.00521 |
| **Directional accuracy** | **57.3%** |
| Precision — Up class | 0.610 |
| Recall — Up class | 0.619 |
| F1 — Up class | 0.614 |
| F1 — Down class | 0.521 |
| **F1 macro** | **0.568** |
| ROC-AUC | 0.585 |

**Correctly classified: 57.3% (1,999 / 3,490 windows)**  
**Incorrectly classified: 42.7% (1,491 / 3,490 windows)**

Confusion matrix (rows = actual direction, columns = predicted direction):

```
                 Predicted Down   Predicted Up
Actual Down           812              758        (1,570 Down days)
Actual Up             733            1,187        (1,920 Up days)
```

The model correctly identifies 61.9% of actual Up days (recall = 1,187 / 1,920) but
only 51.7% of actual Down days (recall = 812 / 1,570). In other words, it is
materially better at recognising bull conditions than bear conditions on training data,
likely because the training set is dominated by the 2010–2018 expansion and the model
has absorbed the persistent upward drift of that period.

#### Validation Set (2019-01-01 to 2021-12-31 — 726 windows)

| Metric | Value |
|---|---|
| MSE | 4.19 × 10⁻⁴ |
| MAE | 0.00683 |
| **Directional accuracy** | **54.1%** |
| Precision — Up class | 0.599 |
| Recall — Up class | 0.576 |
| F1 — Up class | 0.587 |
| F1 — Down class | 0.480 |
| **F1 macro** | **0.534** |
| ROC-AUC | 0.548 |

**Correctly classified: 54.1% (393 / 726 windows)**  
**Incorrectly classified: 45.9% (333 / 726 windows)**

Confusion matrix:

```
                 Predicted Down   Predicted Up
Actual Down           154              159        (313 Down days)
Actual Up             175              238        (413 Up days)
```

On the validation set the model correctly identifies 57.6% of Up days (238 / 413)
and 49.2% of Down days (154 / 313). Down-day performance has collapsed to barely
above random, while Up-day recall has also degraded compared to training. This
asymmetric degradation points directly to the COVID-19 crash of February–March 2020:
sharp drawdown days of −4% to −12% look nothing like anything in the 2005–2018
training data, so the model almost never predicts them.

---

### Choice and Justification of Evaluation Metrics

**Why simple accuracy is insufficient here.** The dataset has a mild class imbalance:
~55% of training days are Up and ~57% of validation days are Up. A model that learns
nothing but the class prior would achieve 55–57% accuracy, which looks reasonable on
paper but provides zero actionable signal. Reporting accuracy alongside precision,
recall, F1, and ROC-AUC prevents this illusion.

**Directional accuracy as the primary classification metric.** Although the network
is trained as a regressor (MSE on log returns), the decision that matters operationally
is binary — should a trader go long or stay flat? Directional accuracy — the fraction
of days where `sign(predicted_return) == sign(actual_return)` — maps exactly onto
this decision. It is the most interpretable and practically relevant metric for this
task, and it is what financial ML papers uniformly report for return-prediction models.

**Precision and Recall, reported per class.** Treating Up and Down as separate
classes and computing precision and recall for each reveals the asymmetry the model
has learned. Precision for the Up class (0.599 on validation) measures: of all the
days the model called Up, what fraction actually went up? Recall for the Up class
(0.576) measures: of all the actual Up days, what fraction did the model catch?
These are distinct failure modes with different trading consequences. A false positive
(predict Up, market falls) produces a direct realized loss. A false negative (predict
Down, market rises) is a missed opportunity — capital that sat idle. Reporting both
allows a practitioner to choose the operating point depending on their risk tolerance.
F1 score is reported as the harmonic mean of precision and recall for each class,
giving equal weight to both error types, and the macro-average F1 (unweighted
average across the two classes) provides a single-number summary that is not
inflated by the majority class the way accuracy is.

**ROC-AUC.** The Receiver Operating Characteristic curve sweeps the decision
threshold over all values and plots the true-positive rate against the false-positive
rate. The area under this curve (AUC) measures whether the model's raw output —
the continuous predicted log return — provides a useful ranking of days from most
bearish to most bullish, irrespective of where the threshold is set. An AUC of 0.548
on the validation set confirms the predicted returns carry a weak but genuine
directional signal. Crucially, AUC above 0.5 means a higher threshold (e.g., only
trading when `|predicted_return| > 0.5%`) would yield better precision at the cost
of fewer trades — an operational lever that accuracy alone does not reveal.

**MSE and MAE as secondary regression metrics.** MSE and MAE capture whether the
model's predictions are calibrated in *magnitude*, not just in sign. A model that
always predicts ±0.001 when actual moves are ±0.020 can still call direction
correctly by chance, but would be useless for position sizing. The validation MAE
(0.00683) is 31% higher than the training MAE (0.00521), confirming that return
magnitude estimates degrade significantly out of sample — another dimension of
generalization failure beyond directional accuracy.

---

### Commentary on Observed Accuracy and Ideas for Improvement

#### What the numbers mean

The headline result is: **57.3% directional accuracy on training data, 54.1% on
validation data — a gap of 3.2 percentage points**. Both MSE and MAE are also
meaningfully higher on the validation set (+23% and +31% respectively). This
pattern is the textbook signature of **overfitting**: the model has partially
memorised patterns specific to the 2005–2018 training regime that do not generalise
cleanly to the 2019–2021 validation regime.

More importantly, the 54.1% validation accuracy must be placed in context:
the majority classifier (always predict Up) would score **56.9%** on the same
validation window, because 2019–2021 was an unusually strong bull market. The BiLSTM
therefore **does not outperform the trivial baseline on the validation set**. This is
a humbling result but an honest one, and it is not unusual in the financial ML
literature — markets are difficult precisely because of competition and the Efficient
Market Hypothesis. The model does beat random guessing (50%) and does have a positive
ROC-AUC (0.548), which means the signal exists but is too weak and too fragile across
regime shifts to generate a reliable directional edge at a fixed threshold.

#### Is this bad, and why does it happen?

It is not a model-breaking failure — it is the expected outcome for a single-modality
price-only model trained once on a fixed historical window. Three structural causes
drive the result:

**1. Non-stationarity and regime shift.** The training set (2005–2018) and
validation set (2019–2021) represent qualitatively different market environments.
Training covers the 2008 financial crisis, a slow 2010–15 recovery, and a late-cycle
expansion; validation contains the fastest equity crash in history (February–March
2020, −34% in 23 trading days) followed by an unprecedented stimulus-driven V-shaped
recovery. Patterns the LSTM learned from 2008–2009 volatility do not transfer to
COVID volatility because the *cause* and *duration* of the shock were completely
different. Financial data is fundamentally non-stationary: its statistical properties
change over time as macroeconomic regimes change, and any model trained statically on
one epoch will degrade as conditions shift.

**2. Overfitting from limited effective sample size.** The training set contains
3,490 calendar windows, but consecutive daily windows share 29 of 30 days of data —
they are almost entirely overlapping. The true number of *independent* observations
is far smaller, perhaps on the order of 3,490 / 30 ≈ 116 non-overlapping samples.
The BiLSTM has ~492,000 parameters, and optimising that many weights against the
equivalent of ~116 independent samples is a severe overparameterisation that
regularisation alone cannot fully cure.

**3. Price data alone is an incomplete information set.** By the Efficient Market
Hypothesis (semi-strong form), all public information — including historical prices —
is already reflected in current prices. A model that only sees price and volume
features is competing against the market's collective knowledge of those same
signals. The multimodal design of this project (adding news, chart images, and social
sentiment) was motivated precisely by this limitation.

#### Ideas for improvement

**1. Stronger regularisation and model compression.** The Part 3 interim report
already flagged that the BiLSTM overfits with hidden size 128 and two layers. For
a dataset with ~116 effective independent samples, the architecture is grossly
overparameterised. Reducing hidden size to 32–64, moving to a single LSTM layer,
and applying variational (locked) dropout — where the same dropout mask is applied
at every timestep rather than resampled — would dramatically reduce effective
parameter count and have been shown in the financial ML literature to substantially
reduce the train/val gap for time-series models.

**2. Expanding-window (walk-forward) retraining.** Training once on a fixed 14-year
window and then freezing the model for the entire validation period is the
single biggest methodological limitation of this submission. In production forecasting
systems and in most academic benchmarks, models are retrained on an *expanding*
window — every month, the new month's data is added to the training set and the model
is retrained from scratch or fine-tuned. This allows the model to continuously adapt
to new regimes. Had this been applied here, the model would have been updated with
2019 data before making 2020 predictions, giving it some exposure to post-2018
market conditions before the COVID shock.

**3. Complete the multimodal architecture.** The core hypothesis of this project —
that fusing price, news, chart images, and social sentiment produces a more robust
predictor than any single modality — was never tested because only the price branch
is integrated in the final model. News headlines carry forward-looking information
about Fed announcements, earnings guidance, and geopolitical events that appears
*before* prices move. Sentiment captures retail investor psychology that drove the
2020–21 meme-stock era in ways that price data cannot. Completing the FinBERT and
sentiment branches, and running the ablation study originally planned, is the
highest-leverage next step for improving validation accuracy. The late-fusion
architecture (described in Part 3) remains intact and ready to be completed.

**4. Temporal self-attention over the 30-day window.** The current model discards all
intermediate BiLSTM hidden states and uses only the final state, losing information
about *which* days in the 30-day window were most influential. A multi-head
self-attention layer applied over the full sequence of hidden states would allow the
model to selectively up-weight the days most predictive of the next-day move (e.g.,
a day with an extreme VIX spike or a Fed rate decision). In NLP, attention
consistently outperforms last-hidden-state pooling for sequence classification;
the same logic applies here, where not all 30 days in a window are equally relevant.

**5. Regime detection as an auxiliary input.** Including an explicit market-regime
indicator as an additional feature — for example, a binary VIX-above-25 flag, a
Hidden Markov Model regime label, or a simple 200-day moving-average trend filter —
would give the model explicit information about the current volatility environment.
The 2020 COVID crash period had VIX readings of 40–85 (versus a 2014–2019 average
of ~14), a regime the model had never been told to treat differently. With regime
awareness, the model could learn separate behaviour for high-volatility states and
generalise better to the COVID period.

**6. Confidence-gated trading signal.** The ROC-AUC of 0.548 means the model's
predicted return is a weak but non-trivial ranking of days by bullishness. Rather
than acting on every day's prediction, only making a directional bet when
`|predicted_return| > threshold` — filtering out low-confidence days — trades
coverage for precision. On the validation set, the model's correct calls are likely
concentrated among days where it predicts a large return magnitude; capturing only
those days would improve effective precision even without changing the model
architecture.

#### Summary

The BiLSTM model demonstrates that a pure price-based time-series model captures
a real but fragile directional signal in S&P 500 data: 57.3% accuracy on the training
set, 54.1% on the validation set, and a positive ROC-AUC of 0.548 on held-out data.
The 3.2-point train/val gap is a direct symptom of overfitting to the 2005–2018
regime and the model's inability to adapt to the COVID shock, which represents a
market regime with no historical analogue in the training data. The validation
accuracy does not surpass the trivial majority-class baseline (56.9%), which
underscores the challenge of price-only prediction on daily timescales and the
importance of completing the multimodal architecture. Walk-forward retraining,
model compression, temporal attention, and the addition of forward-looking news and
sentiment signals are the four highest-leverage improvements identified for closing
this gap.
