# Semester Project: Multimodal Neural Network for S&P 500 Return Prediction
**CSE-60868 Neural Networks | University of Notre Dame | Spring 2026**

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
