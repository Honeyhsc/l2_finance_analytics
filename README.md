
# Limit Order Book (LOB) and Trade Data: Exploratory Analysis and Predictive Modeling

This repository contains a comprehensive analysis of high-frequency trading data using Limit Order Book (LOB) and Trade datasets, with a focus on identifying predictive signals to inform algorithmic trading strategies.

## Project Goals

- Develop a deep understanding of trading behavior and market structure using Level 2 (LOB) data.
- Explore trade and order book features to extract actionable patterns.
- Engineer features such as liquidity, imbalance, mid-price, slippage, and trade intensity.
- Evaluate predictive value of features using statistical analysis and visual diagnostics.
- Build toward time-series models (e.g., LSTM) for price movement prediction.

---

## Key Exploratory Insights

### Market Trend Summary

- Analyzed OHLC prices, spread, and traded volume over 6 months.
- Identified a **liquidity crisis** emerging post-February, marked by widening spreads and falling prices.
- Increased volumes with declining prices suggest panic selling.
- Classification of trades using Lee-Ready algorithm does show 66% sells made post February.

### Spread & Sentiment

- Spread widened as confidence dropped — indicating increasing execution risk.
- Introduced concept: even with directional prediction, wider spreads lead to slippage and poor fill quality.

### Trade Activity Patterns

- Inter-trade durations follow a **right-skewed distribution**, approximating an **exponential** process.
- Observed potential for **Poisson process modeling** (e.g., ACD models).
- Deviations from exponential may signal regime change or actionable market behavior.

### Imbalance Analysis

- Defined imbalance using top-5 and top-10 LOB levels:
  ```python
  imbalance = total_bid_volume / (total_bid_volume + total_ask_volume)
  ```
- Found mean imbalance ~0.7 — indicating a bid-heavy book, but price still declined → **imbalance may lag price**.
- Investigated anomaly: low imbalance bins show positive future return — likely due to rebound after liquidity shock.
- Explored this through return vs. imbalance bin correlation, lag effects, and distributional shifts.

---

## Feature Engineering

Engineered the following features from combined Trade and LOB snapshots:

- `mid_price`, `spread`, `price_ffill`
- `best_bid`, `best_ask`, `top1_bid_vol`, `top1_ask_vol`
- `total_bid_vol`, `total_ask_vol`, `imbalance`
- `slippage`: difference between trade and mid-price
- Inter-trade intervals and volume-based burst detection
- liquidity as a function of top five bids and ask and spread

---

## Visual Diagnostics

- Time series of spread, OHLC, and volume
- Imbalance vs. return heatmaps and correlation trends
- Violin plots for bid/ask depth distributions
- Distribution fits and Q-Q plots for inter-trade durations

---

## Modeling (In Progress)

- Goal: Use engineered features to train **directional price prediction models**.
- LSTM on LOB snapshots with directional loss (cosine + MSE)
- Explore classification vs regression modeling of returns
- Evaluate lead-lag behavior of imbalance and other microstructure features

---

## Tech Stack

- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- PyTorch (for LSTM models)
- Jupyter Notebooks for EDA and experimentation

---

## 📂 Structure

```
📁 data/               # Raw and preprocessed daily LOB and Trade files
📁 notebooks/          # Jupyter notebooks for EDA, visualization, modeling
📁 src/                # Feature extraction, data loaders, modeling scripts
📄 README.md
```

---

## 🔍 Notebooks Overview

- `01_eda_overview.ipynb`: Market-wide summaries, OHLC, spread analysis
- `02_eda_trade_activity.ipynb`: Inter-trade durations, Poisson fit, bursts
- `03_eda_lob_features_correlation.ipynb`: Overall LoB Features and their return analysis across time horizons
- `04_lstm_modeling.ipynb`: Feature preparation, LSTM training with directional loss

---

## Future Work

- Lagged feature dynamics and cross-correlation studies
- Real-time signal simulation using rolling prediction windows
- Trade classification using Lee-Ready algorithm
- Volume and slippage-aware order execution modeling

---

## Contact

For questions, collaborations, or feedback, feel free to open an issue or reach out via email.

---

> "The challenge isn't just in predicting where the price is going, but in knowing how confidently and cheaply you can act on that prediction."
