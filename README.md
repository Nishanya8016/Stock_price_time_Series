# Stock_price_time_Series
# 📈 Temporal Forecasting of Stock Price, Movement, and Volatility

This project compares classical and deep learning time-series models for predicting next-day **stock close price**, **direction of movement**, and **realized volatility**. The dataset used is daily stock data for **RELIANCE** (from the NIFTY 50 index), spanning 2000 to 2021.

---

## 🚀 Project Objectives

- 📊 Predict next-day **closing price** (regression)
- 🔁 Predict next-day **price direction** (classification)
- 📉 Predict next-day **realized volatility** (regression)
- 🔍 Compare model performance across:
  - ARIMA
  - GARCH
  - PROPHET
  - RNN
  - LSTM
  - TCN (Temporal Convolutional Network)
  - Time Series Transformer

---

## 🗂️ Project Structure

```bash
├── notebooks/
│   ├── 01_preprocessing.ipynb         # Data cleaning, feature engineering
│   ├── 02_arima_reliance.ipynb        # Classical ARIMA modeling
│   ├── 03_garch_init.ipynb      # GARCH for volatility prediction
│   ├── 04_rnn_lstm.ipynb              # RNN and LSTM models
│   ├── 05_tcn_transformer.ipynb       # Deep learning: TCN and Transformer
│   └── 06_results_comparison.ipynb    # Evaluation and visualization
├── data/
│   └── RELIANCE.csv                   # Stock data (source: Kaggle Nifty 50)
├── results/
│   └── plots/                         # Prediction vs actual plots
│   └── metrics.csv                    # Evaluation metrics for all models
├── README.md
└── requirements.txt
