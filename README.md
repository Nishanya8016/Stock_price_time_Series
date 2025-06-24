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

## 📅 Dataset
Source: Nifty 50 Stock Market Data (Kaggle)

Stock: RELIANCE

Columns: Date, Open, High, Low, Close, Volume, etc.

Frequency: Daily

Range: Jan 2000 – April 2021
---

## 🗂️ Project Structure

```bash
├── notebooks/
│   ├── 01_arima_init.ipynb         # Classical ARIMA modeling
│   ├── 02_garch_init.ipynb        # GARCH for volatility prediction
│   ├── 03_prophet_init.ipynb      # Classical PROPHET modeling
│   ├── 04_tcn_init.ipynb       # Deep learning: TCN
│   ├── 05_time_series_transformer_init.ipynb     #Deep learning: Transformer
│   ├── 06_rnn_init.ipynb              # RNN model
│   ├── 07_lstm_init.ipynb         # LSTM model
├── data/
│   └── RELIANCE.csv                   # Stock data (source: Kaggle Nifty 50)
├── README.md
└── requirements.txt
```
# 📈 Time Series Stock Price Prediction

This project focuses on comparing various time series forecasting models using the **NIFTY 50 stock market dataset**, specifically for predicting the stock prices of **Reliance**.

## 🧪 Models & Evaluation Metrics

Below are the performance results of different models based on **Accuracy**, **RMSE (Root Mean Squared Error)**, **MAE (Mean Absolute Error)**, **MSE (Mean Squared Error)** and **F1 Score**:

| Model                    | Accuracy |   RMSE    |    MAE   |    MSE   |    F1   |
|--------------------------|----------|-----------|----------|----------|---------|
| ARIMA                    | 0.5080   | 93.6875   | 40.2438  | 8777.347 | 0.5000  |
| Prophet (by Meta)        | 0.5333   | 210.6119  | 210.6119 | 44357.246| 0.5333  |
| TCN                      | 0.5172   | 67.3219   | 67.3219  | 4532.238 | 0.5172  |
| Time-Series Transformer  | 0.6522   | 0.5094    | 0.4079   | 0.2594   |         |
| RNN                      |  0.5138  | 39.8871   | 21.9619  | 1590.9779| 0.6826  |
| LSTM                     | 0.5033   | 70.9129   | 46.2609  | 5028.639 | 0.3336  |

GARCH was only used to find Volatility, not for prediction of Closing price

| Model                    | Accuracy |   RMSE    |    MAE   |    MSE   |
|--------------------------|----------|-----------|----------|----------|
| GARCH                    | 0.2066   | 0.0205    | 0.0138   | 0.0004   |

## 🔍 Summary

- **Best Accuracy**: Achieved by the **Time-Series Transformer** (	0.6522 ).
- **Lowest RMSE**: Achieved by the **GARCH** model (0.0205), showing precise predictions despite its lower accuracy.
- **Most balanced performance**: **TCN** provided a stable middle ground with decent accuracy and RMSE.

## 📦 Installation

Install using:

```bash
pip install -r requirements.txt
```

## 📊 Dataset
The dataset used is from [Kaggle - Nifty 50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data), focusing on `RELIANCE.csv`.

## 🚀 Future Improvements
- Hyperparameter tuning for TCN and Transformer models.
- Experimenting with hybrid models combining ARIMA + Neural Nets.
- Using rolling-window and ensemble predictions for more stability.

