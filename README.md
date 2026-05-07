Stock Price Forecasting using LSTM and GRU Neural Networks

![Python](https://img.shields.io/badge/Python-3.x-blue)  ![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-blue)  ![Task](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-blue)  
## Overview
Develops and compares Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) deep learning models for time series stock price forecasting. Applies a sliding window approach, MinMax normalisation, dropout regularisation, and evaluates using RMSE.
## Problem Statement
Predict future stock closing prices from historical price sequences using recurrent neural networks. This project demonstrates the ability to process sequential time-series data and capture long-range temporal dependencies — a core skill in financial data science and signal processing.


## Dataset

- **Name:** Apple Inc. (AAPL) Historical Stock Price Data
- **Source:** [Download from Kaggle](https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data)
  or download directly using the `yfinance` library (see How to Run)
- **Ticker Symbol:** AAPL (Apple Inc.)
- **Feature Used:** `Close` — daily closing price only
- **Format:** CSV file with columns: 
  `Date, Open, High, Low, Close, Adj Close, Volume`


**Download automatically using yfinance (recommended):**
```python
  import yfinance as yf
  df = yf.download('AAPL', start='2010-01-01', end='2023-01-01')
  df.to_csv('AAPL_stock_data.csv)
 ```

## Approach
-	Applied MinMax scaling to normalise stock price data into the [0, 1] range for improved model convergence.
-	Engineered sequential input data using a 60-timestep sliding window: each input is the previous 60 days of closing prices, and the target is the next day's price.
-	Designed multi-layer LSTM architecture with Dropout regularisation to reduce overfitting.
-	Designed equivalent multi-layer GRU architecture for direct comparison.
-	Trained both models using Adam optimiser and Mean Squared Error loss.
-	Evaluated both models using RMSE on the held-out test set.
-	Visualised predicted vs actual stock prices on a line chart to assess model performance.
## Results

Property	Details
Models	LSTM vs GRU (direct comparison)
Input Window	60 timesteps (60 days of historical prices)
Normalisation	MinMax Scaling (sklearn)
Regularisation	Dropout layers
Evaluation Metric	RMSE (Root Mean Squared Error)
Optimiser	Adam

## Technologies
Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-Learn
## How to Run

```bash
git clone https://github.com/OyelolaIbrahim/stock-price-forecasting-lstm-gru.git
cd stock-price-forecasting-lstm-gru
pip install tensorflow numpy pandas matplotlib scikit-learn
jupyter notebook lstm_gru_stock_forecasting.ipynb
```

