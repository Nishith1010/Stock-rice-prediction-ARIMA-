# Complete ARIMA Stock Price Prediction with Proper Date Handling
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_stock_data(ticker='AAPL', start='2020-01-01', end='2023-12-31'):
    """Fetch stock data and ensure proper datetime frequency"""
    data = yf.download(ticker, start=start, end=end, progress=False)['Close']
    
    # Convert to business day frequency and handle missing values
    data = data.asfreq('B').ffill()
    print(f"\nData frequency set to: {data.index.freq}")
    return data
def check_stationarity(series):
    """Check stationarity using ADF test"""
    result = adfuller(series.dropna())
    print(f"\nADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    return result[1] <= 0.05
def prepare_data(series):
    """Make data stationary and return differencing order"""
    d = 0
    while not check_stationarity(series):
        series = series.diff().dropna()
        d += 1
        print(f"Applied differencing order {d}")
    return series, d

def train_arima(train_data, order=(1,1,1)):
    """Train ARIMA model with proper date handling"""
    model = ARIMA(train_data, 
                order=order,
                freq=train_data.index.freq)
    return model.fit()

def plot_results(train, test, forecast, title):
    """Visualize actual vs predicted prices"""
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Prices', color='green')
    plt.plot(test.index, forecast, label='Forecasted Prices', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()
