
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(ticker, start, end):
    """
    Fetches stock data from Yahoo Finance.
    """
    if not ticker.endswith('.NS'):
        ticker = f"{ticker}.NS"
    
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        return None
    return data

def preprocess_data(data, sequence_length=60):
    """
    Preprocesses data for LSTM model.
    Returns: X_train, y_train, scaler
    """
    # Use 'Close' price for prediction
    # Handle multi-level column index if present (common in new yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        dataset = data['Close'].values
    else:
        dataset = data['Close'].values
        
    dataset = dataset.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    X_train, y_train = [], []
    
    for i in range(sequence_length, len(scaled_data)):
        X_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, scaler

def create_sequences(data, sequence_length=60):
    """Create sequences for inference"""
    X = []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:(i + sequence_length)])
    return np.array(X)
