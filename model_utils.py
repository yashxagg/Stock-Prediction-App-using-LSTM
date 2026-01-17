
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def create_lstm_model(input_shape):
    """
    Creates and compiles an LSTM model.
    """
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(model, X_train, y_train, epochs=25, batch_size=32):
    """
    Trains the LSTM model.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def predict_future(model, last_sequence, scaler, days=30):
    """
    Predicts future stock prices.
    """
    curr_sequence = last_sequence.copy()
    future_predictions = []
    
    for _ in range(days):
        # Predict next value
        # Reshape to [1, sequence_length, 1]
        input_seq = curr_sequence.reshape(1, -1, 1)
        pred = model.predict(input_seq)
        
        future_predictions.append(pred[0, 0])
        
        # Update sequence: remove first, add prediction
        curr_sequence = np.append(curr_sequence[1:], pred)
        
    # Inverse transform predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions
