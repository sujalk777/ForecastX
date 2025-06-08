# app.py (Streamlit version)
import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Set page config
st.set_page_config(page_title="Tata Motors Stock Price Prediction", layout="wide")

# Title
st.title("Tata Motors Stock Price Prediction")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Function to get user input
def get_input():
    start_date = st.sidebar.date_input("Start date", datetime.date(2010, 1, 1))
    end_date = st.sidebar.date_input("End date", datetime.date.today())
    return start_date, end_date

start, end = get_input()

# Function to get the company name
def get_company_name():
    return 'TATAMOTORS.NS'

# Function to get the company data
@st.cache_data
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Get the data
company = get_company_name()
data = get_data(company, start, end)

# Display the data
st.subheader(f"{company} Stock Data from {start} to {end}")
st.write(data.describe())

# Visualization
st.subheader("Closing Price vs Time")
fig = plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
st.pyplot(fig)

# Preprocess data for LSTM
def preprocess_data(data, look_back=60):
    # Use only the 'Close' column
    dataset = data['Close'].values
    dataset = dataset.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create training data
    x_train = []
    y_train = []
    
    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, dataset

# Build LSTM model
def build_lstm_model(input_shape):
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

# Train model and make predictions
def predict_future_prices(data, days=30):
    look_back = 60
    x_train, y_train, scaler, dataset = preprocess_data(data, look_back)
    
    # Build and train model
    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=25, batch_size=32)
    
    # Prepare test data
    last_sequence = dataset[-look_back:]
    last_sequence_scaled = scaler.transform(last_sequence)
    
    predictions = []
    
    for _ in range(days):
        x_test = np.array([last_sequence_scaled[-look_back:, 0]])
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Predict next day
        pred_price = model.predict(x_test)
        
        # Update sequence
        last_sequence_scaled = np.append(last_sequence_scaled, pred_price)
        last_sequence_scaled = last_sequence_scaled.reshape(-1, 1)
        
        # Inverse transform to get actual price
        pred_price = scaler.inverse_transform(pred_price)
        predictions.append(pred_price[0, 0])
    
    return predictions

if st.button("Predict Next 30 Days"):
    with st.spinner('Training model and making predictions...'):
        predictions = predict_future_prices(data)
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(last_date, periods=31, freq='B')[1:]
        
        # Plot predictions
        st.subheader("Predicted Prices for Next 30 Days")
        fig = plt.figure(figsize=(12, 6))
        plt.plot(data.index[-100:], data['Close'][-100:], label='Historical Prices')
        plt.plot(future_dates, predictions, label='Predicted Prices', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.legend()
        st.pyplot(fig)
        
        # Show predictions in a table
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': predictions
        })
        st.write(pred_df)
