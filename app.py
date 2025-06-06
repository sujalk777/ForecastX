import streamlit as st
import pandas as pd
import yfinance as yf
import  matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model

st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
st.title('stock price prediction using LSTM')
st.write('This project is about predicting the stock price of a TATA MOTORS using LSTM')
st.write('The data is taken from yahoo finance')
st.write('The data is from 2010-01-01 to 2024-10-12')
st.write('The model is trained using LSTM')
ticker= 'TATAMOTORS.NS'
data=yf.download(ticker,start='2010-01-01',end='2024-10-12')
data.reset_index(inplace=True)
df=data
#plotly chart of tata motors
st.write('The stock price of TATA MOTORS')
fig=px.line(df,x='Date',y='Close')
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(width=800,height=500)
st.plotly_chart(fig)
st.write('The data is taken from yahoo finance')
#ploting moving average of 100 days
st.subheader('The moving average of 100 days')
ma100=df.Close.rolling(100).mean()
#ploting moving average of 100 day with closing price using matplotlib
plt.figure(figsize=(12,6))
plt.plot(df.Date,df.Close,label='TATA MOTORS')
plt.plot(df.Date,ma100,label='Moving Average 100 days')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
st.pyplot()
#ploting moving average of 200 days
st.subheader('The moving average of 200 days')
ma200=df.Close.rolling(200).mean()
#ploting moving average of 200 day with closing price using matplotlib
plt.figure(figsize=(12,6))
plt.plot(df.Date,df.Close,label='TATA MOTORS')
plt.plot(df.Date,ma100,label='Moving Average 100 days')
plt.plot(df.Date,ma200,label='Moving Average 200 days')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
st.pyplot()
# spliting the data into training and testing
train =pd.DataFrame(data[0:int(len(data)*0.70)])
test=pd.DataFrame(data[int(len(data)*0.70):int(len(data))])
scaler= MinMaxScaler(feature_range=(0, 1))
train_close=train.iloc[:,4:5].values
test_close=test.iloc[:,4:5].values
data_training_array=scaler.fit_transform(train_close)
data_testing_array=scaler.transform(test_close)
X_train=[]
y_train=[]
for i in range(100,data_training_array.shape[0]):
    X_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)
#model is imported

model=load_model('stock_prediction.h5')

#testing
past_100_days = pd.DataFrame(train_close[-100:])
test_df = pd.DataFrame(test_close)
final_df = pd.concat([past_100_days, test_df], ignore_index=True)
input_data=scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
   x_test.append(input_data[i-100: i])
   y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
scaling_factor_number=scaler.scale_[0]
scale_factor=1/scaling_factor_number
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor
#ploting the predicted stock price
st.write('The predicted stock price of TATA MOTORS')
plt.figure(figsize=(12,6))
plt.plot(y_test, color = 'red', label = 'Real TATA MOTORS Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted TATA MOTORS Stock Price')
plt.title('TATA MOTORS Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA MOTORS Stock Price')
plt.legend()
plt.show()
st.pyplot()








