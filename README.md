

# Stock Price Prediction Using LSTM



https://github.com/user-attachments/assets/82af8241-ec86-4c27-b30b-57641a5ab5e7






This project predicts the stock prices of **Tata Motors** using historical stock price data from **2010 to present**. The model is built using **Long Short-Term Memory (LSTM)**, a type of recurrent neural network (RNN), and is implemented in **TensorFlow** with GPU acceleration.

## Project Overview

The goal of this project is to create a model that can predict the **closing price** of Tata Motors based on past stock prices. The dataset used is sourced from **Yahoo Finance**, containing daily stock prices for Tata Motors.

The model leverages **LSTM layers** to capture the temporal dependencies in stock prices and make future predictions. The dataset is preprocessed, normalized, and split into training and testing sets. The model's performance is evaluated using mean squared error (MSE).

---

## Project Structure

```
Stock_Price_Prediction/
│
├── data/
│   └── tata_motors_stock_data.csv   # Historical stock price data (2010-present)
│
├── src/
│   ├── data_preprocessing.py        # Script for data loading and preprocessing
│   ├── model_training.py            # LSTM model definition and training code
│   ├── model_evaluation.py          # Script to evaluate model performance
│   └── utils.py                     # Helper functions (scaling, plotting, etc.)
│
├── notebooks/
│   └── Stock_Price_Prediction.ipynb # Jupyter notebook with step-by-step explanation
│
├── README.md                        # Project readme file
├── requirements.txt                 # List of dependencies for the project
└── lstm_stock_price_prediction.py   # Main script to train and test the model
```

---

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/Stock_Price_Prediction.git
cd Stock_Price_Prediction
```

### 2. Install dependencies:

Use the `requirements.txt` file to install necessary Python packages.

```bash
pip install -r requirements.txt
```

### 3. Download Data:

You can download historical stock price data for Tata Motors from [Yahoo Finance](https://finance.yahoo.com/quote/TATAMOTORS.NS/history/), or use the provided dataset in the `data` folder.

---

## Model Architecture

The LSTM model is designed with the following architecture:
- 4 stacked LSTM layers with increasing units (50, 60, 80, 120).
- **Dropout** layers added between LSTM layers to prevent overfitting.
- A **Dense** layer as the output, predicting the next stock price.

```python
model = Sequential()

# Layer 1
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(100, 1)))
model.add(Dropout(0.2))

# Layer 2
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

# Layer 3
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

# Layer 4
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
```

---

## Usage

### 1. Data Preprocessing

Before training the model, we preprocess the data:
- **Normalization**: Using MinMaxScaler to scale stock prices between 0 and 1.
- **Sliding Window**: Creating sequences of 100 days of stock prices to predict the next day's price.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Creating training data with sliding windows
x_train = []
y_train = []

for i in range(100, len(scaled_data)):
    x_train.append(scaled_data[i-100:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
```

### 2. Training the Model

You can train the model by running the script:

```bash
python lstm_stock_price_prediction.py
```

The model will be trained on the past stock prices and will predict future values.

### 3. Evaluate the Model

Once the model is trained, it can be evaluated on the test data:

```python
model.evaluate(x_test, y_test)
```

You can also plot the predicted vs actual stock prices to visualize the model’s performance.

---

## GPU Acceleration

To accelerate the training, the model uses your GPU. Ensure that **CUDA** and **cuDNN** are installed, and the TensorFlow version is configured for GPU.

You can check if TensorFlow detects your GPU by running:

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

---

## Results

The model’s performance is evaluated based on Mean Squared Error (MSE), and the predicted stock prices are plotted alongside the actual stock prices for comparison.

---


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



