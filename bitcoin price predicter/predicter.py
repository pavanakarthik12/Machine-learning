import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tkinter import *
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ====== REAL-TIME DATA FETCH (from CoinGecko) ======
def fetch_crypto_data(days=30):
    url = f'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}'
    response = requests.get(url)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['day'] = (df['date'] - df['date'].min()).dt.days + 1
    return df[['day', 'price']]

# ====== LINEAR REGRESSION MODEL ======
def linear_regression_predict(df):
    X = df[['day']].values
    y = df['price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    slope, intercept = model.coef_[0], model.intercept_
    next_day = np.array([[df['day'].max() + 1]])
    predicted_price = model.predict(next_day)[0]
    r_squared = model.score(X_test, y_test)
    return slope, intercept, predicted_price, r_squared

# ====== LSTM MODEL ======
def lstm_predict(df):
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(df['price'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(3, len(scaled_prices)):
        X.append(scaled_prices[i-3:i])
        y.append(scaled_prices[i])
    X, y = np.array(X), np.array(y)
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=8, verbose=0)
    last_sequence = scaled_prices[-3:].reshape(1, 3, 1)
    predicted_scaled = model.predict(last_sequence)[0][0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
    return predicted_price

# ====== GUI INTERFACE ======
def run_gui():
    window = Tk()
    window.title("Crypto Price Predictor")
    window.geometry("400x300")

    label = Label(window, text="Fetching Bitcoin price data...", font=("Arial", 12))
    label.pack(pady=10)

    df = fetch_crypto_data(30)
    slope, intercept, pred_lr, r_squared = linear_regression_predict(df)
    pred_lstm = lstm_predict(df)

    result_text = f"""
    Linear Regression:
    Slope: {slope:.2f}
    Intercept: {intercept:.2f}
    Predicted Price (Day {df['day'].max() + 1}): ${pred_lr:.2f}
    R² Score: {r_squared:.2f}

    LSTM Prediction:
    Predicted Price (Next Day): ${pred_lstm:.2f}
    """

    result_label = Label(window, text=result_text, font=("Arial", 10), justify=LEFT)
    result_label.pack(pady=10)

    # Plot
    def plot_data():
        plt.scatter(df['day'], df['price'], label='Actual')
        plt.plot(df['day'], slope*df['day'] + intercept, color='red', label='Linear Regression')
        plt.title('Bitcoin Price Prediction')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    plot_button = Button(window, text="Show Plot", command=plot_data)
    plot_button.pack(pady=10)

    window.mainloop()

# Run the GUI
run_gui()
