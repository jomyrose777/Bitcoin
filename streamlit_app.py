import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Fetch live data from Yahoo Finance
data = yf.download(ticker, period='1d', interval='1m')

# Calculate technical indicators without ta-lib
data['RSI'] = data['Close'].rolling(window=14).apply(lambda x: (x/x.shift(1)-1).mean())
data['BB_Middle'] = data['Close'].rolling(window=20).mean()
data['BB_Upper'] = data['BB_Middle'] + 2*data['Close'].rolling(window=20).std()
data['BB_Lower'] = data['BB_Middle'] - 2*data['Close'].rolling(window=20).std()
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['Stoch_OSC'] = (data['Close'] - data['Close'].rolling(window=14).min()) / (data['Close'].rolling(window=14).max() - data['Close'].rolling(window=14).min())
data['Force_Index'] = data['Close'].diff() * data['Volume']

# Perform sentiment analysis using nltk
sia = SentimentIntensityAnalyzer()
data['Sentiment'] = data['Close'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# Define machine learning model using scikit-learn
X = pd.concat([data['Close'], data['RSI'], data['BB_Middle'], data['BB_Upper'], data['BB_Lower'], data['MACD'], data['Stoch_OSC'], data['Force_Index'], data['Sentiment']], axis=1)
y = data['Close'].shift(-1).dropna()

# Remove NaN values from y and update X accordingly
y = data['Close'].shift(-1).dropna()
X = X.iloc[:-1]  # Remove the last row from X to match y

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define customizable parameters using streamlit
st.title('Bitcoin Model with Advanced Features')
st.write('Select parameters:')
n_estimators = st.slider('n_estimators', 1, 100, 50)
rsi_period = st.slider('RSI period', 1, 100, 14)
bb_period = st.slider('BB period', 1, 100, 20)
sentiment_threshold = st.slider('Sentiment threshold', -1, 1, 0)

from sklearn.ensemble import RandomForestRegressor

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Generate predictions
signals = model.predict(X_test)

# Display buy/sell signals in Streamlit
st.write('Buy/Sell Signals:')
for signal in signals:
    if signal > 0:
        st.write('**BUY**')
    else:
        st.write('**SELL**')

# Plot the price chart
st.line_chart(data['Close'])

# Open the Streamlit app in the default web browser
import webbrowser
url = "http://localhost:8501"
webbrowser.open(url)
