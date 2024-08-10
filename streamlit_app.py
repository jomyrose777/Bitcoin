import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime
import plotly.graph_objects as go
import logging
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure NLTK data is available
def ensure_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

# Ensure NLTK data is downloaded
ensure_nltk_data()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    return dt.tz_convert(est) if dt.tzinfo else est.localize(dt)

# Fetch live data from Yahoo Finance with retries
@st.cache
def fetch_data():
    retries = 3
    for attempt in range(retries):
        try:
            data = yf.download(ticker, period='1d', interval='1m')
            if data.index.tzinfo is None:
                data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
            else:
                data.index = data.index.tz_convert(est)
            return data
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            time.sleep(5)  # Wait before retrying
    st.error("Error fetching data after multiple attempts. Please try again later.")
    return pd.DataFrame()  # Return an empty DataFrame in case of error

# Fetch data
data = fetch_data()
if data.empty:
    st.stop()

# Calculate technical indicators
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
data['MACD'] = ta.trend.MACD(data['Close']).macd()
data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
data['BULLBEAR'] = data['Close']  # Placeholder for sentiment data
data['UO'] = data['Close']  # Placeholder for UO data
data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()

# Drop rows with NaN values
data.dropna(inplace=True)

# Calculate Fibonacci retracement levels
def fibonacci_retracement(high, low):
    diff = high - low
    levels = [high - diff * ratio for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]]
    return levels

high = data['High'].max()
low = data['Low'].min()
fib_levels = fibonacci_retracement(high, low)

# Detect Doji candlestick patterns
def detect_doji(data):
    threshold = 0.001
    data['Doji'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low']) < threshold
    return data

data = detect_doji(data)

# Calculate support and resistance levels
def calculate_support_resistance(data, window=5):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

data = calculate_support_resistance(data)

# Add chart to display support and resistance levels
st.title('Bitcoin Technical Analysis and Signal Summary')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
fig.add_trace(go.Scatter(x=data.index, y=data['Support'], name='Support', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=data.index, y=data['Resistance'], name='Resistance', line=dict(dash='dash')))
fig.update_layout(title='Support and Resistance Levels', xaxis_title='Time', yaxis_title='Price')
st.plotly_chart(fig)

# Fetch and analyze Fear and Greed Index
def fetch_fear_and_greed_index():
    # Replace with actual implementation to fetch the Fear and Greed Index
    return np.random.randint(0, 100)  # Placeholder for actual index value

fear_and_greed_index = fetch_fear_and_greed_index()

# Generate summary of technical indicators
def technical_indicators_summary(data):
    indicators = {
        'RSI': data['RSI'].iloc[-1],
        'STOCH': data['STOCH'].iloc[-1],
        'MACD': data['MACD'].iloc[-1] - data['MACD_Signal'].iloc[-1],
        'ADX': data['ADX'].iloc[-1],
        'CCI': data['CCI'].iloc[-1],
        'BULLBEAR': data['BULLBEAR'].iloc[-1],
        'UO': data['UO'].iloc[-1],
        'ROC': data['ROC'].iloc[-1],
        'WILLIAMSR': data['WILLIAMSR'].iloc[-1],
        'Fear and Greed Index': fear_and_greed_index
    }
    return indicators

indicators = technical_indicators_summary(data)

# Generate summary of moving averages
def moving_averages_summary(data):
    ma = {
        'MA5': data['Close'].rolling(window=5).mean().iloc[-1],
        'MA10': data['Close'].rolling(window=10).mean().iloc[-1],
        'MA20': data['Close'].rolling(window=20).mean().iloc[-1],
        'MA50': data['Close'].rolling(window=50).mean().iloc[-1],
        'MA100': data['Close'].rolling(window=100).mean().iloc[-1],
        'MA200': data['Close'].rolling(window=200).mean().iloc[-1]
    }
    return ma

moving_averages = moving_averages_summary(data)

# Generate buy/sell signals based on indicators and moving averages
def generate_signals(indicators, moving_averages, data):
    signals = {}
    signals['timestamp'] = to_est(data.index[-1]).strftime('%Y-%m-%d %I:%M:%S %p')
    
    # RSI Signal
    if indicators['RSI'] < 30:
        signals['RSI'] = 'Buy'
    elif indicators['RSI'] > 70:
        signals['RSI'] = 'Sell'
    else:
        signals['RSI'] = 'Neutral'
    
    # MACD Signal
    if indicators['MACD'] > 0:
        signals['MACD'] = 'Buy'
    else:
        signals['MACD'] = 'Sell'
    
    # ADX Signal
    if indicators['ADX'] > 25:
        signals['ADX'] = 'Buy'
    else:
        signals['ADX'] = 'Neutral'
    
    # CCI Signal
    if indicators['CCI'] > 100:
        signals['CCI'] = 'Buy'
    elif indicators['CCI'] < -100:
        signals['CCI'] = 'Sell'
    else:
        signals['CCI'] = 'Neutral'
    
    # Moving Averages Signal
    signals['MA'] = 'Buy' if moving_averages['MA5'] > moving_averages['MA10'] else 'Sell'
    
    # Fear and Greed Signal
    if indicators['Fear and Greed Index'] > 50:
        signals['Fear and Greed'] = 'Buy'
    else:
        signals['Fear and Greed'] = 'Sell'
    
    return signals

signals = generate_signals(indicators, moving_averages, data)

# Log signals
log_file = 'signals_log.csv'
try:
    logs = pd.read_csv(log_file)
except FileNotFoundError:
    logs = pd.DataFrame(columns=['timestamp', 'RSI', 'MACD', 'ADX', 'CCI', 'MA', 'Fear and Greed'])

new_log = pd.DataFrame([signals])
logs = pd.concat([logs, new_log], ignore_index=True)
logs.to_csv(log_file, index=False)

# Display the indicators and signals
st.subheader('Technical Indicators Summary')
st.write(indicators)

st.subheader('Moving Averages Summary')
st.write(moving_averages)

st.subheader('Generated Signals')
st.write(signals)
