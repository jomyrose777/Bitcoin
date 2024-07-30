import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pytz
from datetime import datetime
import scipy.stats as stats

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    return dt.tz_convert(est) if dt.tzinfo else est.localize(dt)

# Fetch live data from Yahoo Finance
data = yf.download(ticker, period='1d', interval='1m')

# Convert index to EST if it's not already timezone-aware
if data.index.tzinfo is None:
    data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
else:
    data.index = data.index.tz_convert(est)

# Calculate technical indicators
data['RSI'] = data['Close'].rolling(window=14).apply(lambda x: (x/x.shift(1)-1).mean(), raw=False)
data['BB_Middle'] = data['Close'].rolling(window=20).mean()
data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['Stoch_OSC'] = (data['Close'] - data['Close'].rolling(window=14).min()) / (data['Close'].rolling(window=14).max() - data['Close'].rolling(window=14).min())
data['Force_Index'] = data['Close'].diff() * data['Volume']

# Perform sentiment analysis using nltk
sia = SentimentIntensityAnalyzer()
data['Sentiment'] = data['Close'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# Drop rows with NaN values
data.dropna(inplace=True)

# Advanced Analysis Functions

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
    threshold = 0.001  # Define a threshold for identifying Doji
    data['Doji'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low']) < threshold
    return data

data = detect_doji(data)

# Calculate support and resistance levels
def calculate_support_resistance(data, window=5):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

data = calculate_support_resistance(data)

# Generate summary of technical indicators
def technical_indicators_summary(data):
    indicators = {
        'RSI': data['RSI'].iloc[-1],
        'STOCH': data['Stoch_OSC'].iloc[-1],
        'MACD': data['MACD'].iloc[-1],
        'ADX': data['Force_Index'].iloc[-1],
        'CCI': data['Force_Index'].iloc[-1],
        'BULLBEAR': data['Sentiment'].iloc[-1],
        'UO': data['Sentiment'].iloc[-1],
        'ROC': data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1,
        'WILLIAMSR': (data['High'].rolling(window=14).max() - data['Close']) / (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())
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
    
    if indicators['RSI'] < 30:
        signals['RSI'] = 'Buy'
    elif indicators['RSI'] > 70:
        signals['RSI'] = 'Sell'
    else:
        signals['RSI'] = 'Neutral'
    
    if indicators['MACD'] > 0:
        signals['MACD'] = 'Buy'
    else:
        signals['MACD'] = 'Sell'
    
    if indicators['ADX'] > 25:
        signals['ADX'] = 'Buy'
    else:
        signals['ADX'] = 'Neutral'
    
    if indicators['CCI'] > 100:
        signals['CCI'] = 'Buy'
    elif indicators['CCI'] < -100:
        signals['CCI'] = 'Sell'
    else:
        signals['CCI'] = 'Neutral'
    
    signals['MA'] = 'Buy' if moving_averages['MA5'] > moving_averages['MA10'] else 'Sell'
    
    return signals

signals = generate_signals(indicators, moving_averages, data)

# Log signals
log_file = 'signals_log.csv'
try:
    logs = pd.read_csv(log_file)
except FileNotFoundError:
    logs = pd.DataFrame(columns=['timestamp', 'RSI', 'MACD', 'ADX', 'CCI', 'MA'])

new_log = pd.DataFrame([signals])
logs = pd.concat([logs, new_log], ignore_index=True)
logs.to_csv(log_file, index=False)

# Display the information on Streamlit
st.title('Bitcoin Technical Analysis and Signal Summary')

st.write('### Support Levels:')
st.write(f"{fib_levels[0]:.4f}, {fib_levels[1]:.4f}, {fib_levels[2]:.4f}")

st.write('### Resistance Levels:')
st.write(f"{fib_levels[3]:.4f}, {fib_levels[4]:.4f}, {high:.4f}")

st.write('### Technical Indicators:')
for key, value in indicators.items():
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    st.write(f"{key}: {value:.3f} - {'Buy 🟢' if value > 0 else 'Sell 🔴' if value < 0 else 'Neutral 🟡'}")

st.write('### Moving Averages:')
for key, value in moving_averages.items():
    st.write(f"{key}: {value:.4f} - {'Buy 🟢' if value > data['Close'].iloc[-1] else 'Sell 🔴'}")

st.write('### Summary:')
st.write('Buy 🟢' if 'Buy' in signals.values() else 'Sell 🔴')

st.write('### Signal Entry Rules:')
st.write("Enter the signal during one minute. If the price goes the opposite way, enter from the price rollback or from support/resistance points. Don't forget about risk and money management: do not bet more than 5% of the deposit even with possible overlaps!")

st.write('### Previous Signals:')
st.dataframe(logs)

# Add JavaScript to auto-refresh the Streamlit app every 60 seconds
components.html("""
<script>
setTimeout(function(){
   window.location.reload();
}, 60000);  // Refresh every 60 seconds
</script>
""", height=0)
