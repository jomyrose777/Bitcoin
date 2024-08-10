import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime
import plotly.graph_objects as go
import requests

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    return dt.tz_convert(est) if dt.tzinfo else est.localize(dt)

# Fetch live data from Yahoo Finance
@st.cache
def fetch_data():
    data = yf.download(ticker, period='1d', interval='1m')
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
    else:
        data.index = data.index.tz_convert(est)
    return data

data = fetch_data()

# Calculate technical indicators using the ta library
def calculate_indicators(data):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
    data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['BULLBEAR'] = data['Close'].apply(lambda x: x)  # Replace with actual sentiment if available
    data['UO'] = data['Close'].apply(lambda x: x)  # Replace with actual UO if available
    data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
    data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    data.dropna(inplace=True)
    return data

data = calculate_indicators(data)

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

# Fetch Fear and Greed Index
@st.cache
def fetch_fear_and_greed():
    url = 'https://api.alternative.me/fng/'
    response = requests.get(url)
    data = response.json()
    return data['data'][0]['value']

fear_and_greed = fetch_fear_and_greed()

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
        'WILLIAMSR': data['WILLIAMSR'].iloc[-1]
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
    
    # Fear and Greed Index Signal
    signals['Fear and Greed'] = 'Buy' if int(fear_and_greed) < 50 else 'Sell'
    
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

# Display the information on Streamlit
st.write('### Support Levels:')
st.write(f"{fib_levels[0]:.4f}, {fib_levels[1]:.4f}, {fib_levels[2]:.4f}")

st.write('### Resistance Levels:')
st.write(f"{fib_levels[3]:.4f}, {fib_levels[4]:.4f}, {high:.4f}")

st.write('### Technical Indicators:')
for key, value in indicators.items():
    st.write(f"{key}: {value:.3f} - {'Buy' if value > 0 else 'Sell' if value < 0 else 'Neutral'}")

st.write('### Moving Averages:')
for key, value in moving_averages.items():
    st.write(f"{key}: {value:.4f} - {'Buy' if value > data['Close'].iloc[-1] else 'Sell'}")

st.write('### Fear and Greed Index:')
st.write(f"{fear_and_greed} - {'Buy' if int(fear_and_greed) < 50 else 'Sell'}")

st.write('### Summary:')
st.write('Buy' if 'Buy' in signals.values() else 'Sell')

st.write('### Signal Entry Rules:')
st.write("Enter the signal during one minute. If the price goes the opposite way, enter from the price rollback or from support/resistance points. Don't forget about risk and money management: do not bet more than 5% of the deposit even with possible overlaps!")

st.write('### Previous Signals:')
st.dataframe(logs)

# Example of Perpetual Options Strategy
# Note: This is a simplified version. A more robust strategy should be used based on detailed analysis.
def perpetual_options_strategy(data):
    current_price = data['Close'].iloc[-1]
    if signals['MACD'] == 'Buy' and signals['RSI'] == 'Buy':
        return 'Consider Long Position in Perpetual Options'
    elif signals['MACD'] == 'Sell' and signals['RSI'] == 'Sell':
        return 'Consider Short Position in Perpetual Options'
    else:
        return 'Hold or Wait for Clearer Signals'

options_signal = perpetual_options_strategy(data)
st.write(f'### Perpetual Options Strategy: {options_signal}')

# Calculate accuracy of signals
def calculate_signal_accuracy(logs):
    total_signals = len(logs)
    correct_signals = logs.apply(lambda x: (x['RSI'] == 'Buy' and x['MACD'] == 'Buy') or (x['RSI'] == 'Sell' and x['MACD'] == 'Sell'), axis=1).sum()
    accuracy = (correct_signals / total_signals) * 100 if total_signals > 0 else 0
    return accuracy

accuracy = calculate_signal_accuracy(logs)
st.write(f'### Signal Accuracy: {accuracy:.2f}%')

