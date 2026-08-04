import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="SMA Trading Signals", layout="wide")
st.title("SMA Crossover Trading Signals")

# Sidebar inputs
with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL").upper()
    start_date = st.date_input("Start date", datetime.today() - timedelta(days=365))
    end_date = st.date_input("End date", datetime.today())
    sma_fast = st.number_input("Fast SMA", min_value=1, max_value=200, value=20)
    sma_slow = st.number_input("Slow SMA", min_value=1, max_value=200, value=50)

if not ticker:
    st.warning("Please enter a ticker.")
    st.stop()

# Download data
data = yf.download(ticker, start=str(start_date), end=str(end_date), auto_adjust=True, progress=False)

if data.empty:
    st.error("No data returned. Check the ticker or date range.")
    st.stop()

# Ensure columns are clean
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if col[1] == '' or col[1] == ticker else col[0] for col in data.columns]
data = data.reset_index()

data = data.dropna().reset_index(drop=True)

# Compute SMAs
data["SMA_Fast"] = data["Close"].rolling(window=sma_fast).mean()
data["SMA_Slow"] = data["Close"].rolling(window=sma_slow).mean()

# Signals: 1 = buy, -1 = sell, 0 = hold
data["Signal"] = 0
data.loc[data["SMA_Fast"] > data["SMA_Slow"], "Signal"] = 1
data.loc[data["SMA_Fast"] < data["SMA_Slow"], "Signal"] = -1

# Trade events on crossover
data["Trade"] = 0
fast = data["SMA_Fast"]
slow = data["SMA_Slow"]
prev_fast = fast.shift(1)
prev_slow = slow.shift(1)

buy_cond = (prev_fast <= prev_slow) & (fast > slow)
sell_cond = (prev_fast >= prev_slow) & (fast < slow)

data.loc[buy_cond, "Trade"] = 1
data.loc[sell_cond, "Trade"] = -1

latest_idx = data.index[-1]
latest_signal = data.loc[latest_idx, "Trade"]
signal_label = "BUY" if latest_signal == 1 else ("SELL" if latest_signal == -1 else "HOLD")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker", ticker)
c2.metric("Price", f"{data.loc[latest_idx, 'Close']:.2f}")
c3.metric(f"SMA {sma_fast}", f"{data.loc[latest_idx, 'SMA_Fast']:.2f}")
c4.metric(f"SMA {sma_slow}", f"{data.loc[latest_idx, 'SMA_Slow']:.2f}")
st.info(f"Latest signal: **{signal_label}**")

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(color="gray")))
fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA_Fast"], name=f"SMA {sma_fast}", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA_Slow"], name=f"SMA {sma_slow}", line=dict(color="blue")))

buys = data[data["Trade"] == 1]
sells = data[data["Trade"] == -1]

fig.add_trace(go.Scatter(
    x=buys["Date"], y=buys["Close"],
    mode="markers", name="Buy",
    marker=dict(color="green", symbol="triangle-up", size=10)
))
fig.add_trace(go.Scatter(
    x=sells["Date"], y=sells["Close"],
    mode="markers", name="Sell",
    marker=dict(color="red", symbol="triangle-down", size=10)
))

fig.update_layout(
    title=f"{ticker} Price with SMA Crossovers",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# Show recent signals
st.subheader("Recent Signals")
recent = data[data["Trade"] != 0][["Date", "Close", "SMA_Fast", "SMA_Slow", "Trade"]].tail(20).copy()
recent["Action"] = recent["Trade"].map({1: "Buy", -1: "Sell"})
st.dataframe(recent[["Date", "Close", "SMA_Fast", "SMA_Slow", "Action"]], use_container_width=True)
