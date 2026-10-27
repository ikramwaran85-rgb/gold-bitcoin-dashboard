import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Gold & Bitcoin Paper Bot", page_icon="📈", layout="wide")

st.title("☁️ Gold & Bitcoin Cloud Trading Bot")
st.caption("Paper-trading prototype • No real orders are placed • Market data may be delayed")

with st.sidebar:
    st.header("Strategy settings")
    period = st.selectbox("Historical data", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Candle interval", ["1d", "1h"], index=0)
    fast = st.slider("Fast moving average", 3, 30, 10)
    slow = st.slider("Slow moving average", 10, 100, 30)
    starting_balance = st.number_input("Paper balance (USD)", min_value=100.0, max_value=1_000_000.0, value=10_000.0, step=500.0)
    risk_pct = st.slider("Simulated position size (% of balance)", 1, 25, 5)
    st.info("This prototype uses moving-average signals and a simplified backtest. It does not include fees, slippage, leverage, or broker-specific execution.")

if fast >= slow:
    st.warning("For this strategy, the fast moving average should be lower than the slow moving average.")

@st.cache_data(ttl=60)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    if "Close" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["Close"]).copy()
    return df

def analyze(df, fast, slow, starting_balance, risk_pct):
    out = df.copy()
    out["Fast MA"] = out["Close"].rolling(fast).mean()
    out["Slow MA"] = out["Close"].rolling(slow).mean()
    out["Signal"] = np.where(out["Fast MA"] > out["Slow MA"], 1, 0)
    # Apply yesterday's signal to today's return to avoid using future information.
    out["Market Return"] = out["Close"].pct_change().fillna(0)
    out["Strategy Return"] = out["Signal"].shift(1).fillna(0) * out["Market Return"] * (risk_pct / 100)
    out["Equity"] = starting_balance * (1 + out["Strategy Return"]).cumprod()
    out["Buy"] = (out["Signal"].diff() == 1)
    out["Sell"] = (out["Signal"].diff() == -1)
    return out

assets = [
    ("Bitcoin (BTC/USD)", "BTC-USD"),
    ("Gold futures (GC=F)", "GC=F"),
]
cols = st.columns(2)

for col, (label, ticker) in zip(cols, assets):
    with col:
        st.subheader(label)
        raw = load_data(ticker, period, interval)
        if raw.empty:
            st.error(f"Could not load data for {ticker}. Try another interval or historical period.")
            continue
        result = analyze(raw, fast, slow, starting_balance, risk_pct)
        last = result.iloc[-1]
        previous = result.iloc[-2] if len(result) > 1 else last
        price = float(last["Close"])
        change = ((price / float(previous["Close"])) - 1) * 100 if float(previous["Close"]) else 0
        current_signal = "BUY / HOLD (paper)" if float(last["Signal"]) == 1 else "CASH / WAIT"
        m1, m2 = st.columns(2)
        m1.metric("Latest close", f"${price:,.2f}", f"{change:+.2f}%")
        m2.metric("Strategy state", current_signal)
        st.line_chart(result[["Close", "Fast MA", "Slow MA"]].dropna(), height=260)
        st.caption("Price and moving averages. A BUY state means the fast average is above the slow average; it is not a guarantee of future performance.")
        st.markdown("**Paper strategy summary**")
        total_return = (float(result["Equity"].iloc[-1]) / starting_balance - 1) * 100
        peak = result["Equity"].cummax()
        drawdown = ((result["Equity"] / peak) - 1).min() * 100
        a, b, c = st.columns(3)
        a.metric("Simulated return", f"{total_return:+.2f}%")
        b.metric("Max drawdown", f"{drawdown:.2f}%")
        c.metric("Signal changes", int((result["Signal"].diff().abs() == 1).sum()))
        events = result.loc[result["Buy"] | result["Sell"], ["Close", "Fast MA", "Slow MA", "Buy", "Sell"]].tail(8).copy()
        if not events.empty:
            events["Action"] = np.where(events["Buy"], "BUY (paper)", "SELL (paper)")
            events = events[["Action", "Close", "Fast MA", "Slow MA"]]
            st.dataframe(events.sort_index(ascending=False), use_container_width=True)
        else:
            st.write("No moving-average crossovers in the selected period.")

st.divider()
st.subheader("How to interpret this prototype")
st.markdown("""
- **BUY / HOLD (paper):** The fast moving average is above the slow moving average.
- **CASH / WAIT:** The fast moving average is at or below the slow moving average.
- Gold uses the `GC=F` gold-futures ticker as a market-data proxy; it is not a broker's exact XAU/USD spot quote.
- The displayed backtest is simplified and educational. It excludes transaction costs, taxes, slippage, financing, and execution constraints.
- **No real trading is connected.** Never put exchange passwords or API secrets into this app. Live execution should only be added after careful testing with a compatible broker, secure secret storage, and explicit risk limits.
""")
st.caption("Data provided by Yahoo Finance through yfinance. Availability, delay, and terms can vary.")
