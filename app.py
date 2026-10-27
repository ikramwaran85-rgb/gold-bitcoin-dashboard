import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Gold & Bitcoin Pro Dashboard", page_icon="📈", layout="wide")

st.title("📈 Gold & Bitcoin Trading Dashboard")
st.caption("Technical-analysis and paper-trading prototype • No real orders are placed • Market data may be delayed")

ASSETS = {
    "Bitcoin (BTC/USD)": "BTC-USD",
    "Gold futures (GC=F)": "GC=F",
}

with st.sidebar:
    st.header("Chart settings")
    selected_assets = st.multiselect(
        "Markets to display",
        list(ASSETS.keys()),
        default=list(ASSETS.keys()),
    )
    period = st.selectbox("Historical data", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Candlestick interval", ["1d", "1h", "30m"], index=0)
    st.divider()
    st.header("Indicator settings")
    ema_fast = st.slider("Fast EMA", 5, 50, 10)
    ema_slow = st.slider("Slow EMA", 10, 200, 30)
    rsi_period = st.slider("RSI period", 5, 30, 14)
    rsi_low = st.slider("RSI lower level", 10, 40, 30)
    rsi_high = st.slider("RSI upper level", 60, 90, 70)
    bb_period = st.slider("Bollinger Bands period", 10, 50, 20)
    bb_std = st.slider("Bollinger Bands width (std dev)", 1.0, 3.0, 2.0, 0.25)
    st.divider()
    st.header("Paper-trading settings")
    starting_balance = st.number_input("Paper balance (USD)", min_value=100.0, max_value=1_000_000.0, value=10_000.0, step=500.0)
    position_pct = st.slider("Position size (% of balance)", 1, 25, 5)
    st.info("Signals are educational examples, not guarantees. This prototype does not connect to a broker or exchange.")

@st.cache_data(ttl=300)
def download_data(ticker, selected_period, selected_interval):
    data = yf.download(
        ticker,
        period=selected_period,
        interval=selected_interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.loc[:, ~data.columns.duplicated()].copy()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in needed):
        return pd.DataFrame()
    data = data[needed].dropna(subset=["Open", "High", "Low", "Close"])
    return data

def add_indicators(data):
    d = data.copy()
    d["EMA Fast"] = d["Close"].ewm(span=ema_fast, adjust=False).mean()
    d["EMA Slow"] = d["Close"].ewm(span=ema_slow, adjust=False).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    d["MACD"] = d["Close"].ewm(span=12, adjust=False).mean() - d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD Histogram"] = d["MACD"] - d["MACD Signal"]

    d["BB Middle"] = d["Close"].rolling(bb_period).mean()
    rolling_std = d["Close"].rolling(bb_period).std()
    d["BB Upper"] = d["BB Middle"] + bb_std * rolling_std
    d["BB Lower"] = d["BB Middle"] - bb_std * rolling_std

    # Example rule: EMA crossover plus MACD confirmation; RSI filter avoids some extremes.
    bullish = (d["EMA Fast"] > d["EMA Slow"]) & (d["MACD"] > d["MACD Signal"]) & (d["RSI"] < rsi_high)
    bearish = (d["EMA Fast"] < d["EMA Slow"]) & (d["MACD"] < d["MACD Signal"]) & (d["RSI"] > rsi_low)
    d["Signal"] = np.select([bullish, bearish], ["BUY (paper)", "SELL (paper)"], default="WAIT")
    return d

def make_chart(data, title):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=("Candlesticks, EMA & Bollinger Bands", "RSI", "MACD"),
    )
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
        name="Candles"
    ), row=1, col=1)
    for col, color in [("EMA Fast", "#1f77b4"), ("EMA Slow", "#ff7f0e")]:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col, line=dict(width=1.5, color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["BB Upper"], name="BB Upper", line=dict(width=1, dash="dot"), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["BB Middle"], name="BB Middle", line=dict(width=1, dash="dash"), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["BB Lower"], name="BB Lower", line=dict(width=1, dash="dot"), opacity=0.7, fill="tonexty"), row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="#9467bd", width=1.5)), row=2, col=1)
    fig.add_hline(y=rsi_high, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=rsi_low, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="#1f77b4", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MACD Signal"], name="MACD Signal", line=dict(color="#ff7f0e", width=1.5)), row=3, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data["MACD Histogram"], name="MACD Histogram", opacity=0.5), row=3, col=1)
    fig.update_layout(title=title, height=850, xaxis_rangeslider_visible=False, template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig

if not selected_assets:
    st.warning("Choose at least one market in the sidebar.")
    st.stop()

tabs = st.tabs(selected_assets)
for tab, asset_name in zip(tabs, selected_assets):
    with tab:
        ticker = ASSETS[asset_name]
        raw = download_data(ticker, period, interval)
        if raw.empty or len(raw) < max(ema_slow, bb_period, rsi_period) + 5:
            st.error(f"Could not load enough market data for {asset_name}. Try a different historical period or interval.")
            continue

        data = add_indicators(raw)
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        price_change = (latest["Close"] / previous["Close"] - 1) * 100 if previous["Close"] else 0
        state = latest["Signal"]
        a, b, c, d = st.columns(4)
        a.metric("Latest close", f"${latest['Close']:,.2f}", f"{price_change:+.2f}%")
        b.metric("Paper signal", state)
        c.metric("RSI", f"{latest['RSI']:.1f}" if pd.notna(latest["RSI"]) else "N/A")
        d.metric("MACD", f"{latest['MACD']:.3f}" if pd.notna(latest["MACD"]) else "N/A")

        st.plotly_chart(make_chart(data, asset_name), use_container_width=True)

        # Transparent, simplified historical paper simulation. Position changes on the next bar.
        valid = data.dropna(subset=["EMA Fast", "EMA Slow", "MACD", "MACD Signal", "RSI"]).copy()
        valid["Long"] = (valid["EMA Fast"] > valid["EMA Slow"]) & (valid["MACD"] > valid["MACD Signal"]) & (valid["RSI"] < rsi_high)
        valid["Position"] = valid["Long"].shift(1).fillna(False).astype(int)
        valid["Market Return"] = valid["Close"].pct_change().fillna(0)
        valid["Strategy Return"] = valid["Position"] * valid["Market Return"]
        valid["Equity"] = starting_balance * (1 + valid["Strategy Return"]).cumprod()
        peak = valid["Equity"].cummax()
        drawdown = (valid["Equity"] / peak - 1) * 100
        total_return = (valid["Equity"].iloc[-1] / starting_balance - 1) * 100 if len(valid) else 0
        max_drawdown = drawdown.min() if len(valid) else 0
        buys = int((valid["Long"] & ~valid["Long"].shift(1, fill_value=False)).sum()) if len(valid) else 0

        st.subheader("Paper-strategy summary")
        x, y, z = st.columns(3)
        x.metric("Simulated return", f"{total_return:+.2f}%")
        y.metric("Max drawdown", f"{max_drawdown:.2f}%")
        z.metric("Long-entry signals", str(buys))
        st.caption(f"Illustrative backtest only. Position size setting ({position_pct}%) is displayed for planning but this simple simulation assumes full exposure when long. It excludes fees, spread, slippage, funding, taxes, and real execution.")
        with st.expander("Latest indicator values"):
            cols = ["Close", "EMA Fast", "EMA Slow", "RSI", "MACD", "MACD Signal", "BB Upper", "BB Middle", "BB Lower", "Signal"]
            st.dataframe(data[cols].tail(10).sort_index(ascending=False), use_container_width=True)
        st.download_button(
            f"Download {asset_name} indicator data (CSV)",
            data.to_csv().encode("utf-8"),
            file_name=f"{ticker.replace('=', '_')}_indicators.csv",
            mime="text/csv",
        )

st.divider()
st.caption("Educational software only; not financial advice. This app does not place real trades. Historical results and indicators do not predict future returns.")
