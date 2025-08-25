import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
import time

# --- Streamlit Page Config ---
st.set_page_config(page_title="BTC Price & Volume Analysis", layout="wide")

# --- App Title ---
st.title("📊 BTC Price & Volume Analysis with RSI(9), Volume Heatmap & Divergence Detection")

# --- Parameters ---
symbol = "BTCUSDT"
interval = st.selectbox("Select Interval", ["1m", "5m", "15m", "1h", "4h", "1d"], index=1)
limit = 300  # number of candles

# --- Fetch Data with Fallback ---
@st.cache_data(ttl=60)
def get_crypto_data(symbol="BTCUSDT", interval="1m", limit=300):
    # --- Try Binance API ---
    binance_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(binance_url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data:
                df = pd.DataFrame(data, columns=[
                    "time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
                return df
    except Exception as e:
        st.warning(f"Binance fetch failed: {e}")

    # --- Fallback: CoinGecko OHLC ---
    st.info("⚠️ Using CoinGecko fallback (Binance data unavailable).")
    cg_url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1"
    try:
        r = requests.get(cg_url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data:
                df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df["volume"] = np.nan  # CoinGecko doesn’t provide volume
                return df
    except Exception as e:
        st.error(f"CoinGecko fetch failed: {e}")

    return pd.DataFrame()  # Empty fallback

# --- RSI Calculation ---
def calculate_rsi(series, period=9):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Divergence Detection ---
def detect_divergence(df, rsi_col="RSI9"):
    divergences = []
    for i in range(2, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-2] and df[rsi_col].iloc[i] < df[rsi_col].iloc[i-2]:
            divergences.append("Bearish")
        elif df["close"].iloc[i] < df["close"].iloc[i-2] and df[rsi_col].iloc[i] > df[rsi_col].iloc[i-2]:
            divergences.append("Bullish")
        else:
            divergences.append(None)
    return [None, None] + divergences

# --- Get Data ---
df = get_crypto_data(symbol, interval, limit)

if df.empty:
    st.error("❌ No data available from Binance or CoinGecko.")
    st.stop()

# --- Add RSI & Divergence ---
df["RSI9"] = calculate_rsi(df["close"], 9)
df["Divergence"] = detect_divergence(df)

# --- Convert UTC hour to IST hour ---
def utc_to_ist_hour(utc_hour):
    ist_hour = (utc_hour + 5 + (30/60)) % 24
    return ist_hour

df["Hour"] = df["time"].dt.hour
df["Hour_IST"] = df["Hour"].apply(utc_to_ist_hour)

# --- Hourly Average Volume in IST ---
avg_volume_by_hour = (
    df.groupby("Hour_IST")["volume"]
    .mean()
    .reset_index()
    .sort_values("volume", ascending=False)
)

# --- Price Chart with Divergence + Peak Hour Shading ---
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
))

# Volume bars (only if volume available)
if not df["volume"].isna().all():
    fig.add_trace(go.Bar(
        x=df["time"], y=df["volume"], name="Volume", yaxis="y2", opacity=0.3
    ))

# Highlight divergences
bearish_points = df[df["Divergence"] == "Bearish"]
bullish_points = df[df["Divergence"] == "Bullish"]

fig.add_trace(go.Scatter(
    x=bearish_points["time"], y=bearish_points["close"], mode="markers",
    name="Bearish Div", marker=dict(color="red", size=8, symbol="triangle-down")
))
fig.add_trace(go.Scatter(
    x=bullish_points["time"], y=bullish_points["close"], mode="markers",
    name="Bullish Div", marker=dict(color="green", size=8, symbol="triangle-up")
))

# Shade Top 3 Peak Hours in IST
if not avg_volume_by_hour.empty:
    top_hours_ist = avg_volume_by_hour.head(3)["Hour_IST"].astype(int).tolist()
    for hour in top_hours_ist:
        mask = df["Hour_IST"].astype(int) == hour
        if mask.any():
            x0 = df.loc[mask, "time"].min()
            x1 = df.loc[mask, "time"].max()
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor="rgba(255, 215, 0, 0.2)",
                layer="below",
                line_width=0,
                annotation_text=f"Peak {hour:02d}:00 IST",
                annotation_position="top left"
            )

fig.update_layout(
    xaxis_rangeslider_visible=False,
    yaxis=dict(title="Price"),
    yaxis2=dict(title="Volume", overlaying="y", side="right"),
    height=600,
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# --- RSI Chart ---
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["RSI9"], mode="lines", name="RSI(9)", line=dict(color="orange")))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
fig_rsi.update_layout(title="RSI(9)", height=300, template="plotly_dark")
st.plotly_chart(fig_rsi, use_container_width=True)

# --- Volume Heatmap (IST) ---
if not df["volume"].isna().all():
    heatmap = px.density_heatmap(
        df, x="Hour_IST", y=df["time"].dt.date, z="volume",
        color_continuous_scale="Viridis", title="Hourly Volume Heatmap (IST)"
    )
    st.plotly_chart(heatmap, use_container_width=True)

# --- Average Volume by Hour (IST) ---
if not avg_volume_by_hour.empty and not df["volume"].isna().all():
    bar_chart = px.bar(
        avg_volume_by_hour,
        x="Hour_IST",
        y="volume",
        title="Average BTC Volume by Hour (IST)",
        labels={"volume": "Average Volume", "Hour_IST": "Hour of Day (IST)"},
        color="volume",
        color_continuous_scale="Viridis"
    )
    bar_chart.update_xaxes(
        tickmode="array",
        tickvals=list(range(0, 24)),
        ticktext=[f"{int(h):02d}:00" for h in range(0, 24)]
    )
    st.plotly_chart(bar_chart, use_container_width=True)

    # Show top 3 hours
    st.subheader("🔥 Top Trading Hours (IST)")
    for _, row in avg_volume_by_hour.head(3).iterrows():
        hour_label = f"{int(row['Hour_IST']):02d}:00"
        st.write(f"{hour_label} → {row['volume']:.2f} average volume")

# --- Latest Price & RSI ---
latest_price = df["close"].iloc[-1]
latest_rsi = df["RSI9"].iloc[-1]
col1, col2 = st.columns(2)
col1.metric("Latest BTC Price", f"${latest_price:,.2f}")
col2.metric("Latest RSI(9)", f"{latest_rsi:.2f}")

# --- Auto-refresh ---
refresh_sec = st.slider("Auto-refresh every (seconds)", 5, 60, 15)
time.sleep(refresh_sec)
st.rerun()
