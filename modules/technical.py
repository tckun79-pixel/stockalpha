"""
technical.py — 5-year technical analysis: 15+ indicators + Plotly charts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # Trend
    df["SMA20"]  = c.rolling(20).mean()
    df["SMA50"]  = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["EMA12"]  = c.ewm(span=12, adjust=False).mean()
    df["EMA26"]  = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    df["BB_mid"]   = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Stochastic
    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    df["STOCH_K"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

    # Williams %R
    df["WILLR"] = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

    # ATR
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Historical volatility
    df["HV30"] = c.pct_change().rolling(30).std() * np.sqrt(252) * 100

    # Volume indicators
    df["OBV"]      = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["Vol_SMA20"] = v.rolling(20).mean()
    tp = (h + l + c) / 3
    df["VWAP"] = (tp * v).rolling(20).sum() / v.rolling(20).sum()

    return df


def find_support_resistance(df: pd.DataFrame, n: int = 3) -> dict:
    close  = df["Close"].dropna()
    window = 20
    mins, maxs = [], []

    for i in range(window, len(close) - window):
        sl = close.iloc[i - window: i + window + 1]
        if close.iloc[i] == sl.min(): mins.append(close.iloc[i])
        if close.iloc[i] == sl.max(): maxs.append(close.iloc[i])

    def cluster(levels, tol=0.02):
        out = []
        for lv in sorted(levels):
            if not out or abs(lv - out[-1]) / out[-1] > tol:
                out.append(lv)
        return out[:n]

    return {
        "support":    cluster(sorted(mins)),
        "resistance": cluster(sorted(maxs, reverse=True)),
    }


def build_chart(df: pd.DataFrame) -> go.Figure:
    dp = df.last("1Y").copy() if len(df) > 252 else df.copy()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.18, 0.17],
        vertical_spacing=0.02,
        subplot_titles=("Price & Indicators", "Volume", "MACD", "RSI"),
    )

    fig.add_trace(go.Candlestick(
        x=dp.index, open=dp["Open"], high=dp["High"], low=dp["Low"], close=dp["Close"],
        name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    for col, color, w in [("SMA20","#60a5fa",1),("SMA50","#f59e0b",1.5),("SMA200","#a78bfa",2)]:
        if col in dp.columns:
            fig.add_trace(go.Scatter(x=dp.index, y=dp[col], name=col,
                                     line=dict(color=color, width=w)), row=1, col=1)

    if "BB_upper" in dp.columns:
        fig.add_trace(go.Scatter(x=dp.index, y=dp["BB_upper"], name="BB Upper",
                                  line=dict(color="#64748b", dash="dot", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dp.index, y=dp["BB_lower"], name="BB Lower",
                                  line=dict(color="#64748b", dash="dot", width=1),
                                  fill="tonexty", fillcolor="rgba(100,116,139,0.07)"), row=1, col=1)

    colors_v = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(dp["Close"], dp["Open"])]
    fig.add_trace(go.Bar(x=dp.index, y=dp["Volume"], marker_color=colors_v,
                          showlegend=False), row=2, col=1)
    if "Vol_SMA20" in dp.columns:
        fig.add_trace(go.Scatter(x=dp.index, y=dp["Vol_SMA20"],
                                  line=dict(color="#94a3b8", width=1), showlegend=False), row=2, col=1)

    if "MACD" in dp.columns:
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in dp["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=dp.index, y=dp["MACD_hist"], marker_color=hist_colors,
                              showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=dp.index, y=dp["MACD"],
                                  line=dict(color="#60a5fa", width=1.5), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=dp.index, y=dp["MACD_signal"],
                                  line=dict(color="#f59e0b", width=1.5), showlegend=False), row=3, col=1)

    if "RSI" in dp.columns:
        fig.add_trace(go.Scatter(x=dp.index, y=dp["RSI"],
                                  line=dict(color="#00d4aa", width=2), showlegend=False), row=4, col=1)
        fig.add_hline(y=70, line_color="#ef5350", line_dash="dot", line_width=1, row=4, col=1)
        fig.add_hline(y=30, line_color="#26a69a", line_dash="dot", line_width=1, row=4, col=1)

    fig.update_layout(
        height=700,
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#e8eaf0", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#1e2433", row=i, col=1)
        fig.update_yaxes(gridcolor="#1e2433", row=i, col=1)

    return fig


def get_indicator_signals(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    close  = df["Close"].iloc[-1]
    signals = {}

    if "SMA200" in df.columns and not pd.isna(latest["SMA200"]):
        signals["Trend (vs SMA200)"] = "Bullish 📈" if close > latest["SMA200"] else "Bearish 📉"
    if "SMA50" in df.columns and not pd.isna(latest["SMA50"]):
        signals["Trend (vs SMA50)"] = "Above SMA50" if close > latest["SMA50"] else "Below SMA50"

    if "RSI" in df.columns and not pd.isna(latest["RSI"]):
        rsi = round(latest["RSI"], 1)
        signals["RSI (14)"] = (
            f"{rsi} — Overbought 🔴" if rsi > 70 else
            f"{rsi} — Oversold 🟢"  if rsi < 30 else
            f"{rsi} — Neutral ⚪"
        )

    if "MACD" in df.columns and "MACD_signal" in df.columns:
        if not pd.isna(latest["MACD"]) and not pd.isna(latest["MACD_signal"]):
            signals["MACD"] = "Bullish 📈" if latest["MACD"] > latest["MACD_signal"] else "Bearish 📉"

    if all(c in df.columns for c in ["BB_upper","BB_lower"]):
        bb_u, bb_l = latest["BB_upper"], latest["BB_lower"]
        if not pd.isna(bb_u) and (bb_u - bb_l) != 0:
            pct = (close - bb_l) / (bb_u - bb_l) * 100
            signals["Bollinger Position"] = f"{pct:.0f}% (0=Lower, 100=Upper)"

    if "HV30" in df.columns and not pd.isna(latest["HV30"]):
        signals["Hist. Volatility (30d)"] = f"{latest['HV30']:.1f}%"

    return signals


def compute_performance(df: pd.DataFrame) -> dict:
    close = df["Close"].dropna()
    if len(close) < 2:
        return {}
    total_ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
    max_dd    = ((close - close.cummax()) / close.cummax() * 100).min()
    ann_vol   = close.pct_change().std() * np.sqrt(252) * 100
    sign = "+" if total_ret >= 0 else ""
    return {
        "Total Return (5Y)":      f"{sign}{total_ret:.1f}%",
        "Max Drawdown":           f"{max_dd:.1f}%",
        "Annualized Volatility":  f"{ann_vol:.1f}%",
        "Current Price":          f"{close.iloc[-1]:.2f}",
    }
