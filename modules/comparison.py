"""
comparison.py — Multi-ticker comparison engine for StockAlpha.
Compares 2–5 tickers across price performance, fundamentals, and technicals.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from modules.data_fetcher import fetch_all, get_current_price, get_currency_symbol


def fetch_comparison_data(tickers: list) -> dict:
    """Fetch data for all tickers. Returns dict keyed by ticker."""
    results = {}
    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue
        data = fetch_all(t)
        if data["valid"]:
            results[t] = data
        else:
            results[t] = {"valid": False, "error": data.get("error", "Unknown error"), "ticker": t}
    return results


def build_normalized_price_chart(datasets: dict, period: str = "1Y") -> go.Figure:
    """Normalized price performance chart (base=100) for all tickers."""
    period_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}
    days = period_map.get(period, 252)

    COLORS = ["#00d4aa", "#60a5fa", "#f59e0b", "#a78bfa", "#f87171"]
    fig = go.Figure()

    for i, (ticker, data) in enumerate(datasets.items()):
        if not data.get("valid"):
            continue
        hist = data["price_history"]["Close"].dropna()
        hist = hist.tail(days) if len(hist) > days else hist
        if hist.empty:
            continue
        normalized = (hist / hist.iloc[0]) * 100
        color = COLORS[i % len(COLORS)]
        info = data.get("info", {})
        name = info.get("shortName") or ticker

        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            name=f"{ticker} — {name}",
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{ticker}</b><br>Date: %{{x|%Y-%m-%d}}<br>Return: %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_hline(y=100, line_color="#2a2d3a", line_dash="dot", line_width=1)
    fig.update_layout(
        title=f"Normalized Price Performance — {period} (Base=100)",
        height=420,
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#e8eaf0", size=11),
        xaxis=dict(gridcolor="#1e2433"),
        yaxis=dict(gridcolor="#1e2433", ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    return fig


def build_volume_chart(datasets: dict, period: str = "1Y") -> go.Figure:
    """Normalized average volume comparison bar chart."""
    period_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}
    days = period_map.get(period, 252)
    COLORS = ["#00d4aa", "#60a5fa", "#f59e0b", "#a78bfa", "#f87171"]

    tickers, avg_vols = [], []
    for ticker, data in datasets.items():
        if not data.get("valid"):
            continue
        vol = data["price_history"]["Volume"].dropna().tail(days)
        tickers.append(ticker)
        avg_vols.append(vol.mean())

    fig = go.Figure(go.Bar(
        x=tickers, y=avg_vols,
        marker_color=COLORS[:len(tickers)],
        text=[f"{v/1e6:.1f}M" if v >= 1e6 else f"{v:,.0f}" for v in avg_vols],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Avg Daily Volume ({period})",
        height=300,
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#e8eaf0", size=11),
        xaxis=dict(gridcolor="#1e2433"),
        yaxis=dict(gridcolor="#1e2433"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def build_rsi_comparison_chart(datasets: dict, period: str = "1Y") -> go.Figure:
    """RSI comparison across tickers."""
    period_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}
    days = period_map.get(period, 252)
    COLORS = ["#00d4aa", "#60a5fa", "#f59e0b", "#a78bfa", "#f87171"]

    fig = go.Figure()
    for i, (ticker, data) in enumerate(datasets.items()):
        if not data.get("valid"):
            continue
        close = data["price_history"]["Close"].dropna().tail(days + 20)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        rsi = rsi.tail(days)
        fig.add_trace(go.Scatter(
            x=rsi.index, y=rsi.values,
            name=ticker, line=dict(color=COLORS[i % len(COLORS)], width=1.5),
        ))

    fig.add_hline(y=70, line_color="#ef5350", line_dash="dot", line_width=1,
                  annotation_text="Overbought 70")
    fig.add_hline(y=30, line_color="#26a69a", line_dash="dot", line_width=1,
                  annotation_text="Oversold 30")
    fig.update_layout(
        title="RSI (14) Comparison",
        height=280,
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#e8eaf0", size=11),
        xaxis=dict(gridcolor="#1e2433"),
        yaxis=dict(gridcolor="#1e2433", range=[0, 100]),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    return fig


def build_drawdown_chart(datasets: dict, period: str = "1Y") -> go.Figure:
    """Drawdown comparison chart."""
    period_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}
    days = period_map.get(period, 252)
    COLORS = ["#00d4aa", "#60a5fa", "#f59e0b", "#a78bfa", "#f87171"]

    fig = go.Figure()
    for i, (ticker, data) in enumerate(datasets.items()):
        if not data.get("valid"):
            continue
        close = data["price_history"]["Close"].dropna().tail(days)
        rolling_max = close.cummax()
        drawdown = (close - rolling_max) / rolling_max * 100
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            name=ticker, line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            fill="tozeroy", fillcolor=f"rgba({','.join(str(int(COLORS[i%len(COLORS)].lstrip('#')[j:j+2],16)) for j in (0,2,4))},0.05)",
        ))

    fig.update_layout(
        title="Drawdown Comparison (%)",
        height=280,
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#e8eaf0", size=11),
        xaxis=dict(gridcolor="#1e2433"),
        yaxis=dict(gridcolor="#1e2433", ticksuffix="%"),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    return fig


def build_fundamental_comparison(datasets: dict) -> dict:
    """Extract and compare key fundamental metrics across tickers."""
    METRICS = [
        ("Market Cap",         "marketCap",                     True,  False),
        ("P/E (Trailing)",     "trailingPE",                    False, False),
        ("P/E (Forward)",      "forwardPE",                     False, False),
        ("P/S Ratio",          "priceToSalesTrailing12Months",  False, False),
        ("P/B Ratio",          "priceToBook",                   False, False),
        ("EPS (Trailing)",     "trailingEps",                   False, False),
        ("Revenue Growth",     "revenueGrowth",                 False, True),
        ("Earnings Growth",    "earningsGrowth",                False, True),
        ("Profit Margin",      "profitMargins",                 False, True),
        ("ROE",                "returnOnEquity",                False, True),
        ("ROA",                "returnOnAssets",                False, True),
        ("Debt/Equity",        "debtToEquity",                  False, False),
        ("Current Ratio",      "currentRatio",                  False, False),
        ("Dividend Yield",     "dividendYield",                 False, True),
        ("52W High",           "fiftyTwoWeekHigh",              False, False),
        ("52W Low",            "fiftyTwoWeekLow",               False, False),
        ("Beta",               "beta",                          False, False),
    ]

    rows = []
    for label, key, is_mcap, is_pct in METRICS:
        row = {"Metric": label}
        for ticker, data in datasets.items():
            if not data.get("valid"):
                row[ticker] = "—"
                continue
            val = data["info"].get(key)
            if val is None:
                row[ticker] = "—"
            elif is_mcap:
                row[ticker] = f"${val/1e12:.2f}T" if val >= 1e12 else f"${val/1e9:.2f}B" if val >= 1e9 else f"${val/1e6:.0f}M"
            elif is_pct:
                row[ticker] = f"{val*100:.1f}%" if val else "—"
            else:
                row[ticker] = f"{val:.2f}" if isinstance(val, float) else str(val)
        rows.append(row)

    return rows


def build_performance_comparison(datasets: dict, period: str = "1Y") -> list:
    """Return/volatility/sharpe comparison table."""
    period_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}
    days = period_map.get(period, 252)

    rows = []
    for ticker, data in datasets.items():
        if not data.get("valid"):
            rows.append({"Ticker": ticker, "Return": "Error", "Ann. Vol": "—",
                         "Max DD": "—", "Sharpe": "—", "Beta": "—"})
            continue

        close = data["price_history"]["Close"].dropna().tail(days)
        info  = data.get("info", {})
        if len(close) < 5:
            rows.append({"Ticker": ticker, "Return": "No data", "Ann. Vol": "—",
                         "Max DD": "—", "Sharpe": "—", "Beta": "—"})
            continue

        ret     = (close.iloc[-1] / close.iloc[0] - 1) * 100
        daily_r = close.pct_change().dropna()
        ann_vol = daily_r.std() * np.sqrt(252) * 100
        max_dd  = ((close - close.cummax()) / close.cummax() * 100).min()
        sharpe  = (daily_r.mean() / daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0
        beta    = info.get("beta")
        name    = info.get("shortName") or ticker

        rows.append({
            "Ticker":   ticker,
            "Name":     name,
            "Return":   f"{ret:+.1f}%",
            "Ann. Vol": f"{ann_vol:.1f}%",
            "Max DD":   f"{max_dd:.1f}%",
            "Sharpe":   f"{sharpe:.2f}",
            "Beta":     f"{beta:.2f}" if beta else "—",
            "_ret":     ret,
            "_sharpe":  sharpe,
        })

    return rows


def build_radar_chart(datasets: dict) -> go.Figure:
    """Radar chart comparing normalized scores across 5 dimensions."""
    COLORS = ["#00d4aa", "#60a5fa", "#f59e0b", "#a78bfa", "#f87171"]
    categories = ["Valuation\n(Lower=Better)", "Profitability", "Growth",
                  "Momentum", "Financial Health"]

    def norm_score(val, low_better=False, vmin=0, vmax=100):
        if val is None: return 50
        val = max(vmin, min(vmax, val))
        score = (val - vmin) / (vmax - vmin) * 100
        return 100 - score if low_better else score

    fig = go.Figure()
    for i, (ticker, data) in enumerate(datasets.items()):
        if not data.get("valid"): continue
        info  = data["info"]
        close = data["price_history"]["Close"].dropna()

        # Valuation: forward P/E (lower = better), normalize 0-50
        fpe   = info.get("forwardPE") or 25
        val_s = norm_score(fpe, low_better=True, vmin=5, vmax=50)

        # Profitability: profit margin %
        pm    = (info.get("profitMargins") or 0) * 100
        prof_s = norm_score(pm, vmin=-20, vmax=40)

        # Growth: revenue growth %
        rg    = (info.get("revenueGrowth") or 0) * 100
        grow_s = norm_score(rg, vmin=-20, vmax=50)

        # Momentum: % above 200-day SMA
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        mom_s  = norm_score((close.iloc[-1] / sma200 - 1) * 100, vmin=-30, vmax=30)

        # Financial health: current ratio
        cr     = info.get("currentRatio") or 1
        health_s = norm_score(cr, vmin=0.5, vmax=3)

        scores = [val_s, prof_s, grow_s, mom_s, health_s]
        scores_closed = scores + [scores[0]]
        cats_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=scores_closed, theta=cats_closed,
            fill="toself",
            fillcolor=f"rgba({','.join(str(int(COLORS[i%len(COLORS)].lstrip('#')[j:j+2],16)) for j in (0,2,4))},0.12)",
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            name=ticker,
        ))

    fig.update_layout(
        title="Multi-Dimensional Score Radar",
        polar=dict(
            bgcolor="#1a1d27",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2a2d3a",
                            tickcolor="#8b8fa8", tickfont=dict(size=9, color="#8b8fa8")),
            angularaxis=dict(gridcolor="#2a2d3a", tickcolor="#8b8fa8",
                             tickfont=dict(size=10, color="#e8eaf0")),
        ),
        paper_bgcolor="#0f1117",
        font=dict(color="#e8eaf0", size=11),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=50, b=60),
        height=420,
    )
    return fig


def build_correlation_matrix(datasets: dict, period: str = "1Y") -> go.Figure:
    """Correlation heatmap of returns."""
    period_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}
    days = period_map.get(period, 252)

    returns_df = pd.DataFrame()
    for ticker, data in datasets.items():
        if not data.get("valid"): continue
        close = data["price_history"]["Close"].dropna().tail(days)
        returns_df[ticker] = close.pct_change()

    if returns_df.shape[1] < 2:
        return None

    corr = returns_df.corr()
    tickers = list(corr.columns)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=tickers, y=tickers,
        colorscale=[[0, "#ef5350"], [0.5, "#1a1d27"], [1, "#26a69a"]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=12, color="#e8eaf0"),
        hoverongaps=False,
    ))
    fig.update_layout(
        title=f"Return Correlation Matrix ({period})",
        height=300 + len(tickers) * 30,
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#e8eaf0", size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(side="bottom"),
    )
    return fig
