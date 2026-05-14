"""
wheel_scanner.py — Multi-ticker Wheel Strategy Scanner
CSP (Cash-Secured Put) and CC (Covered Call) opportunity screening.
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ── HELPERS ─────────────────────────────────────────────────────────────────────

def _fmt(val, decimals=2, suffix=''):
    """Format a numeric value, returning '—' for None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '—'
    return f'{val:.{decimals}f}{suffix}'


def _nearest_strike(target, strikes):
    """Return the strike closest to target."""
    return min(strikes, key=lambda s: abs(s - target))


def _hv30(ticker_obj):
    """Annualized 30-day historical volatility from log returns."""
    hist = ticker_obj.history(period="2mo")
    if hist.empty or len(hist) < 10:
        return None
    closes = hist["Close"]
    log_ret = np.log(closes / closes.shift(1)).dropna()
    hv = log_ret.tail(30).std() * math.sqrt(252)
    return hv * 100  # return as percentage


def _days_to_expiry(exp_date_str):
    """Calculate calendar days between today and expiry."""
    exp = datetime.strptime(exp_date_str, "%Y-%m-%d")
    return (exp - datetime.now()).days


def _get_options_chain(ticker_obj, min_dte=7):
    """Fetch nearest options expiry >= min_dte days away. Returns (expiry_str, calls_df, puts_df) or None."""
    try:
        exps = ticker_obj.options
    except Exception:
        return None
    if not exps:
        return None

    target_date = datetime.now() + timedelta(days=min_dte)
    # Find the first expiry >= target_date
    candidates = [e for e in exps if datetime.strptime(e, "%Y-%m-%d") >= target_date]
    if not candidates:
        return None
    expiry = candidates[0]
    try:
        chain = ticker_obj.option_chain(expiry)
    except Exception:
        return None
    return expiry, chain.calls, chain.puts


def analyze_ticker(ticker_str):
    """
    Fetch and analyze one ticker.
    Returns a dict with keys: ticker, price, expiry, dte, csp, cc, hv30, error.
    """
    out = {"ticker": ticker_str, "error": None, "csp": None, "cc": None}

    try:
        tk = yf.Ticker(ticker_str)
        info = tk.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            hist = tk.history(period="5d")
            if hist.empty:
                return {**out, "error": "No price data"}
            price = float(hist["Close"].iloc[-1])
        out["price"] = price

        # Options chain
        result = _get_options_chain(tk, min_dte=7)
        if result is None:
            return {**out, "error": "No options data"}
        expiry, calls_df, puts_df = result
        out["expiry"] = expiry
        out["dte"] = _days_to_expiry(expiry)

        # CSP analysis — put strike closest to 90% of price
        target_csp = round(price * 0.90, 2)
        available_puts = puts_df[puts_df["strike"] <= price * 0.95]
        if available_puts.empty:
            available_puts = puts_df
        csp_strike = _nearest_strike(target_csp, available_puts["strike"].values)
        csp_row = puts_df[puts_df["strike"] == csp_strike].iloc[0]
        bid = csp_row.get("bid", 0) or 0
        iv = (csp_row.get("impliedVolatility", 0) or 0) * 100
        delta = csp_row.get("delta", None)
        premium_yield = (bid / price) * 100 if price > 0 else 0.0
        annualized = premium_yield * 365 / out["dte"] if out["dte"] > 0 else 0.0
        out["csp"] = {
            "strike": float(csp_strike),
            "bid": float(bid),
            "premium_yield": round(premium_yield, 2),
            "annualized": round(annualized, 2),
            "iv": round(iv, 2),
            "delta": round(float(delta), 2) if delta is not None else None,
        }

        # CC analysis — call strike closest to 105% of price
        target_cc = round(price * 1.05, 2)
        available_calls = calls_df[calls_df["strike"] >= price * 1.02]
        if available_calls.empty:
            available_calls = calls_df
        cc_strike = _nearest_strike(target_cc, available_calls["strike"].values)
        cc_row = calls_df[calls_df["strike"] == cc_strike].iloc[0]
        bid = cc_row.get("bid", 0) or 0
        iv = (cc_row.get("impliedVolatility", 0) or 0) * 100
        delta = cc_row.get("delta", None)
        premium_yield = (bid / price) * 100 if price > 0 else 0.0
        annualized = premium_yield * 365 / out["dte"] if out["dte"] > 0 else 0.0
        out["cc"] = {
            "strike": float(cc_strike),
            "bid": float(bid),
            "premium_yield": round(premium_yield, 2),
            "annualized": round(annualized, 2),
            "iv": round(iv, 2),
            "delta": round(float(delta), 2) if delta is not None else None,
        }

        # Historical vol
        out["hv30"] = round(_hv30(tk), 2) if _hv30(tk) else None

    except Exception as e:
        return {**out, "error": f"Error: {str(e)[:80]}"}

    return out


def build_results_df(results, owned_set):
    """
    Build a display DataFrame from analysis results.
    owned_set: set of ticker strings that are owned (for CC display).
    """
    rows = []
    for r in results:
        ticker = r["ticker"]
        if r["error"]:
            rows.append({
                "Ticker": ticker,
                "Price": None,
                "Expiry": r.get("error", "Error"),
                "DTE": None,
                "CSP Strike": None,
                "CSP Bid": None,
                "CSP Yield%": None,
                "CSP Ann%": None,
                "CSP IV": None,
                "CSP Delta": None,
                "CC Strike": None,
                "CC Bid": None,
                "CC Yield%": None,
                "CC Ann%": None,
                "CC IV": None,
                "CC Delta": None,
                "HV30": None,
                "IV/HV Ratio": None,
            })
            continue

        csp = r["csp"]
        cc = r["cc"] if ticker in owned_set else None
        iv_hv = (round(csp["iv"] / r["hv30"], 2)
                 if csp and r.get("hv30") and r["hv30"] > 0
                 else None)

        rows.append({
            "Ticker": ticker,
            "Price": r["price"],
            "Expiry": r["expiry"],
            "DTE": r["dte"],
            "CSP Strike": csp["strike"],
            "CSP Bid": csp["bid"],
            "CSP Yield%": csp["premium_yield"],
            "CSP Ann%": csp["annualized"],
            "CSP IV": csp["iv"],
            "CSP Delta": csp["delta"],
            "CC Strike": cc["strike"] if cc else None,
            "CC Bid": cc["bid"] if cc else None,
            "CC Yield%": cc["premium_yield"] if cc else None,
            "CC Ann%": cc["annualized"] if cc else None,
            "CC IV": cc["iv"] if cc else None,
            "CC Delta": cc["delta"] if cc else None,
            "HV30": r.get("hv30"),
            "IV/HV Ratio": iv_hv,
        })
    return pd.DataFrame(rows)


# ── STREAMLIT UI ─────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = "TSLA, NVDA, AAPL, MSFT, AMD"

_COL_ORDER = [
    "Ticker", "Price", "Expiry", "DTE",
    "CSP Strike", "CSP Bid", "CSP Yield%", "CSP Ann%", "CSP IV", "CSP Delta",
    "CC Strike", "CC Bid", "CC Yield%", "CC Ann%", "CC IV", "CC Delta",
    "HV30", "IV/HV Ratio",
]


def render_wheel_scanner():
    """Main entry point — renders the Wheel Scanner tab content."""
    st.markdown("### Wheel Strategy Scanner")
    st.markdown(
        "<div style='font-size:.82rem;color:#8b8fa8;margin-bottom:12px'>"
        "Scan tickers for Cash-Secured Put (CSP) and Covered Call (CC) opportunities. "
        "Mark a ticker as <b>Owned</b> to see CC data.</div>",
        unsafe_allow_html=True,
    )

    # ── Input area ───────────────────────────────────────────────────────────
    ticker_text = st.text_area(
        "Tickers (comma-separated)",
        value=DEFAULT_TICKERS,
        height=80,
        key="ws_tickers_input",
    )
    tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]

    col1, col2 = st.columns([1, 5])
    with col1:
        refresh = st.button("🔄 Refresh", use_container_width=True, key="ws_refresh")
    with col2:
        st.markdown(
            f"<div style='font-size:.78rem;color:#5a5d6e;padding-top:6px'>{len(tickers)} ticker(s)</div>",
            unsafe_allow_html=True,
        )

    # ── Owned checkboxes (render before data fetch) ──────────────────────────
    owned_default = {t: False for t in tickers}
    # Persist owned state across refreshes
    if "ws_owned" not in st.session_state:
        st.session_state["ws_owned"] = {}
    for t in tickers:
        if t not in st.session_state["ws_owned"]:
            st.session_state["ws_owned"][t] = False

    st.markdown("#### Owned Tickers")
    owned_cols = st.columns(min(len(tickers), 8))
    for i, t in enumerate(tickers):
        with owned_cols[i % len(owned_cols)]:
            st.session_state["ws_owned"][t] = st.checkbox(
                t, value=st.session_state["ws_owned"].get(t, False),
                key=f"ws_own_{t}",
            )
    owned_set = {t for t in tickers if st.session_state["ws_owned"].get(t)}

    if not tickers:
        st.info("Enter at least one ticker.")
        return

    # ── Fetch & analyze ──────────────────────────────────────────────────────
    if refresh or "ws_results_cache" not in st.session_state:
        with st.spinner("Fetching options data..."):
            results = []
            for t in tickers:
                r = analyze_ticker(t)
                results.append(r)
            st.session_state["ws_results_cache"] = results
            st.session_state["ws_results_tickers"] = tickers

    results = st.session_state.get("ws_results_cache", [])
    cached_tickers = st.session_state.get("ws_results_tickers", [])

    # If tickers changed (user edited text area), re-fetch
    if tickers != cached_tickers:
        with st.spinner("Fetching options data..."):
            results = []
            for t in tickers:
                r = analyze_ticker(t)
                results.append(r)
            st.session_state["ws_results_cache"] = results
            st.session_state["ws_results_tickers"] = tickers

    # ── Build & display DataFrame ────────────────────────────────────────────
    df = build_results_df(results, owned_set)

    if df.empty:
        st.warning("No results to display.")
        return

    # ── Fill None/NaN in numeric columns before styling ────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].where(pd.notna, None)

    # ── Apply styling ────────────────────────────────────────────────────────
    def _highlight_row(row):
        styles = [""] * len(row)
        iv_hv = row.get("IV/HV Ratio")
        dte = row.get("DTE")
        csp_ann = row.get("CSP Ann%")

        for i, col_name in enumerate(row.index):
            s = ""
            # Green background for IV/HV > 1.3
            if col_name == "IV/HV Ratio" and iv_hv is not None and iv_hv > 1.3:
                s += "background-color: #1a3a2a; color: #4cdf8b; font-weight: bold;"
            # Red for DTE < 7
            if col_name == "DTE" and dte is not None and dte < 7:
                s += "background-color: #3a1a1a; color: #ef5350; font-weight: bold;"
            # Bold for CSP Ann% > 30
            if col_name == "CSP Ann%" and csp_ann is not None and csp_ann > 30:
                s += "font-weight: bold;"
            styles[i] = s
        return styles

    styled_df = df.style.apply(_highlight_row, axis=1)

    st.markdown("#### Options Chain Scanner Results")
    st.dataframe(
        styled_df,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "Expiry": st.column_config.TextColumn("Expiry", width="small"),
            "DTE": st.column_config.NumberColumn("DTE", format="%d"),
            "CSP Strike": st.column_config.NumberColumn("CSP Strike", format="$%.2f"),
            "CSP Bid": st.column_config.NumberColumn("CSP Bid", format="$%.2f"),
            "CSP Yield%": st.column_config.NumberColumn("CSP Yield%", format="%.2f%%"),
            "CSP Ann%": st.column_config.NumberColumn("CSP Ann%", format="%.2f%%"),
            "CSP IV": st.column_config.NumberColumn("CSP IV", format="%.1f%%"),
            "CSP Delta": st.column_config.NumberColumn("CSP Delta", format="%.2f"),
            "CC Strike": st.column_config.NumberColumn("CC Strike", format="$%.2f"),
            "CC Bid": st.column_config.NumberColumn("CC Bid", format="$%.2f"),
            "CC Yield%": st.column_config.NumberColumn("CC Yield%", format="%.2f%%"),
            "CC Ann%": st.column_config.NumberColumn("CC Ann%", format="%.2f%%"),
            "CC IV": st.column_config.NumberColumn("CC IV", format="%.1f%%"),
            "CC Delta": st.column_config.NumberColumn("CC Delta", format="%.2f"),
            "HV30": st.column_config.NumberColumn("HV30", format="%.2f%%"),
            "IV/HV Ratio": st.column_config.NumberColumn("IV/HV", format="%.2f"),
        },
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # ── Legenda ──────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.75rem;color:#5a5d6e;margin-top:8px;line-height:1.8'>"
        "🟢 <b>IV/HV &gt; 1.3</b> — elevated IV, good for selling premium &nbsp;&nbsp;|&nbsp;&nbsp;"
        "🔴 <b>DTE &lt; 7</b> — too close to expiry<br>"
        "• CSP strikes at ~90% of spot (10% OTM) &nbsp;&nbsp;|&nbsp;&nbsp;"
        "• CC strikes at ~105% of spot (5% OTM, owned tickers only)"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Per-ticker detail expanders ──────────────────────────────────────────
    st.markdown("#### Per-Ticker Details")
    for r in results:
        t = r["ticker"]
        if r["error"]:
            with st.expander(f"❌ {t} — {r['error']}"):
                st.warning(f"Cannot analyze {t}: {r['error']}")
            continue

        csp = r["csp"]
        cc = r["cc"] if t in owned_set else None
        with st.expander(f"📊 {t} @ ${_fmt(r['price'])} — Exp: {r['expiry']} ({r['dte']}d)"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Cash-Secured Put (10% OTM)**")
                st.metric("Strike", f"${_fmt(csp['strike'], decimals=2)}")
                st.metric("Bid Premium", f"${_fmt(csp['bid'], decimals=2)}")
                st.metric("Yield", _fmt(csp['premium_yield'], suffix='%'))
                st.metric("Annualized Yield", _fmt(csp['annualized'], suffix='%'))
                st.metric("IV", _fmt(csp['iv'], decimals=1, suffix='%'))
                st.metric("Delta", _fmt(csp['delta'], decimals=2))
            with c2:
                if cc:
                    st.markdown("**Covered Call (5% OTM)**")
                    st.metric("Strike", f"${_fmt(cc['strike'], decimals=2)}")
                    st.metric("Bid Premium", f"${_fmt(cc['bid'], decimals=2)}")
                    st.metric("Yield", _fmt(cc['premium_yield'], suffix='%'))
                    st.metric("Annualized Yield", _fmt(cc['annualized'], suffix='%'))
                    st.metric("IV", _fmt(cc['iv'], decimals=1, suffix='%'))
                    st.metric("Delta", _fmt(cc['delta'], decimals=2))
                else:
                    st.markdown("**Covered Call**")
                    st.info("Mark ticker as 'Owned' above to see CC data.")
            st.caption(f"HV30: {_fmt(r.get('hv30'), suffix='%')} | IV/HV: {_fmt(round(csp['iv']/r['hv30'],2) if r.get('hv30') else None, decimals=2)}")