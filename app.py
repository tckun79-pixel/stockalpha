"""
app.py — StockAlpha: Deep Stock Analysis Dashboard
US / SG (SGX) / HK (HKEX) tickers supported.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="StockAlpha | Deep Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

from modules.data_fetcher  import fetch_all, get_current_price, get_currency_symbol
from modules.fundamental   import compute_quarterly_metrics
from modules.technical     import (compute_indicators, find_support_resistance,
                                   build_chart, get_indicator_signals, compute_performance)
from modules.forecast      import run_forecast
from modules.ui_components import (render_metric_card, render_earnings_table,
                                   render_flag_alerts, render_forecast_summary,
                                   render_valuation_table)
from modules.comparison    import (
    fetch_comparison_data, build_normalized_price_chart,
    build_volume_chart, build_rsi_comparison_chart,
    build_drawdown_chart, build_fundamental_comparison,
    build_performance_comparison, build_radar_chart,
    build_correlation_matrix,
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <rect width="32" height="32" rx="8" fill="#00d4aa"/>
        <polyline points="4,24 10,14 16,18 22,8 28,12"
                  stroke="#0f1117" stroke-width="2.5"
                  stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        <circle cx="28" cy="12" r="2.5" fill="#0f1117"/>
      </svg>
      <span style="font-size:1.2rem;font-weight:700;color:#e8eaf0">StockAlpha</span>
    </div>
    <p style="font-size:.72rem;color:#8b8fa8;margin-bottom:20px;margin-left:42px">
      Institutional-grade analysis
    </p>
    """, unsafe_allow_html=True)

    ticker_input = st.text_input(
        "Ticker Symbol",
        placeholder="e.g. AAPL · D05.SI · 0700.HK",
        help="US: AAPL | SGX: D05.SI | HKEX: 0700.HK",
    ).strip().upper()

    market_override = st.selectbox(
        "Market",
        ["Auto-detect", "US (NYSE/NASDAQ)", "SG (SGX)", "HK (HKEX)"],
    )
    mmap = {
        "Auto-detect":      "Auto",
        "US (NYSE/NASDAQ)": "US",
        "SG (SGX)":         "SG",
        "HK (HKEX)":        "HK",
    }
    market_sel = mmap[market_override]

    run_btn = st.button("🔍 Run Deep Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:.75rem;color:#8b8fa8;line-height:1.6">
      <b style="color:#e8eaf0">Supported Markets</b><br>
      🇺🇸 NYSE / NASDAQ (no suffix)<br>
      🇸🇬 SGX — append <code>.SI</code><br>
      🇭🇰 HKEX — append <code>.HK</code>
    </div>
    <div style="font-size:.68rem;color:#5a5d6e;margin-top:16px;font-style:italic;
                border-top:1px solid #1e2130;padding-top:12px;line-height:1.5">
      For informational purposes only.<br>
      Not financial advice. Data via yfinance.
    </div>
    """, unsafe_allow_html=True)


# ── SESSION STATE INIT ────────────────────────────────────────────────────────
for _key, _default in [
    ("analysis_done",   False),
    ("analysis_data",   None),
    ("fund",            None),
    ("df_tech",         None),
    ("sr",              None),
    ("signals",         None),
    ("perf",            None),
    ("chart",           None),
    ("fc",              None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── LANDING (only shown before first analysis) ────────────────────────────────
if not st.session_state["analysis_done"] and not (run_btn and ticker_input):
    st.markdown("""
    <div style="text-align:center;padding:60px 20px 40px">
      <div style="font-size:2.2rem;font-weight:700;color:#e8eaf0;
                  letter-spacing:-.03em;margin-bottom:12px">Deep Stock Analysis</div>
      <div style="font-size:1rem;color:#8b8fa8;max-width:520px;margin:0 auto 32px">
        5-quarter fundamental · 5-year technical · 6–12 month forecast · multi-ticker comparison
      </div>
      <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:48px">
        <span style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:20px;
                     padding:6px 16px;font-size:.8rem;color:#8b8fa8">AAPL · Apple</span>
        <span style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:20px;
                     padding:6px 16px;font-size:.8rem;color:#8b8fa8">NVDA · NVIDIA</span>
        <span style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:20px;
                     padding:6px 16px;font-size:.8rem;color:#8b8fa8">D05.SI · DBS</span>
        <span style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:20px;
                     padding:6px 16px;font-size:.8rem;color:#8b8fa8">0700.HK · Tencent</span>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "📊", "Fundamental",
         "Revenue, EPS, margins, FCF, valuation ratios and red flag detection over 5 quarters."),
        (c2, "📈", "Technical",
         "MACD, RSI, Bollinger Bands, SMA/EMA, ATR, OBV, support & resistance levels."),
        (c3, "🔮", "Forecast",
         "Ensemble model (Linear Regression + Bollinger + Forward P/E) for 6M & 12M targets."),
        (c4, "⚖️", "Compare",
         "Side-by-side comparison of 2–5 tickers: performance, radar scores, correlation & fundamentals."),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;
                        padding:24px;text-align:center;height:160px">
              <div style="font-size:1.8rem;margin-bottom:10px">{icon}</div>
              <div style="font-size:.9rem;font-weight:600;color:#e8eaf0;margin-bottom:8px">{title}</div>
              <div style="font-size:.78rem;color:#8b8fa8;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()


# NEW
# ── FETCH & CACHE when Run button clicked ─────────────────────────────────────
if run_btn and ticker_input:
    with st.spinner(f"⏳ Fetching data for **{ticker_input}**..."):
        _data = fetch_all(ticker_input, market_sel)

    if not _data["valid"]:
        st.error(f"❌ {_data['error']}")
        st.info("💡 US tickers: AAPL | SGX: D05.SI | HKEX: 0700.HK")
        st.stop()

    with st.spinner("⚙️ Running fundamental analysis..."):
        _fund = compute_quarterly_metrics(
            _data["income_stmt"], _data["balance_sheet"], _data["cashflow"], _data["info"]
        )
    with st.spinner("📈 Computing technical indicators..."):
        _df_tech = compute_indicators(_data["price_history"])
        _sr      = find_support_resistance(_df_tech)
        _signals = get_indicator_signals(_df_tech)
        _perf    = compute_performance(_df_tech)
        _chart   = build_chart(_df_tech)
    with st.spinner("🔮 Generating price forecast..."):
        _fc = run_forecast(_data["price_history"], _data["info"], _fund)

    st.session_state.update({
        "analysis_done": True,
        "analysis_data": _data,
        "fund":    _fund,
        "df_tech": _df_tech,
        "sr":      _sr,
        "signals": _signals,
        "perf":    _perf,
        "chart":   _chart,
        "fc":      _fc,
    })

# ── LOAD FROM SESSION STATE (every rerun uses cached results) ─────────────────
data          = st.session_state["analysis_data"]
info          = data["info"]
price_history = data["price_history"]
market        = data["market"]
ticker        = data["ticker"]
fund          = st.session_state["fund"]
df_tech       = st.session_state["df_tech"]
sr            = st.session_state["sr"]
signals       = st.session_state["signals"]
perf          = st.session_state["perf"]
chart         = st.session_state["chart"]
fc            = st.session_state["fc"]

currency  = get_currency_symbol(market, info)
cur_price = get_current_price(info) or float(price_history["Close"].iloc[-1])

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Fundamental",
    "📈 Technical",
    "🔮 Forecast (6–12M)",
    "⚖️ Compare Tickers",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FUNDAMENTAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Last 5 Quarters — Earnings Overview")
    quarters  = fund.get("quarters", [])
    valuation = fund.get("valuation", {})

    if quarters:
        latest = quarters[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            rev = latest.get("Revenue")
            rv  = (f"{currency}{rev/1e9:.2f}B" if rev and rev >= 1e9 else
                   f"{currency}{rev/1e6:.0f}M" if rev else "—")
            render_metric_card("Latest Revenue", rv,
                               f"{latest.get('Rev YoY%', '—')}% YoY",
                               (latest.get("Rev YoY%") or 0) > 0)
        with c2:
            eps = latest.get("EPS Diluted")
            render_metric_card("EPS (Diluted)",
                               f"{eps:.2f}" if eps else "—",
                               f"Basic: {latest.get('EPS Basic', '—')}")
        with c3:
            nm = latest.get("Net Margin%")
            render_metric_card("Net Margin",
                               f"{nm:.1f}%" if nm else "—",
                               f"Op: {latest.get('Op Margin%', '—')}%",
                               (nm or 0) > 0)
        with c4:
            pe = valuation.get("P/E (Trailing)")
            render_metric_card("P/E (Trailing)",
                               f"{pe:.1f}x" if pe else "—",
                               f"Fwd: {valuation.get('P/E (Forward)', '—')}")

        st.markdown("<br>", unsafe_allow_html=True)
        render_earnings_table(quarters)

        st.markdown("#### Earnings Trend Charts")
        ql = [q["Quarter"] for q in quarters][::-1]

        def dark_fig(title, h=260):
            f = go.Figure()
            f.update_layout(
                title=title, height=h,
                plot_bgcolor="#0f1117", paper_bgcolor="#1a1d27",
                font=dict(color="#e8eaf0", size=10),
                xaxis=dict(gridcolor="#1e2433"),
                yaxis=dict(gridcolor="#1e2433"),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            return f

        col1, col2 = st.columns(2)
        with col1:
            rv2 = [(q.get("Revenue") or 0) / 1e9 for q in quarters][::-1]
            f = dark_fig("Revenue (Billions)")
            f.add_trace(go.Bar(x=ql, y=rv2, marker_color="#00d4aa"))
            st.plotly_chart(f, width="stretch")
        with col2:
            ev = [q.get("EPS Diluted") or 0 for q in quarters][::-1]
            ec = ["#26a69a" if v >= 0 else "#ef5350" for v in ev]
            f = dark_fig("EPS (Diluted)")
            f.add_trace(go.Bar(x=ql, y=ev, marker_color=ec))
            st.plotly_chart(f, width="stretch")

        col3, col4 = st.columns(2)
        with col3:
            f = dark_fig("Margin Trends (%)")
            for key, name, color in [
                ("Gross Margin%", "Gross", "#60a5fa"),
                ("Op Margin%",    "Op",    "#f59e0b"),
                ("Net Margin%",   "Net",   "#00d4aa"),
            ]:
                vals = [q.get(key) or 0 for q in quarters][::-1]
                f.add_trace(go.Scatter(
                    x=ql, y=vals, name=name,
                    line=dict(color=color, width=2), mode="lines+markers",
                ))
            f.update_layout(legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(f, width="stretch")
        with col4:
            fv = [(q.get("FCF") or 0) / 1e9 for q in quarters][::-1]
            fc_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in fv]
            f = dark_fig("Free Cash Flow (Billions)")
            f.add_trace(go.Bar(x=ql, y=fv, marker_color=fc_colors))
            st.plotly_chart(f, width="stretch")

    st.markdown("#### 🚩 Earnings Quality Check")
    render_flag_alerts(fund.get("flags", []))

    st.markdown("#### Valuation Ratios")
    cv1, cv2 = st.columns([1, 2])
    with cv1:
        render_valuation_table(valuation)
    with cv2:
        if fund.get("summary_text"):
            st.markdown(f"""
            <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:8px;
                        padding:16px 20px;font-size:.85rem;color:#c4c6d4;line-height:1.7">
              <b style="color:#00d4aa">Analysis Summary</b><br><br>
              {fund['summary_text']}
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TECHNICAL
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 1-Year Price Chart (from 5Y dataset)")
    st.plotly_chart(chart, width="stretch")

    st.markdown("#### Indicator Signals")
    sig_cols = st.columns(min(len(signals), 4))
    for i, (k, v) in enumerate(signals.items()):
        with sig_cols[i % len(sig_cols)]:
            st.markdown(f"""
            <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:8px;
                        padding:12px 14px;margin-bottom:10px">
              <div style="font-size:.72rem;color:#8b8fa8;margin-bottom:4px">{k}</div>
              <div style="font-size:.85rem;color:#e8eaf0">{v}</div>
            </div>""", unsafe_allow_html=True)

    ts1, ts2 = st.columns(2)
    with ts1:
        st.markdown("#### Support & Resistance")
        sup = sr.get("support", [])
        res = sr.get("resistance", [])
        sr_rows_html = ""
        for i in range(max(len(sup), len(res))):
            s  = f"{currency}{sup[i]:,.2f}" if i < len(sup) else "—"
            r  = f"{currency}{res[i]:,.2f}" if i < len(res) else "—"
            bg = "#1e2130" if i % 2 == 0 else "#1a1d27"
            sr_rows_html += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:8px 14px;color:#26a69a;font-family:\'Courier New\',monospace;font-size:.82rem">🟢 {s}</td>'
                f'<td style="padding:8px 14px;color:#ef5350;font-family:\'Courier New\',monospace;font-size:.82rem;text-align:right">🔴 {r}</td>'
                f'</tr>'
            )
        st.markdown(f"""
        <div style="border:1px solid #2a2d3a;border-radius:8px;overflow:hidden">
          <table style="width:100%;border-collapse:collapse">
            <thead><tr style="border-bottom:1px solid #2a2d3a">
              <th style="padding:8px 14px;color:#8b8fa8;font-size:.75rem;text-align:left">Support</th>
              <th style="padding:8px 14px;color:#8b8fa8;font-size:.75rem;text-align:right">Resistance</th>
            </tr></thead>
            <tbody>{sr_rows_html}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

    with ts2:
        st.markdown("#### 5-Year Performance")
        for k, v in perf.items():
            color = ("#ef5350" if k == "Max Drawdown"
                     else "#26a69a" if not str(v).startswith("-")
                     else "#ef5350")
            st.markdown(f"""
            <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:8px;
                        padding:12px 16px;margin-bottom:8px;
                        display:flex;justify-content:space-between">
              <span style="color:#8b8fa8;font-size:.82rem">{k}</span>
              <span style="color:{color};font-size:.85rem;
                           font-family:'Courier New',monospace">{v}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.error(fc["disclaimer"], icon="⚠️")

    if fc.get("error"):
        st.warning(fc["error"])
    else:
        render_forecast_summary(fc["6m"], fc["12m"], fc["confidence"])

        if fc.get("chart"):
            st.plotly_chart(fc["chart"], width="stretch")

        fc1, fc2 = st.columns(2)
        with fc1:
            with st.expander("📐 Key Assumptions", expanded=True):
                for k, v in fc.get("assumptions", {}).items():
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;
                                padding:6px 0;border-bottom:1px solid #1e2130">
                      <span style="color:#8b8fa8;font-size:.8rem">{k}</span>
                      <span style="color:#e8eaf0;font-size:.8rem;
                                   font-family:'Courier New',monospace">{v}</span>
                    </div>""", unsafe_allow_html=True)
        with fc2:
            with st.expander("🔬 Methodology"):
                st.markdown("""
                <div style="font-size:.8rem;color:#8b8fa8;line-height:1.7">
                  <b style="color:#e8eaf0">3-Model Ensemble</b><br><br>
                  <b>1. Linear Regression (40%)</b> — trend line extrapolation on 1Y price data.<br><br>
                  <b>2. Bollinger Mean-Reversion (30%)</b> — forecasts partial reversion toward
                  20-day rolling mean.<br><br>
                  <b>3. Forward P/E Implied (30%)</b> — forward EPS × current P/E multiple.
                  Falls back to 55/45 split if unavailable.<br><br>
                  <b>Confidence</b> — model spread vs current price:
                  &lt;10% = High · &lt;25% = Medium · else Low.
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COMPARE TICKERS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ⚖️ Multi-Ticker Comparison")
    st.markdown(
        f"<div style='font-size:.82rem;color:#8b8fa8;margin-bottom:16px'>"
        f"Compare 2–5 tickers side by side. Current ticker "
        f"<code style='color:#00d4aa'>{ticker}</code> is pre-loaded."
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Persistent state init ──────────────────────────────────────────────
    if "cmp_datasets" not in st.session_state: st.session_state["cmp_datasets"] = {}
    if "cmp_period"   not in st.session_state: st.session_state["cmp_period"]   = "1Y"
    if "cmp_tickers"  not in st.session_state: st.session_state["cmp_tickers"]  = ""
    if "cmp_ran"      not in st.session_state: st.session_state["cmp_ran"]      = False

    # ── Input row ──────────────────────────────────────────────────────────
    ci1, ci2, ci3 = st.columns([3, 1, 1])
    with ci1:
        extra_input = st.text_input(
            "Add tickers to compare (comma-separated)",
            value=st.session_state["cmp_tickers"],
            placeholder="e.g. MSFT, GOOGL, META  or  D05.SI, O39.SI",
            key="cmp_input_box",
        )
    with ci2:
        period_sel = st.selectbox(
            "Period", ["1M", "3M", "6M", "1Y", "2Y", "5Y"],
            index=["1M", "3M", "6M", "1Y", "2Y", "5Y"].index(
                st.session_state["cmp_period"]
            ),
            key="cmp_period_box",
        )
    with ci3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_compare = st.button("🔄 Compare", use_container_width=True, key="btn_compare")

    # ── Fetch on button click → store in session_state ─────────────────────
    if run_compare:
        extra_tickers = [t.strip().upper() for t in extra_input.split(",") if t.strip()]
        all_tickers   = list(dict.fromkeys([ticker] + extra_tickers))
        if len(all_tickers) > 5:
            st.warning("Maximum 5 tickers. Using first 5.")
            all_tickers = all_tickers[:5]
        if len(all_tickers) < 2:
            st.warning("Enter at least one additional ticker.")
        else:
            with st.spinner(f"Fetching: {', '.join(all_tickers)}..."):
                datasets = fetch_comparison_data(all_tickers)
            st.session_state["cmp_datasets"] = datasets
            st.session_state["cmp_tickers"]  = extra_input
            st.session_state["cmp_period"]   = period_sel
            st.session_state["cmp_ran"]      = True

    # ── Render from session_state (persists across reruns) ─────────────────
    if not st.session_state["cmp_ran"]:
        st.info("👆 Enter additional tickers above and click **🔄 Compare** to start.")
    else:
        cmp_datasets   = st.session_state["cmp_datasets"]
        cmp_period     = st.session_state["cmp_period"]

        for t, d in cmp_datasets.items():
            if not d.get("valid"):
                st.error(f"❌ {t}: {d.get('error', 'Failed to fetch')}")

        valid_datasets = {t: d for t, d in cmp_datasets.items() if d.get("valid")}

        if len(valid_datasets) < 2:
            st.error("Need at least 2 valid tickers to compare.")
        else:
            CMP_COLORS = ["#00d4aa", "#60a5fa", "#f59e0b", "#a78bfa", "#f87171"]

            # ── Performance Summary Table ──────────────────────────────────
            st.markdown("#### 📋 Performance Summary")
            perf_rows = build_performance_comparison(valid_datasets, cmp_period)

            hdr = "".join(
                f'<th style="padding:8px 14px;text-align:right;color:#8b8fa8;font-size:.75rem">{h}</th>'
                for h in ["Ticker", "Name", "Return", "Ann. Vol", "Max DD", "Sharpe", "Beta"]
            )
            p_rows_html = ""
            for idx, row in enumerate(perf_rows):
                bg    = "#1e2130" if idx % 2 == 0 else "#1a1d27"
                color = CMP_COLORS[idx % len(CMP_COLORS)]
                ret   = row.get("Return", "—")
                ret_c = "#26a69a" if "+" in str(ret) else "#ef5350" if "-" in str(ret) else "#e8eaf0"
                p_rows_html += (
                    f'<tr style="background:{bg}">'
                    f'<td style="padding:8px 14px;font-weight:600;color:{color};'
                    f"font-family:'Courier New',monospace;font-size:.82rem\">{row['Ticker']}</td>"
                    f'<td style="padding:8px 14px;font-size:.78rem;color:#8b8fa8">{row.get("Name", "—")}</td>'
                    f'<td style="padding:8px 14px;text-align:right;color:{ret_c};'
                    f"font-family:'Courier New',monospace;font-size:.82rem\">{ret}</td>"
                    f'<td style="padding:8px 14px;text-align:right;color:#e8eaf0;'
                    f"font-family:'Courier New',monospace;font-size:.82rem\">{row.get('Ann. Vol', '—')}</td>"
                    f'<td style="padding:8px 14px;text-align:right;color:#ef5350;'
                    f"font-family:'Courier New',monospace;font-size:.82rem\">{row.get('Max DD', '—')}</td>"
                    f'<td style="padding:8px 14px;text-align:right;color:#e8eaf0;'
                    f"font-family:'Courier New',monospace;font-size:.82rem\">{row.get('Sharpe', '—')}</td>"
                    f'<td style="padding:8px 14px;text-align:right;color:#e8eaf0;'
                    f"font-family:'Courier New',monospace;font-size:.82rem\">{row.get('Beta', '—')}</td>"
                    f'</tr>'
                )
            st.markdown(f"""
            <div style="overflow-x:auto;border-radius:8px;border:1px solid #2a2d3a;margin-bottom:20px">
              <table style="width:100%;border-collapse:collapse">
                <thead><tr style="border-bottom:1px solid #2a2d3a">{hdr}</tr></thead>
                <tbody>{p_rows_html}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

            # ── Normalized Price Chart ─────────────────────────────────────
            st.plotly_chart(
                build_normalized_price_chart(valid_datasets, cmp_period),
                width="stretch",
            )

            # ── Radar + Correlation ────────────────────────────────────────
            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(build_radar_chart(valid_datasets), width="stretch")
            with r2:
                corr_fig = build_correlation_matrix(valid_datasets, cmp_period)
                if corr_fig:
                    st.plotly_chart(corr_fig, width="stretch")

            # ── RSI ────────────────────────────────────────────────────────
            st.plotly_chart(
                build_rsi_comparison_chart(valid_datasets, cmp_period),
                width="stretch",
            )

            # ── Drawdown ──────────────────────────────────────────────────
            st.plotly_chart(
                build_drawdown_chart(valid_datasets, cmp_period),
                width="stretch",
            )

            # ── Volume ────────────────────────────────────────────────────
            st.plotly_chart(
                build_volume_chart(valid_datasets, cmp_period),
                width="stretch",
            )

            # ── Fundamental Comparison Table ──────────────────────────────
            st.markdown("#### 📊 Fundamental Metrics Comparison")
            fund_rows     = build_fundamental_comparison(valid_datasets)
            valid_tickers = list(valid_datasets.keys())

            fh = '<th style="padding:8px 14px;text-align:left;color:#8b8fa8;font-size:.75rem">Metric</th>'
            for t in valid_tickers:
                fh += (f'<th style="padding:8px 14px;text-align:right;'
                       f'color:#8b8fa8;font-size:.75rem">{t}</th>')

            fund_rows_html = ""
            for idx, row in enumerate(fund_rows):
                bg    = "#1e2130" if idx % 2 == 0 else "#1a1d27"
                cells = f'<td style="padding:8px 14px;color:#8b8fa8;font-size:.78rem">{row["Metric"]}</td>'
                for t in valid_tickers:
                    v = row.get(t, "—")
                    cells += (f'<td style="padding:8px 14px;text-align:right;color:#e8eaf0;'
                              f"font-size:.82rem;font-family:'Courier New',monospace\">{v}</td>")
                fund_rows_html += f'<tr style="background:{bg}">{cells}</tr>'

            st.markdown(f"""
            <div style="overflow-x:auto;border-radius:8px;border:1px solid #2a2d3a">
              <table style="width:100%;border-collapse:collapse">
                <thead><tr style="border-bottom:1px solid #2a2d3a">{fh}</tr></thead>
                <tbody>{fund_rows_html}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

            # ── Clear button ───────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Comparison", key="btn_clear_cmp"):
                st.session_state["cmp_datasets"] = {}
                st.session_state["cmp_ran"]      = False
                st.session_state["cmp_tickers"]  = ""
                st.rerun()
