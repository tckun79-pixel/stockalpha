"""
ui_components.py — Reusable Streamlit UI components for StockAlpha.
"""

import streamlit as st
from typing import Optional


def render_metric_card(label, value, delta=None, positive=None):
    delta_html = ""
    if delta:
        color = "#26a69a" if positive else "#ef5350" if positive is False else "#94a3b8"
        delta_html = f'<div style="font-size:0.78rem;color:{color};margin-top:2px">{delta}</div>'
    st.markdown(f"""
    <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:8px;
                padding:14px 18px;min-height:80px">
      <div style="font-size:0.72rem;color:#8b8fa8;text-transform:uppercase;
                  letter-spacing:.06em;margin-bottom:4px">{label}</div>
      <div style="font-size:1.3rem;font-weight:600;color:#e8eaf0;
                  font-family:'Courier New',monospace">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


def render_earnings_table(quarters: list):
    if not quarters:
        st.info("No quarterly earnings data available.")
        return

    def fmt(val, pct=False, money=True):
        if val is None: return "—"
        if pct:
            c = "#26a69a" if val > 0 else "#ef5350"
            return f'<span style="color:{c}">{val:+.1f}%</span>'
        if money:
            if abs(val) >= 1e9: return f"B"
            if abs(val) >= 1e6: return f"M"
            return f""
        return f"{val:.2f}"

    headers = ["Metric"] + [q["Quarter"] for q in quarters]
    header_html = "".join(
        f'<th style="padding:8px 12px;text-align:right;color:#8b8fa8;font-size:.75rem">{h}</th>'
        for h in headers)

    rows_config = [
        ("Revenue",      "Revenue",      False, True),
        ("Rev YoY%",     "Rev YoY%",     True,  False),
        ("Rev QoQ%",     "Rev QoQ%",     True,  False),
        ("Gross Margin%","Gross Margin%",True,  False),
        ("Op Margin%",   "Op Margin%",   True,  False),
        ("Net Margin%",  "Net Margin%",  True,  False),
        ("EPS Diluted",  "EPS Diluted",  False, False),
        ("FCF",          "FCF",          False, True),
        ("D/E Ratio",    "D/E Ratio",    False, False),
        ("Current Ratio","Current Ratio",False, False),
    ]

    rows_html = ""
    for idx, (display, key, is_pct, is_money) in enumerate(rows_config):
        bg = "#1e2130" if idx % 2 == 0 else "#1a1d27"
        cells = f'<td style="padding:8px 12px;color:#8b8fa8;font-size:.78rem">{display}</td>'
        for q in quarters:
            v = fmt(q.get(key), pct=is_pct, money=is_money)
            cells += (f'<td style="padding:8px 12px;text-align:right;font-size:.82rem;'
                      f'font-family:\'Courier New\',monospace;color:#e8eaf0">{v}</td>')
        rows_html += f'<tr style="background:{bg}">{cells}</tr>'

    st.markdown(f"""
    <div style="overflow-x:auto;border-radius:8px;border:1px solid #2a2d3a">
      <table style="width:100%;border-collapse:collapse">
        <thead><tr style="border-bottom:1px solid #2a2d3a">{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)


def render_flag_alerts(flags: list):
    if not flags:
        st.success("✅ No red flags detected in the latest quarter.")
        return
    for flag in flags:
        st.warning(flag)


def render_forecast_summary(d6, d12, confidence):
    cc = {"High":"#26a69a","Medium":"#f59e0b","Low":"#ef5350"}.get(confidence, "#8b8fa8")

    def row(label, d):
        b = f"{d.get('bull',0):.2f}" if d.get('bull') else "—"
        m = f"{d.get('base',0):.2f}" if d.get('base') else "—"
        r = f"{d.get('bear',0):.2f}" if d.get('bear') else "—"
        mn = "'Courier New',monospace"
        return (f'<tr><td style="padding:10px 14px;color:#8b8fa8;font-size:.82rem">{label}</td>'
                f'<td style="padding:10px 14px;text-align:right;color:#26a69a;font-family:{mn}">{b}</td>'
                f'<td style="padding:10px 14px;text-align:right;color:#00d4aa;font-family:{mn}">{m}</td>'
                f'<td style="padding:10px 14px;text-align:right;color:#ef5350;font-family:{mn}">{r}</td></tr>')

    st.markdown(f"""
    <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:8px;
                overflow:hidden;margin-bottom:16px">
      <div style="padding:12px 16px;border-bottom:1px solid #2a2d3a;
                  display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:.82rem;font-weight:600;color:#e8eaf0">Price Targets</span>
        <span style="font-size:.75rem;font-weight:600;color:{cc};
                     background:{cc}22;padding:3px 10px;border-radius:20px">
          Confidence: {confidence}</span>
      </div>
      <table style="width:100%;border-collapse:collapse">
        <thead><tr style="border-bottom:1px solid #2a2d3a">
          <th style="padding:8px 14px;text-align:left;color:#8b8fa8;font-size:.75rem">Horizon</th>
          <th style="padding:8px 14px;text-align:right;color:#26a69a;font-size:.75rem">Bull 🟢</th>
          <th style="padding:8px 14px;text-align:right;color:#00d4aa;font-size:.75rem">Base ⚪</th>
          <th style="padding:8px 14px;text-align:right;color:#ef5350;font-size:.75rem">Bear 🔴</th>
        </tr></thead>
        <tbody>{row("6 Months", d6)}{row("12 Months", d12)}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)


def render_valuation_table(valuation: dict):
    if not valuation: return
    rows = ""
    for k, v in valuation.items():
        vs = f"{v:.2f}" if isinstance(v, float) else str(v) if v is not None else "—"
        rows += (f'<tr style="border-bottom:1px solid #1e2433">'
                 f'<td style="padding:8px 14px;color:#8b8fa8;font-size:.82rem">{k}</td>'
                 f'<td style="padding:8px 14px;text-align:right;color:#e8eaf0;'
                 f'font-size:.82rem;font-family:\'Courier New\',monospace">{vs}</td></tr>')
    st.markdown(f"""
    <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:8px;overflow:hidden">
      <table style="width:100%;border-collapse:collapse">{rows}</table>
    </div>""", unsafe_allow_html=True)
