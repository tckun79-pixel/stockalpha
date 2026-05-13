"""
fundamental.py — 5-quarter fundamental analysis engine.
"""

import pandas as pd
import numpy as np
from typing import Optional


def safe_get(df: pd.DataFrame, row: str, col_idx: int = 0) -> Optional[float]:
    try:
        val = df.loc[row].iloc[col_idx]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def pct_change(current, previous) -> Optional[float]:
    if current is None or previous is None or previous == 0:
        return None
    return round((current - previous) / abs(previous) * 100, 2)


def compute_quarterly_metrics(income, balance, cashflow, info) -> dict:
    results = {"quarters": [], "flags": [], "summary_text": ""}

    if income.empty:
        results["flags"].append("No quarterly income statement data available.")
        return results

    n_quarters = min(5, income.shape[1])

    def find_row(df, keys):
        for k in keys:
            if k in df.index:
                return k
        return None

    rev_row  = find_row(income,  ["Total Revenue", "TotalRevenue"])
    gp_row   = find_row(income,  ["Gross Profit", "GrossProfit"])
    op_row   = find_row(income,  ["Operating Income", "OperatingIncome", "EBIT"])
    ni_row   = find_row(income,  ["Net Income", "NetIncome"])
    epsb_row = find_row(income,  ["Basic EPS", "BasicEPS"])
    epsd_row = find_row(income,  ["Diluted EPS", "DilutedEPS"])
    ocf_row  = find_row(cashflow, ["Operating Cash Flow", "OperatingCashFlow", "Total Cash From Operating Activities"]) if not cashflow.empty else None
    cap_row  = find_row(cashflow, ["Capital Expenditure", "CapitalExpenditure"]) if not cashflow.empty else None
    td_row   = find_row(balance, ["Total Debt", "TotalDebt", "Long Term Debt"]) if not balance.empty else None
    eq_row   = find_row(balance, ["Stockholders Equity", "StockholdersEquity", "Total Stockholder Equity"]) if not balance.empty else None
    ca_row   = find_row(balance, ["Current Assets", "TotalCurrentAssets"]) if not balance.empty else None
    cl_row   = find_row(balance, ["Current Liabilities", "TotalCurrentLiabilities"]) if not balance.empty else None

    quarters_data = []
    for i in range(n_quarters):
        col_date     = income.columns[i]
        quarter_label = col_date.strftime("%b %Y") if hasattr(col_date, "strftime") else str(col_date)[:10]

        revenue      = safe_get(income, rev_row, i) if rev_row else None
        prev_revenue = safe_get(income, rev_row, i + 1) if rev_row and i + 1 < income.shape[1] else None
        yoy_revenue  = safe_get(income, rev_row, i + 4) if rev_row and i + 4 < income.shape[1] else None

        gross_profit = safe_get(income, gp_row,  i) if gp_row  else None
        op_income    = safe_get(income, op_row,   i) if op_row   else None
        net_income   = safe_get(income, ni_row,   i) if ni_row   else None
        eps_basic    = safe_get(income, epsb_row,  i) if epsb_row else None
        eps_diluted  = safe_get(income, epsd_row,  i) if epsd_row else None

        gross_margin = round(gross_profit / revenue * 100, 2) if gross_profit and revenue else None
        op_margin    = round(op_income    / revenue * 100, 2) if op_income    and revenue else None
        net_margin   = round(net_income   / revenue * 100, 2) if net_income   and revenue else None

        ocf   = safe_get(cashflow, ocf_row, i) if ocf_row else None
        capex = safe_get(cashflow, cap_row, i) if cap_row else None
        fcf   = (ocf + capex) if (ocf is not None and capex is not None) else ocf

        td  = safe_get(balance, td_row, i) if td_row else None
        eq  = safe_get(balance, eq_row, i) if eq_row else None
        ca  = safe_get(balance, ca_row, i) if ca_row else None
        cl  = safe_get(balance, cl_row, i) if cl_row else None

        quarters_data.append({
            "Quarter":       quarter_label,
            "Revenue":       revenue,
            "Rev QoQ%":      pct_change(revenue, prev_revenue),
            "Rev YoY%":      pct_change(revenue, yoy_revenue),
            "Gross Profit":  gross_profit,
            "Gross Margin%": gross_margin,
            "Op Income":     op_income,
            "Op Margin%":    op_margin,
            "Net Income":    net_income,
            "Net Margin%":   net_margin,
            "EPS Basic":     eps_basic,
            "EPS Diluted":   eps_diluted,
            "FCF":           fcf,
            "D/E Ratio":     round(td / eq, 2) if (td and eq and eq != 0) else None,
            "Current Ratio": round(ca / cl, 2) if (ca and cl and cl != 0) else None,
        })

    results["quarters"] = quarters_data

    flags = []
    if len(quarters_data) >= 2:
        latest, prior = quarters_data[0], quarters_data[1]
        if (latest.get("Rev QoQ%") or 0) < -10:
            flags.append(f"⚠️ Revenue declined {abs(latest['Rev QoQ%']):.1f}% QoQ")
        if latest.get("Gross Margin%") and prior.get("Gross Margin%"):
            if latest["Gross Margin%"] < prior["Gross Margin%"] - 3:
                flags.append(f"⚠️ Gross margin compressed from {prior['Gross Margin%']:.1f}% → {latest['Gross Margin%']:.1f}%")
        if latest.get("D/E Ratio") and prior.get("D/E Ratio") and prior["D/E Ratio"] > 0:
            if latest["D/E Ratio"] > prior["D/E Ratio"] * 1.2:
                flags.append(f"⚠️ D/E ratio rose >20% ({prior['D/E Ratio']:.2f} → {latest['D/E Ratio']:.2f})")
        if (latest.get("FCF") or 0) < 0:
            flags.append("⚠️ Negative Free Cash Flow this quarter")
        if (latest.get("Net Margin%") or 0) < 0:
            flags.append("⚠️ Net loss this quarter")

    results["flags"] = flags

    if quarters_data:
        q = quarters_data[0]
        rev_trend = "growing" if (q.get("Rev YoY%") or 0) > 0 else "declining"
        results["summary_text"] = (
            f"Revenue is {rev_trend} YoY ({q.get('Rev YoY%', 'N/A')}%). "
            f"Latest net margin: {q.get('Net Margin%', 'N/A')}%. "
            f"EPS (diluted): {q.get('EPS Diluted', 'N/A')}. "
            f"{'No red flags detected.' if not flags else f'{len(flags)} concern(s) flagged.'}"
        )

    results["valuation"] = {
        "P/E (Trailing)": info.get("trailingPE"),
        "P/E (Forward)":  info.get("forwardPE"),
        "P/S Ratio":      info.get("priceToSalesTrailing12Months"),
        "P/B Ratio":      info.get("priceToBook"),
        "ROE %":          round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else None,
        "ROA %":          round(info.get("returnOnAssets",  0) * 100, 2) if info.get("returnOnAssets")  else None,
        "Profit Margin %": round(info.get("profitMargins",  0) * 100, 2) if info.get("profitMargins")  else None,
    }

    return results
