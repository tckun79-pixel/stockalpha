"""
data_fetcher.py — Market-aware ticker data fetcher using yfinance.
Supports US (no suffix), SG (.SI), HK (.HK) tickers.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from typing import Optional
from datetime import datetime


def detect_market(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if ticker.endswith(".SI"):
        return "SGX"
    elif ticker.endswith(".HK"):
        return "HKEX"
    return "US"


def normalize_ticker(ticker: str, market_override: str = "Auto") -> str:
    ticker = ticker.upper().strip()
    if market_override == "SG" and not ticker.endswith(".SI"):
        return ticker + ".SI"
    elif market_override == "HK" and not ticker.endswith(".HK"):
        return ticker + ".HK"
    return ticker


@st.cache_data(ttl=3600)
def fetch_all(ticker_raw: str, market_override: str = "Auto") -> dict:
    ticker = normalize_ticker(ticker_raw, market_override)
    market = detect_market(ticker)

    result = {
        "ticker": ticker,
        "market": market,
        "valid": False,
        "error": None,
        "info": {},
        "price_history": pd.DataFrame(),
        "income_stmt": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "earnings_dates": [],
    }

    try:
        yf_obj = yf.Ticker(ticker)
        info = yf_obj.info or {}

        if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
            fast = yf_obj.fast_info
            if fast.last_price is None:
                result["error"] = f"No data found for ticker '{ticker}'. Check the symbol and market suffix."
                return result

        result["info"] = info

        hist = yf_obj.history(period="5y", interval="1d", auto_adjust=True)
        if hist.empty:
            result["error"] = f"No price history for '{ticker}'."
            return result
        result["price_history"] = hist

        # Extract earnings timestamps for post-earnings move analysis
        earnings_ts = info.get("earningsTimestamp")
        if earnings_ts is not None:
            if isinstance(earnings_ts, list):
                earnings_ts = earnings_ts[-1]
            # fetch quarterly earnings dates for historical lookback
            try:
                earnings_df = yf_obj.earnings_dates
                if earnings_df is not None and not earnings_df.empty:
                    result["earnings_dates"] = [
                        ts.to_pydatetime().timestamp()
                        for ts in earnings_df.index[:12]
                    ]
            except Exception:
                pass

        inc = yf_obj.quarterly_income_stmt
        bal = yf_obj.quarterly_balance_sheet
        cf  = yf_obj.quarterly_cashflow

        result["income_stmt"]  = inc.iloc[:, :5] if inc is not None and not inc.empty else pd.DataFrame()
        result["balance_sheet"] = bal.iloc[:, :5] if bal is not None and not bal.empty else pd.DataFrame()
        result["cashflow"]     = cf.iloc[:, :5]  if cf  is not None and not cf.empty  else pd.DataFrame()

        result["valid"] = True

    except Exception as e:
        result["error"] = f"Data fetch failed: {str(e)}"

    return result


def get_current_price(info: dict) -> Optional[float]:
    for key in ["currentPrice", "regularMarketPrice", "previousClose"]:
        val = info.get(key)
        if val and val > 0:
            return val
    return None


def get_currency_symbol(market: str, info: dict) -> str:
    currency = info.get("currency", "")
    mapping = {"USD": "$", "SGD": "S$", "HKD": "HK$", "GBP": "£", "EUR": "€"}
    if currency in mapping:
        return mapping[currency]
    if market == "SGX":
        return "S$"
    if market == "HKEX":
        return "HK$"
    return "$"
