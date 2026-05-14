"""
ticker_analysis.py — IV Rank, earnings calendar, EPS beat/miss history,
and average post-earnings move, all via yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional


def compute_iv_rank_percentile(price_history: pd.DataFrame) -> dict:
    """
    Estimate IV Rank and IV Percentile from daily close returns as a proxy.
    IV Rank  = (current 20d vol - min(252d vol)) / (max(252d vol) - min(252d vol))
    IV Perc  = percentile of current 20d vol over the past 252 days.
    """
    if price_history is None or price_history.empty:
        return {"iv_rank": None, "iv_percentile": None}

    closes = price_history["Close"]
    if len(closes) < 252:
        return {"iv_rank": None, "iv_percentile": None}

    # Daily log returns → annualized vol
    log_ret = np.log(closes / closes.shift(1))
    rolling_20d = log_ret.rolling(20).std() * np.sqrt(252)

    current_vol = rolling_20d.iloc[-1]
    vol_series = rolling_20d.dropna()

    if len(vol_series) < 2 or pd.isna(current_vol) or current_vol == 0:
        return {"iv_rank": None, "iv_percentile": None}

    min_vol = vol_series.min()
    max_vol = vol_series.max()

    iv_rank = (current_vol - min_vol) / (max_vol - min_vol) if (max_vol - min_vol) > 0 else None
    iv_percentile = (vol_series < current_vol).sum() / len(vol_series)

    return {
        "iv_rank": round(iv_rank * 100, 1) if iv_rank is not None else None,
        "iv_percentile": round(iv_percentile * 100, 1) if iv_percentile is not None else None,
        "current_vol": round(current_vol * 100, 2),
    }


def get_next_earnings_date(info: dict) -> Optional[dict]:
    """
    Extract next earnings date from yfinance info dict.
    Returns date string and days-away countdown.
    """
    er = info.get("earningsTimestamp")
    if er is None:
        er = info.get("earningsDate")

    if er is None:
        return None

    # Could be a list or a single timestamp
    if isinstance(er, list):
        er = er[-1]

    try:
        # er is often a unix timestamp
        dt = pd.to_datetime(er, unit="s")
    except Exception:
        try:
            dt = pd.to_datetime(er)
        except Exception:
            return None

    now = pd.Timestamp.now(tz=None)
    if dt.tz is not None:
        dt = dt.tz_localize(None)

    days_away = (dt - now).days
    return {
        "date": dt.strftime("%b %d, %Y"),
        "days_away": days_away,
        "is_past": days_away < 0,
    }


def get_earnings_history(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch last 8 quarters of EPS actual vs estimate from yfinance earnings history.
    Tries multiple yfinance attributes in order: earnings_dates, earnings_history,
    quarterly_earnings. Returns whatever is available, or an empty DataFrame with
    a clear message if all are None.
    """
    try:
        yf_obj = yf.Ticker(ticker)
    except Exception:
        return None

    # Try attributes in order of preference
    earnings = None
    source = None
    for attr, name in [
        ("earnings_dates", "earnings_dates"),
        ("earnings_history", "earnings_history"),
        ("quarterly_earnings", "quarterly_earnings"),
    ]:
        try:
            candidate = getattr(yf_obj, attr, None)
            if candidate is not None and not (isinstance(candidate, pd.DataFrame) and candidate.empty):
                earnings = candidate
                source = name
                break
        except Exception:
            continue

    if earnings is None:
        return pd.DataFrame(columns=["Quarter", "Estimate", "Actual", "Beat", "Surprise %"])

    # earnings_history returns a DataFrame with a different structure
    if source == "earnings_history":
        earnings = earnings.sort_index(ascending=False).head(8)
        rows = []
        for idx, row in earnings.iterrows():
            try:
                est = row.get("Estimate")
                actual = row.get("Actual")
            except Exception:
                continue
            if pd.isna(est) and pd.isna(actual):
                continue
            est_v = float(est) if pd.notna(est) else None
            act_v = float(actual) if pd.notna(actual) else None

            beat = None
            surp_pct = None
            if est_v is not None and act_v is not None and est_v != 0:
                beat = act_v >= est_v
                surp_pct = round((act_v - est_v) / abs(est_v) * 100, 2)

            try:
                quarter_str = idx.strftime("%b %Y") if hasattr(idx, "strftime") else str(idx)[:7]
            except Exception:
                continue

            rows.append({
                "Quarter": quarter_str,
                "Estimate": est_v,
                "Actual": act_v,
                "Beat": beat,
                "Surprise %": surp_pct,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Quarter", "Estimate", "Actual", "Beat", "Surprise %"])

    # quarterly_earnings returns a DataFrame with 'revenue' and 'earnings' columns
    if source == "quarterly_earnings":
        earnings = earnings.sort_index(ascending=False).head(8)
        rows = []
        for idx, row in earnings.iterrows():
            try:
                actual = row.get("earnings")
            except Exception:
                continue
            if pd.isna(actual):
                continue
            act_v = float(actual) if pd.notna(actual) else None

            try:
                quarter_str = idx.strftime("%b %Y") if hasattr(idx, "strftime") else str(idx)[:7]
            except Exception:
                continue

            rows.append({
                "Quarter": quarter_str,
                "Estimate": None,
                "Actual": act_v,
                "Beat": None,
                "Surprise %": None,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Quarter", "Estimate", "Actual", "Beat", "Surprise %"])

    # earnings_dates (original path)
    earnings = earnings.sort_index(ascending=False).head(8)

    rows = []
    for idx, row in earnings.iterrows():
        try:
            est = row.get("EPS Estimate")
            actual = row.get("EPS Actual")
        except Exception:
            continue
        if pd.isna(est) and pd.isna(actual):
            continue
        est_v = float(est) if pd.notna(est) else None
        act_v = float(actual) if pd.notna(actual) else None

        beat = None
        surp_pct = None
        if est_v is not None and act_v is not None and est_v != 0:
            beat = act_v >= est_v
            surp_pct = round((act_v - est_v) / abs(est_v) * 100, 2)

        try:
            quarter_str = idx.strftime("%b %Y") if hasattr(idx, "strftime") else str(idx)[:7]
        except Exception:
            continue

        rows.append({
            "Quarter": quarter_str,
            "Estimate": est_v,
            "Actual": act_v,
            "Beat": beat,
            "Surprise %": surp_pct,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Quarter", "Estimate", "Actual", "Beat", "Surprise %"])


def compute_avg_post_earnings_move(price_history: pd.DataFrame, earnings_dates: list, lookback_quarters: int = 8) -> Optional[dict]:
    """
    For the last N earnings dates, compute the ±% move 1 day after.
    Returns average absolute move and individual moves.
    """
    try:
        if price_history is None or price_history.empty or not earnings_dates:
            return None
    except Exception:
        return None

    closes = price_history["Close"]
    moves = []

    for ed in earnings_dates:
        try:
            dt = pd.to_datetime(ed, unit="s")
        except Exception:
            try:
                dt = pd.to_datetime(ed)
            except Exception:
                continue

        if dt.tzinfo is None:
            dt = pd.Timestamp(dt).tz_localize("America/New_York")
        else:
            dt = pd.Timestamp(dt).tz_convert("America/New_York")

        # Find the next trading day close after earnings
        mask = closes.index > dt
        post_dates = closes[mask]
        if len(post_dates) < 1:
            continue

        post_close = post_dates.iloc[0]

        # Find close on or just before earnings date
        mask2 = closes.index <= dt
        pre_dates = closes[mask2]
        if len(pre_dates) < 1:
            continue
        pre_close = pre_dates.iloc[-1]

        if pre_close > 0:
            move_pct = round((post_close - pre_close) / pre_close * 100, 2)
            moves.append(move_pct)

        if len(moves) >= lookback_quarters:
            break

    if not moves:
        return None

    avg_abs = round(sum(abs(m) for m in moves) / len(moves), 2)
    return {
        "avg_abs_move_pct": avg_abs,
        "individual_moves": moves,
        "positive_count": sum(1 for m in moves if m > 0),
        "negative_count": sum(1 for m in moves if m < 0),
        "total_count": len(moves),
    }