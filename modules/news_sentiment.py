"""
news_sentiment.py — Fetches headlines via yfinance.Ticker.news and scores sentiment
via DeepSeek V4 Flash (OpenRouter), with caching and trend analysis.
"""

import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()


def _score_headline_via_api(headline: str) -> str:
    """Score a single headline as Positive / Neutral / Negative via OpenRouter."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Neutral"

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        response = client.chat.completions.create(
            model="deepseek/deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial sentiment classifier. "
                        "Respond with exactly one word: Positive, Neutral, or Negative. "
                        "No explanation, no punctuation."
                    ),
                },
                {"role": "user", "content": headline},
            ],
            max_tokens=10,
            temperature=0.0,
        )

        label = response.choices[0].message.content.strip().lower()
        if "positive" in label:
            return "Positive"
        elif "negative" in label:
            return "Negative"
        return "Neutral"
    except Exception:
        return "Neutral"


def _score_headlines(headlines: list[dict]) -> list[dict]:
    """Score a batch of headlines via OpenRouter, one API call per headline."""
    results = []
    for item in headlines:
        title = item.get("title", "").strip()
        if not title:
            continue
        sentiment = _score_headline_via_api(title)
        results.append({
            "title": title,
            "sentiment": sentiment,
            "timestamp": item.get("providerPublishTime"),
        })
    return results


import streamlit as st


@st.cache_data(ttl=1800)
def fetch_and_score_news(ticker: str) -> dict:
    """
    Fetch recent headlines via yfinance and score sentiment.
    Returns dict with keys: 'headlines' (list of scored items), 'summary'.
    """
    import yfinance as yf

    try:
        yf_obj = yf.Ticker(ticker)
        raw_news = yf_obj.news or []
    except Exception:
        return {"headlines": [], "summary": {"total": 0, "positive": 0, "neutral": 0, "negative": 0, "recent_negative": False}}

    if not raw_news:
        return {"headlines": [], "summary": {"total": 0, "positive": 0, "neutral": 0, "negative": 0, "recent_negative": False}}

    scored = _score_headlines(raw_news[:10])

    total = len(scored)
    positive = sum(1 for h in scored if h["sentiment"] == "Positive")
    neutral  = sum(1 for h in scored if h["sentiment"] == "Neutral")
    negative = sum(1 for h in scored if h["sentiment"] == "Negative")

    # Check if majority of headlines in last 48 hours are negative
    now = datetime.now(timezone.utc)
    recent_negative = False
    recent = [h for h in scored if h.get("timestamp") and now - datetime.fromtimestamp(h["timestamp"], tz=timezone.utc) <= timedelta(hours=48)]
    if recent:
        recent_neg_count = sum(1 for h in recent if h["sentiment"] == "Negative")
        if recent_neg_count > len(recent) / 2:
            recent_negative = True

    summary = {
        "total": total,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "recent_negative": recent_negative,
    }

    return {"headlines": scored, "summary": summary}