"""
ai_summary.py — Objective AI-generated stock summary via OpenRouter + DeepSeek V3.2
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()


def build_prompt(ticker: str, info: dict, fund: dict, fc: dict,
                 perf: dict, signals: dict) -> str:
    quarters = fund.get("quarters", [])
    q0       = quarters[0] if quarters else {}
    val      = fund.get("valuation", {})
    fc6      = fc.get("6m",  {})
    fc12     = fc.get("12m", {})
    conf     = fc.get("confidence", "Low")
    cur_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    beta      = info.get("beta") or "N/A"
    sector    = info.get("sector") or "N/A"
    industry  = info.get("industry") or "N/A"

    return f"""You are a professional equity analyst. Provide a concise, objective analysis of {ticker}.
Write in plain English. No hype, no promotional language, no buy/sell recommendations.
Be precise and data-driven. Flag negatives as clearly as positives.
Use short punchy sentences. Maximum 220 words total across all sections.

## INPUT DATA

Company: {info.get('longName', ticker)} | Sector: {sector} | Industry: {industry}
Current Price: {cur_price} | Beta: {beta}

Latest Quarter Fundamentals:
- Revenue YoY: {q0.get('Rev YoY%', 'N/A')}% | QoQ: {q0.get('Rev QoQ%', 'N/A')}%
- Gross Margin: {q0.get('Gross Margin%', 'N/A')}% | Op Margin: {q0.get('Op Margin%', 'N/A')}% | Net Margin: {q0.get('Net Margin%', 'N/A')}%
- EPS Diluted: {q0.get('EPS Diluted', 'N/A')} | FCF: {q0.get('FCF', 'N/A')}
- D/E Ratio: {q0.get('D/E Ratio', 'N/A')} | Current Ratio: {q0.get('Current Ratio', 'N/A')}

Valuation:
- P/E Trailing: {val.get('P/E (Trailing)', 'N/A')} | Forward: {val.get('P/E (Forward)', 'N/A')}
- P/S: {val.get('P/S Ratio', 'N/A')} | P/B: {val.get('P/B Ratio', 'N/A')}
- ROE: {val.get('ROE %', 'N/A')}% | ROA: {val.get('ROA %', 'N/A')}%

Red Flags: {'; '.join(fund.get('flags', [])) or 'None detected'}

Technical Signals: {'; '.join(f'{k}: {v}' for k, v in signals.items())}

5-Year Performance: {perf.get('Total Return (5Y)', 'N/A')} | Max DD: {perf.get('Max Drawdown', 'N/A')} | Ann. Vol: {perf.get('Annualized Volatility', 'N/A')}

Model Forecast:
- 6M:  Bear {fc6.get('bear','N/A')}  / Base {fc6.get('base','N/A')}  / Bull {fc6.get('bull','N/A')}
- 12M: Bear {fc12.get('bear','N/A')} / Base {fc12.get('base','N/A')} / Bull {fc12.get('bull','N/A')}
- Model Confidence: {conf}

## OUTPUT FORMAT — use exactly these 4 headers, no others:

### Business & Growth
[2-3 sentences on revenue trend, margin direction, earnings quality]

### Financial Health
[1-2 sentences on balance sheet, debt, cash flow]

### Valuation
[1-2 sentences on whether current multiples are stretched, fair, or cheap vs growth]

### Key Risks
- [specific factual risk 1]
- [specific factual risk 2]
- [specific factual risk 3]
"""


def get_ai_summary(ticker: str, info: dict, fund: dict, fc: dict,
                   perf: dict, signals: dict) -> dict:
    """
    Returns dict with keys: 'text', 'model', 'error'
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "text":  None,
            "model": None,
            "error": "OPENAI_API_KEY not set in .env file.",
        }

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        prompt = build_prompt(ticker, info, fund, fc, perf, signals)

        response = client.chat.completions.create(
            model="deepseek/deepseek-v3.2",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise, objective equity analyst. "
                        "No hype. No buy/sell recommendations. "
                        "Facts and data only. Be concise."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=450,
            temperature=0.2,
            extra_headers={
                "HTTP-Referer": "https://stockalpha.stockwheel.cc",
                "X-Title":      "StockAlpha",
            },
        )

        text = response.choices[0].message.content.strip()
        return {
            "text":  text,
            "model": "DeepSeek V3.2 via OpenRouter",
            "error": None,
        }

    except Exception as e:
        return {
            "text":  None,
            "model": None,
            "error": str(e),
        }


def parse_ai_sections(text: str) -> dict:
    """Parse the 4-section markdown output into a dict."""
    headers = ["Business & Growth", "Financial Health", "Valuation", "Key Risks"]
    result  = {h: "" for h in headers}
    parts   = re.split(r"###\s+", text)
    for part in parts:
        part = part.strip()
        for h in headers:
            if part.startswith(h):
                result[h] = part[len(h):].strip()
                break
    return result
