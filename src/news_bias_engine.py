from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd
import requests
import streamlit as st


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

NEWS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://news.google.com/",
    "Connection": "keep-alive",
}

POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "gain", "gains", "growth", "strong",
    "stronger", "record", "upgrade", "upgrades", "bullish", "rally", "rallies",
    "expand", "expands", "expansion", "profit", "profits", "outperform",
    "outperforms", "demand", "boost", "boosts", "breakout", "momentum",
    "partnership", "approval", "wins", "win", "rebound", "recovery", "tailwind",
    "raises", "raised", "raise", "positive", "optimism", "optimistic",
}

NEGATIVE_WORDS = {
    "miss", "misses", "drop", "drops", "fall", "falls", "weak", "weaker",
    "downgrade", "downgrades", "bearish", "selloff", "sell-off", "slump",
    "warns", "warning", "lawsuit", "probe", "investigation", "recall", "cuts",
    "cut", "decline", "declines", "risk", "risks", "pressure", "pressured",
    "headwind", "bankruptcy", "delay", "delays", "loss", "losses", "crash",
    "crashes", "fraud", "concern", "concerns", "tariff", "tariffs",
}

STRONG_POSITIVE_PHRASES = {
    "beats earnings",
    "raises guidance",
    "record revenue",
    "strong demand",
    "price target raised",
    "upgraded to buy",
}

STRONG_NEGATIVE_PHRASES = {
    "misses earnings",
    "cuts guidance",
    "price target cut",
    "downgraded to sell",
    "sec probe",
    "criminal investigation",
}


def _clean_text(text: str) -> str:
    txt = html.unescape(text or "")
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-]+", (text or "").lower())


def _headline_sentiment_score(title: str) -> int:
    txt = (title or "").lower()
    tokens = set(_tokenize(txt))

    score = 0

    for word in POSITIVE_WORDS:
        if word in tokens:
            score += 1

    for word in NEGATIVE_WORDS:
        if word in tokens:
            score -= 1

    for phrase in STRONG_POSITIVE_PHRASES:
        if phrase in txt:
            score += 2

    for phrase in STRONG_NEGATIVE_PHRASES:
        if phrase in txt:
            score -= 2

    return score


def _build_query(ticker: str, company_name: str = "", themes: str = "") -> str:
    ticker = str(ticker or "").strip().upper()
    company_name = str(company_name or "").strip()
    theme_parts = [x.strip() for x in str(themes or "").split(",") if x.strip()]

    parts = []
    if ticker:
        parts.append(f'"{ticker}"')
    if company_name:
        parts.append(f'"{company_name}"')

    if theme_parts:
        parts.extend([f'"{x}"' for x in theme_parts[:2]])

    if not parts:
        return "stock market"

    return " OR ".join(parts)


def _fetch_google_news_rss(query: str, limit: int = 12) -> list[dict[str, Any]]:
    params = {
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }

    r = requests.get(
        GOOGLE_NEWS_RSS,
        params=params,
        headers=NEWS_HEADERS,
        timeout=20,
    )

    if r.status_code != 200:
        return []

    try:
        root = ET.fromstring(r.text)
    except Exception:
        return []

    items = []
    for item in root.findall(".//item")[:limit]:
        title = _clean_text(item.findtext("title", default=""))
        link = _clean_text(item.findtext("link", default=""))
        pub_date = _clean_text(item.findtext("pubDate", default=""))
        source = ""
        source_el = item.find("source")
        if source_el is not None and source_el.text:
            source = _clean_text(source_el.text)

        if title:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "source": source,
                }
            )

    return items


@st.cache_data(ttl=60 * 30, show_spinner=False)
def build_news_bias_snapshot(
    ticker: str,
    company_name: str = "",
    themes: str = "",
    limit: int = 12,
) -> dict[str, Any]:
    query = _build_query(ticker=ticker, company_name=company_name, themes=themes)
    headlines = _fetch_google_news_rss(query=query, limit=limit)

    if not headlines:
        return {
            "score": 0.0,
            "bucket": "Neutral",
            "headline_count": 0,
            "query": query,
            "headlines_df": pd.DataFrame(columns=["title", "source", "pub_date", "headline_score"]),
            "top_positive": [],
            "top_negative": [],
        }

    rows = []
    total = 0
    for h in headlines:
        hs = _headline_sentiment_score(h["title"])
        total += hs
        rows.append(
            {
                "title": h["title"],
                "source": h["source"],
                "pub_date": h["pub_date"],
                "headline_score": hs,
                "link": h["link"],
            }
        )

    df = pd.DataFrame(rows)

    avg = total / max(1, len(rows))
    bias_score = max(-12.0, min(12.0, round(avg * 3.0, 1)))

    if bias_score >= 4:
        bucket = "Positiv"
    elif bias_score <= -4:
        bucket = "Negativ"
    else:
        bucket = "Neutral"

    top_positive = []
    if not df.empty:
        top_positive = (
            df.sort_values("headline_score", ascending=False)
            .head(3)["title"]
            .astype(str)
            .tolist()
        )

    top_negative = []
    if not df.empty:
        top_negative = (
            df.sort_values("headline_score", ascending=True)
            .head(3)["title"]
            .astype(str)
            .tolist()
        )

    return {
        "score": bias_score,
        "bucket": bucket,
        "headline_count": len(rows),
        "query": query,
        "headlines_df": df,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }