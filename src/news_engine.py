# src/news_engine.py
from __future__ import annotations

from urllib.parse import quote


def google_news_link(query: str) -> str:
    q = quote(str(query).strip())
    return f"https://news.google.com/search?q={q}&hl=da&gl=DK&ceid=DK%3Ada"


def build_asset_news_links(ticker: str, name: str = "", themes: str = "") -> dict:
    t = str(ticker).strip().upper()
    n = str(name).strip()
    th = str(themes).strip()

    return {
        "ticker_news": google_news_link(t),
        "company_news": google_news_link(f"{t} {n}".strip()),
        "theme_news": google_news_link(th) if th else "",
    }