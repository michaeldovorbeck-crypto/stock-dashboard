from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

WATCHLIST_FILE = DATA_DIR / "watchlist.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio_positions.json"
TRANSACTIONS_FILE = DATA_DIR / "portfolio_transactions.json"
RECENT_ASSETS_FILE = DATA_DIR / "recent_assets.json"


def _read_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


# -----------------------------------
# Watchlist
# -----------------------------------

def load_watchlist() -> list[str]:
    data = _read_json(WATCHLIST_FILE, [])
    if not isinstance(data, list):
        return []
    return [str(x).strip().upper() for x in data if str(x).strip()]


def save_watchlist(watchlist: list[str]) -> None:
    cleaned = []
    seen = set()
    for x in watchlist:
        t = str(x).strip().upper()
        if t and t not in seen:
            cleaned.append(t)
            seen.add(t)
    _write_json(WATCHLIST_FILE, cleaned)


# -----------------------------------
# Legacy portfolio positions
# -----------------------------------

def load_portfolio_positions() -> list[dict]:
    data = _read_json(PORTFOLIO_FILE, [])
    if not isinstance(data, list):
        return []
    return data


def save_portfolio_positions(positions: list[dict]) -> None:
    _write_json(PORTFOLIO_FILE, positions)


# -----------------------------------
# Portfolio transactions
# -----------------------------------

def load_portfolio_transactions() -> list[dict]:
    data = _read_json(TRANSACTIONS_FILE, [])
    if not isinstance(data, list):
        return []
    return data


def save_portfolio_transactions(transactions: list[dict]) -> None:
    _write_json(TRANSACTIONS_FILE, transactions)


# -----------------------------------
# Recent assets / history
# -----------------------------------

def load_recent_assets() -> list[str]:
    data = _read_json(RECENT_ASSETS_FILE, [])
    if not isinstance(data, list):
        return []
    return [str(x).strip().upper() for x in data if str(x).strip()]


def save_recent_assets(assets: list[str]) -> None:
    cleaned = []
    seen = set()
    for x in assets:
        t = str(x).strip().upper()
        if t and t not in seen:
            cleaned.append(t)
            seen.add(t)
    _write_json(RECENT_ASSETS_FILE, cleaned[:50])


def add_recent_asset(ticker: str, max_items: int = 20) -> list[str]:
    t = str(ticker or "").strip().upper()
    if not t:
        return load_recent_assets()

    recent = load_recent_assets()
    recent = [x for x in recent if x != t]
    recent.insert(0, t)
    recent = recent[:max_items]
    save_recent_assets(recent)
    return recent