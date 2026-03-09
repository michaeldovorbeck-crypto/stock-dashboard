# src/signal_log_engine.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd


LOG_FILE = Path("data") / "signal_log.csv"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _safe_read_log() -> pd.DataFrame:
    try:
        if not LOG_FILE.exists():
            return pd.DataFrame()
        return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame()


def append_signal_log(
    source: str,
    ticker: str,
    action: str,
    timing_score,
    theme: str = "",
    strategy_score=None,
    note: str = "",
) -> None:
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "source": str(source).strip(),
        "ticker": str(ticker).strip().upper(),
        "theme": str(theme).strip(),
        "action": str(action).strip(),
        "timing_score": timing_score,
        "strategy_score": strategy_score,
        "note": str(note).strip(),
    }

    old = _safe_read_log()
    new = pd.DataFrame([row])

    if old.empty:
        out = new
    else:
        out = pd.concat([old, new], ignore_index=True)

    out.to_csv(LOG_FILE, index=False)


def read_signal_log(limit: int = 200) -> pd.DataFrame:
    df = _safe_read_log()
    if df.empty:
        return pd.DataFrame()
    return df.tail(limit).iloc[::-1].reset_index(drop=True)


def signal_summary() -> dict:
    df = _safe_read_log()
    if df.empty:
        return {
            "count": 0,
            "buy_ratio_pct": None,
            "hold_ratio_pct": None,
            "sell_ratio_pct": None,
        }

    actions = df["action"].astype(str).str.upper()

    return {
        "count": int(len(df)),
        "buy_ratio_pct": round(float((actions == "BUY").mean() * 100.0), 2),
        "hold_ratio_pct": round(float((actions == "HOLD").mean() * 100.0), 2),
        "sell_ratio_pct": round(float((actions == "SELL").mean() * 100.0), 2),
    }