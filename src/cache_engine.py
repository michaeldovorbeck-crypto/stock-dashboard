# src/cache_engine.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

CACHE_DIR = Path("data") / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_FILE = CACHE_DIR / "global_quant_snapshot.csv"


def save_snapshot(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.to_csv(SNAPSHOT_FILE, index=False)


def load_snapshot() -> pd.DataFrame:
    try:
        if not SNAPSHOT_FILE.exists():
            return pd.DataFrame()
        return pd.read_csv(SNAPSHOT_FILE)
    except Exception:
        return pd.DataFrame()