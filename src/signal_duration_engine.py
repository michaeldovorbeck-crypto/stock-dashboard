from __future__ import annotations

import pandas as pd

from src.unified_signal_engine import build_technical_signal_history


def _current_streak_info(signal_df: pd.DataFrame) -> dict:
    if signal_df.empty or "Technical Signal" not in signal_df.columns or "Date" not in signal_df.columns:
        return {
            "current_signal": "Ukendt",
            "streak_trading_days": 0,
            "streak_calendar_days": 0,
            "last_switch_date": None,
            "previous_signal": "Ukendt",
        }

    work = signal_df.sort_values("Date").reset_index(drop=True)
    current_signal = str(work.iloc[-1]["Technical Signal"])
    latest_date = pd.to_datetime(work.iloc[-1]["Date"])

    streak_count = 0
    streak_start_idx = len(work) - 1

    for idx in range(len(work) - 1, -1, -1):
        if str(work.iloc[idx]["Technical Signal"]) == current_signal:
            streak_count += 1
            streak_start_idx = idx
        else:
            break

    streak_start_date = pd.to_datetime(work.iloc[streak_start_idx]["Date"])
    streak_calendar_days = int((latest_date - streak_start_date).days) + 1

    previous_signal = "Ukendt"
    if streak_start_idx > 0:
        previous_signal = str(work.iloc[streak_start_idx - 1]["Technical Signal"])

    return {
        "current_signal": current_signal,
        "streak_trading_days": streak_count,
        "streak_calendar_days": streak_calendar_days,
        "last_switch_date": streak_start_date,
        "previous_signal": previous_signal,
    }


def _distribution_for_window(signal_df: pd.DataFrame, days: int) -> dict:
    if signal_df.empty or "Date" not in signal_df.columns or "Technical Signal" not in signal_df.columns:
        return {
            "Periode": f"{days} dage",
            "KØB": 0,
            "HOLD": 0,
            "SÆLG": 0,
            "Dominerende": "Ukendt",
        }

    work = signal_df.sort_values("Date").reset_index(drop=True)
    latest_date = pd.to_datetime(work["Date"]).max()
    cutoff = latest_date - pd.Timedelta(days=int(days))
    subset = work[work["Date"] >= cutoff].copy()

    counts = subset["Technical Signal"].value_counts().to_dict()
    buy_n = int(counts.get("KØB", 0))
    hold_n = int(counts.get("HOLD", 0))
    sell_n = int(counts.get("SÆLG", 0))

    dominant = "Ukendt"
    if (buy_n + hold_n + sell_n) > 0:
        dominant = max(
            [("KØB", buy_n), ("HOLD", hold_n), ("SÆLG", sell_n)],
            key=lambda x: x[1],
        )[0]

    return {
        "Periode": f"{days} dage",
        "KØB": buy_n,
        "HOLD": hold_n,
        "SÆLG": sell_n,
        "Dominerende": dominant,
    }


def build_signal_duration_snapshot(price_df: pd.DataFrame) -> dict:
    signal_df = build_technical_signal_history(price_df)

    streak = _current_streak_info(signal_df)

    distribution_rows = [
        _distribution_for_window(signal_df, 30),
        _distribution_for_window(signal_df, 90),
        _distribution_for_window(signal_df, 180),
        _distribution_for_window(signal_df, 365),
    ]
    distribution_df = pd.DataFrame(distribution_rows)

    recent_df = pd.DataFrame()
    if not signal_df.empty:
        recent_df = signal_df.sort_values("Date", ascending=False).head(60).reset_index(drop=True)

    return {
        "signal_df": signal_df,
        "distribution_df": distribution_df,
        "recent_df": recent_df,
        **streak,
    }