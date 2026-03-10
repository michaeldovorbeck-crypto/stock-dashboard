from __future__ import annotations

import pandas as pd

from src.analysis_engine import build_asset_analysis


TRANSACTION_COLUMNS = [
    "Date",
    "Ticker",
    "Account",
    "Side",
    "Shares",
    "Price",
    "Fee",
    "Note",
]


def empty_transactions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSACTION_COLUMNS)


def normalize_transactions_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_transactions_df()

    work = df.copy()

    for col in TRANSACTION_COLUMNS:
        if col not in work.columns:
            work[col] = ""

    work = work[TRANSACTION_COLUMNS].copy()

    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work["Ticker"] = work["Ticker"].astype(str).str.strip().str.upper()
    work["Account"] = work["Account"].astype(str).str.strip()
    work["Side"] = work["Side"].astype(str).str.strip().str.upper()
    work["Shares"] = pd.to_numeric(work["Shares"], errors="coerce").fillna(0.0)
    work["Price"] = pd.to_numeric(work["Price"], errors="coerce").fillna(0.0)
    work["Fee"] = pd.to_numeric(work["Fee"], errors="coerce").fillna(0.0)
    work["Note"] = work["Note"].astype(str).fillna("").str.strip()

    work = work[work["Ticker"] != ""].copy()
    work = work[work["Shares"] > 0].copy()
    work = work[work["Side"].isin(["BUY", "SELL"])].copy()

    work = work.sort_values(["Date", "Ticker", "Account"]).reset_index(drop=True)
    return work


def add_transaction(
    transactions_df: pd.DataFrame,
    date_value,
    ticker: str,
    account: str,
    side: str,
    shares: float,
    price: float,
    fee: float = 0.0,
    note: str = "",
) -> pd.DataFrame:
    work = normalize_transactions_df(transactions_df)

    new_row = pd.DataFrame(
        [
            {
                "Date": pd.to_datetime(date_value, errors="coerce"),
                "Ticker": str(ticker or "").strip().upper(),
                "Account": str(account or "").strip(),
                "Side": str(side or "").strip().upper(),
                "Shares": float(shares or 0.0),
                "Price": float(price or 0.0),
                "Fee": float(fee or 0.0),
                "Note": str(note or "").strip(),
            }
        ]
    )

    out = pd.concat([work, new_row], ignore_index=True)
    return normalize_transactions_df(out)


def remove_transaction_by_index(transactions_df: pd.DataFrame, idx: int) -> pd.DataFrame:
    work = normalize_transactions_df(transactions_df)
    if work.empty:
        return work
    if idx < 0 or idx >= len(work):
        return work
    work = work.drop(index=idx).reset_index(drop=True)
    return work


def transaction_display_df(transactions_df: pd.DataFrame) -> pd.DataFrame:
    work = normalize_transactions_df(transactions_df)
    if work.empty:
        return empty_transactions_df()

    out = work.copy()
    out["Date"] = out["Date"].dt.date.astype(str)
    return out


def _signed_shares(side: str, shares: float) -> float:
    return float(shares) if str(side).upper() == "BUY" else -float(shares)


def build_positions_from_transactions(transactions_df: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    tx = normalize_transactions_df(transactions_df)
    if tx.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Account",
                "Net Shares",
                "Avg Cost",
                "Last Price",
                "Market Value",
                "Cost Basis",
                "Unrealized P/L",
                "Unrealized P/L %",
                "Buys",
                "Sells",
            ]
        )

    rows = []

    grouped = tx.groupby(["Ticker", "Account"], dropna=False)

    for (ticker, account), grp in grouped:
        grp = grp.sort_values("Date").reset_index(drop=True)

        buy_qty = 0.0
        buy_cost = 0.0
        sell_qty = 0.0

        for _, row in grp.iterrows():
            side = str(row["Side"]).upper()
            shares = float(row["Shares"])
            price = float(row["Price"])
            fee = float(row["Fee"])

            if side == "BUY":
                buy_qty += shares
                buy_cost += shares * price + fee
            elif side == "SELL":
                sell_qty += shares

        net_shares = buy_qty - sell_qty
        if net_shares <= 0:
            continue

        avg_cost = buy_cost / buy_qty if buy_qty > 0 else 0.0
        cost_basis = net_shares * avg_cost

        analysis = build_asset_analysis(ticker, years=years)
        last_price = pd.to_numeric(analysis.get("last"), errors="coerce") if analysis else None
        last_price = None if pd.isna(last_price) else float(last_price)

        market_value = None
        unrealized_pl = None
        unrealized_pl_pct = None

        if last_price is not None:
            market_value = net_shares * last_price
            unrealized_pl = market_value - cost_basis
            unrealized_pl_pct = (unrealized_pl / cost_basis * 100.0) if cost_basis > 0 else None

        rows.append(
            {
                "Ticker": ticker,
                "Account": account,
                "Net Shares": round(net_shares, 4),
                "Avg Cost": round(avg_cost, 4),
                "Last Price": None if last_price is None else round(last_price, 4),
                "Market Value": None if market_value is None else round(market_value, 2),
                "Cost Basis": round(cost_basis, 2),
                "Unrealized P/L": None if unrealized_pl is None else round(unrealized_pl, 2),
                "Unrealized P/L %": None if unrealized_pl_pct is None else round(unrealized_pl_pct, 2),
                "Buys": round(buy_qty, 4),
                "Sells": round(sell_qty, 4),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["Market Value", "Ticker"], ascending=[False, True], na_position="last")
    return out.reset_index(drop=True)


def build_account_overview_from_positions(positions_df: pd.DataFrame) -> pd.DataFrame:
    if positions_df is None or positions_df.empty:
        return pd.DataFrame(columns=["Account", "Positions", "Net Shares", "Cost Basis", "Market Value", "Unrealized P/L"])

    work = positions_df.copy()

    grouped = (
        work.groupby("Account", dropna=False)
        .agg(
            Positions=("Ticker", "count"),
            **{
                "Net Shares": ("Net Shares", "sum"),
                "Cost Basis": ("Cost Basis", "sum"),
                "Market Value": ("Market Value", "sum"),
                "Unrealized P/L": ("Unrealized P/L", "sum"),
            },
        )
        .reset_index()
    )

    for col in ["Net Shares", "Cost Basis", "Market Value", "Unrealized P/L"]:
        if col in grouped.columns:
            grouped[col] = pd.to_numeric(grouped[col], errors="coerce").round(2)

    return grouped.sort_values("Market Value", ascending=False, na_position="last").reset_index(drop=True)