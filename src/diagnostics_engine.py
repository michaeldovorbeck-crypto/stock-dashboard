from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.data_sources import fetch_history_with_meta


def get_ticker_diagnostics(ticker: str, years: int = 5) -> dict[str, Any]:
    meta = fetch_history_with_meta(ticker, years=years)

    df = meta.get("df", pd.DataFrame())
    source = meta.get("source", "")
    used_symbol = meta.get("used_symbol", "")
    attempts = meta.get("attempts", [])
    alternatives = meta.get("alternatives", [])

    ok = not df.empty
    message = f"Data fundet via {source} ({used_symbol})" if ok else "Ingen data fundet via Twelve Data, Yahoo eller Stooq"

    return {
        "ok": ok,
        "df": df,
        "source": source,
        "used_symbol": used_symbol,
        "attempts": attempts,
        "alternatives": alternatives,
        "message": message,
    }


def attempts_to_df(attempts: list[dict[str, str]]) -> pd.DataFrame:
    if not attempts:
        return pd.DataFrame(columns=["source", "symbol", "status"])

    df = pd.DataFrame(attempts)
    for col in ["source", "symbol", "status"]:
        if col not in df.columns:
            df[col] = ""

    return df[["source", "symbol", "status"]].copy()


def best_alternative_suggestions(
    original_ticker: str,
    alternatives: list[str],
    used_symbol: str = "",
    max_items: int = 3,
) -> list[str]:
    original = (original_ticker or "").strip().upper()
    used = (used_symbol or "").strip().upper()

    out: list[str] = []
    seen = set()

    for alt in alternatives:
        a = (alt or "").strip().upper()
        if not a or a == original or (used and a == used) or a in seen:
            continue
        out.append(a)
        seen.add(a)

    return out[:max_items]


def render_data_status_banner(diag: dict[str, Any]) -> None:
    if diag.get("ok"):
        source = diag.get("source", "")
        used_symbol = diag.get("used_symbol", "")
        rows = len(diag.get("df", pd.DataFrame()))
        st.success(f"Datakilde: {source} | Brugt ticker: {used_symbol} | Rækker: {rows}")
    else:
        st.warning(diag.get("message", "Ingen data fundet"))


def render_alternative_ticker_buttons(
    original_ticker: str,
    alternatives: list[str],
    used_symbol: str = "",
    state_key: str = "ticker",
    max_items: int = 3,
) -> None:
    suggestions = best_alternative_suggestions(
        original_ticker=original_ticker,
        alternatives=alternatives,
        used_symbol=used_symbol,
        max_items=max_items,
    )

    if not suggestions:
        return

    st.info("Lokal ticker gav ingen eller svage data. Prøv et alternativ:")
    for alt in suggestions:
        if st.button(alt, key=f"alt_ticker_{state_key}_{alt}"):
            st.session_state[state_key] = alt
            st.rerun()


def render_diagnostics_expander(
    diag: dict[str, Any],
    title: str = "Data diagnostics",
    show_success_log: bool = True,
) -> None:
    attempts = diag.get("attempts", [])
    df_attempts = attempts_to_df(attempts)

    with st.expander(title, expanded=False):
        st.write(diag.get("message", ""))

        if diag.get("source"):
            st.caption(f"Valgt kilde: {diag.get('source')} | Brugt ticker: {diag.get('used_symbol')}")

        if df_attempts.empty:
            st.write("Ingen diagnostics tilgængelige.")
            return

        if not show_success_log:
            mask = ~df_attempts["status"].astype(str).str.contains("Success", case=False, na=False)
            df_attempts = df_attempts[mask].reset_index(drop=True)

        st.dataframe(df_attempts, use_container_width=True, hide_index=True)


def render_diagnostics_tab(diag: dict[str, Any]) -> None:
    st.subheader("Diagnostics")
    st.write(diag.get("message", ""))

    st.write(f"**Datakilde:** {diag.get('source', '') or '-'}")
    st.write(f"**Brugt ticker:** {diag.get('used_symbol', '') or '-'}")
    st.write(f"**Forsøg:** {len(diag.get('attempts', []))}")

    attempts = attempts_to_df(diag.get("attempts", []))
    if attempts.empty:
        st.info("Ingen diagnostics-data.")
    else:
        st.dataframe(attempts, use_container_width=True, hide_index=True)

    suggestions = best_alternative_suggestions(
        original_ticker="",
        alternatives=diag.get("alternatives", []),
        used_symbol=diag.get("used_symbol", ""),
        max_items=5,
    )

    if suggestions:
        st.markdown("**Alternative tickers**")
        st.write(", ".join(suggestions))