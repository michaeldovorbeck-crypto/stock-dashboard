from __future__ import annotations

import pandas as pd
import streamlit as st

from src.help_texts import HELP_TEXT
from src.help_ui import page_intro
from src.macro_engine import macro_snapshot
from src.ui_style import apply_pro_style, render_badges, render_info_card


def _safe(v, fallback="—"):
    if v is None:
        return fallback
    try:
        if pd.isna(v):
            return fallback
    except Exception:
        pass
    txt = str(v).strip()
    return txt if txt else fallback


def _macro_badges(macro: dict) -> list[tuple[str, str]]:
    regime = str(macro.get("regime", "") or "").strip()
    inflation = pd.to_numeric(macro.get("inflation_yoy_pct"), errors="coerce")
    industrial = pd.to_numeric(macro.get("industrial_production_yoy_pct"), errors="coerce")
    unemployment = pd.to_numeric(macro.get("unemployment"), errors="coerce")

    badges: list[tuple[str, str]] = []

    if regime.lower() == "risk-on":
        badges.append((f"Regime: {regime}", "buy"))
    elif regime.lower() == "risk-off":
        badges.append((f"Regime: {regime}", "sell"))
    else:
        badges.append((f"Regime: {_safe(regime)}", "neutral"))

    if pd.notna(inflation):
        if inflation > 4:
            badges.append((f"Inflation høj: {inflation:.1f}%", "sell"))
        else:
            badges.append((f"Inflation: {inflation:.1f}%", "neutral"))

    if pd.notna(industrial):
        if industrial > 0:
            badges.append((f"Industriproduktion +", "buy"))
        else:
            badges.append((f"Industriproduktion -", "sell"))

    if pd.notna(unemployment):
        if unemployment < 5:
            badges.append((f"Ledighed lav", "buy"))
        else:
            badges.append((f"Ledighed høj", "hold"))

    return badges


def _render_macro_summary(macro: dict) -> None:
    st.markdown("### Makro-overblik")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime", _safe(macro.get("regime")), help=HELP_TEXT["macro_regime"])
    c2.metric("Inflation YoY %", _safe(macro.get("inflation_yoy_pct")))
    c3.metric("Industrial YoY %", _safe(macro.get("industrial_production_yoy_pct")))
    c4.metric("Arbejdsløshed", _safe(macro.get("unemployment")))

    c5, c6, c7 = st.columns(3)
    c5.metric("US 10Y", _safe(macro.get("us_10y")))
    c6.metric("US 2Y", _safe(macro.get("us_2y")))
    c7.metric("Yield spread", _safe(macro.get("us10y_minus_2y")))


def _render_macro_cards(macro: dict) -> None:
    st.markdown("### Nøglekort")

    c1, c2, c3 = st.columns(3)

    with c1:
        render_info_card(
            "Makroregime",
            _safe(macro.get("regime")),
            "Overordnet markedsmiljø: risk-on eller risk-off",
        )

    with c2:
        render_info_card(
            "Inflation",
            f"{_safe(macro.get('inflation_yoy_pct'))}%",
            "Årlig inflationstakt",
        )

    with c3:
        render_info_card(
            "Industriproduktion",
            f"{_safe(macro.get('industrial_production_yoy_pct'))}%",
            "Signal om økonomisk aktivitet",
        )


def _render_macro_interpretation(macro: dict) -> None:
    regime = str(macro.get("regime", "") or "").strip()
    inflation = pd.to_numeric(macro.get("inflation_yoy_pct"), errors="coerce")
    industrial = pd.to_numeric(macro.get("industrial_production_yoy_pct"), errors="coerce")
    spread = pd.to_numeric(macro.get("us10y_minus_2y"), errors="coerce")

    bullets = []

    if regime.lower() == "risk-on":
        bullets.append("Markedet er i et mere risikovilligt miljø")
    elif regime.lower() == "risk-off":
        bullets.append("Markedet er i et mere defensivt miljø")
    else:
        bullets.append("Makroregimet er uklart eller neutralt")

    if pd.notna(inflation):
        if inflation > 4:
            bullets.append("Inflation er fortsat høj og kan presse renter og værdiansættelser")
        else:
            bullets.append("Inflation virker mere kontrolleret")

    if pd.notna(industrial):
        if industrial > 0:
            bullets.append("Industriproduktion peger på fortsat aktivitet i økonomien")
        else:
            bullets.append("Industriproduktion viser svaghed i økonomien")

    if pd.notna(spread):
        if spread < 0:
            bullets.append("Negativ rentekurve kan signalere øget recessionrisiko")
        else:
            bullets.append("Rentekurven er ikke inverteret")

    st.markdown("### Hvad betyder det?")
    st.info(" / ".join(bullets[:4]))


def _render_macro_detail_table(macro: dict) -> None:
    rows = []
    for label, key in [
        ("Regime", "regime"),
        ("Inflation YoY %", "inflation_yoy_pct"),
        ("Industrial Production YoY %", "industrial_production_yoy_pct"),
        ("Arbejdsløshed", "unemployment"),
        ("US 10Y", "us_10y"),
        ("US 2Y", "us_2y"),
        ("US10Y - US2Y", "us10y_minus_2y"),
    ]:
        rows.append({"Metric": label, "Value": macro.get(key)})

    st.markdown("### Makro-detaljer")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_macro_4() -> None:
    apply_pro_style()

    st.markdown("## Macro 4.0 PRO")
    page_intro("macro")

    macro = macro_snapshot()

    _render_macro_summary(macro)
    render_badges(_macro_badges(macro))
    _render_macro_cards(macro)

    left, right = st.columns([2, 1])

    with left:
        _render_macro_interpretation(macro)

    with right:
        render_info_card(
            "Makrofokus",
            _safe(macro.get("regime")),
            "Brug makroregimet som kontekst for risiko, sektorrotation og signalstyrke",
        )

    _render_macro_detail_table(macro)