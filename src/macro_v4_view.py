from __future__ import annotations

import streamlit as st


def render_macro_4() -> None:
    st.subheader("Macro engine")

    regime = st.selectbox(
        "Macro regime",
        ["RISK_ON", "NEUTRAL", "RISK_OFF", "HIGH_INFLATION", "RECESSION"],
        index=1,
        key="macro_regime_select",
    )

    modifier_map = {
        "RISK_ON": 0.90,
        "NEUTRAL": 1.00,
        "RISK_OFF": 1.15,
        "HIGH_INFLATION": 1.10,
        "RECESSION": 1.20,
    }

    modifier = modifier_map.get(regime, 1.00)

    st.session_state["macro_regime"] = regime
    st.session_state["macro_risk_modifier"] = modifier

    c1, c2 = st.columns(2)
    c1.metric("Macro regime", regime)
    c2.metric("Risk modifier", f"{modifier:.2f}")

    st.caption("Macro regime bruges nu direkte i Portfolio Risk Engine.")