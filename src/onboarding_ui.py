from __future__ import annotations

import streamlit as st


ONBOARDING_STEPS = [
    {
        "title": "Velkommen til Stock Dashboard",
        "body": """
Dette dashboard hjælper dig med at analysere aktier, ETF'er, temaer og porteføljer i én samlet løsning.

Du kan bruge det til:
- at finde nye investeringsidéer
- at analysere et enkelt aktiv i dybden
- at forstå makroregimet
- at se hvilke temaer og strategier der er stærke lige nu
- at følge og evaluere din portefølje
""",
    },
    {
        "title": "Analyse-fanen",
        "body": """
Analyse er dit vigtigste cockpit.

Her kan du:
- søge efter en ticker
- se signal, timing, trend og momentum
- læse aktivets kontekst
- sammenligne med peers og benchmarks
- se diagnostics, hvis data fejler

Brug Analyse-fanen når du vil forstå *ét aktiv i dybden*.
""",
    },
    {
        "title": "Screening og Quant",
        "body": """
Screening finder kandidater i et helt univers.

Quant rangerer aktiver mere systematisk efter score og signalstyrke.

Brug disse faner når du vil:
- finde nye kandidater
- sortere store universer
- opdage hvilke aktier der ser stærkest ud lige nu
""",
    },
    {
        "title": "Macro, Themes, Discovery og Strategy",
        "body": """
Disse faner giver top-down kontekst.

Macro:
- viser risk-on / risk-off
- inflation, renter og økonomisk aktivitet

Themes:
- viser strukturelle investeringstrends

Discovery:
- finder nye og voksende trends tidligt

Strategy:
- viser stærke ETF'er og leaders
""",
    },
    {
        "title": "Portfolio og næste skridt",
        "body": """
Portfolio hjælper dig med at holde styr på dine positioner, eksponeringer og rebalancering.

God arbejdsgang:
1. Start i Macro eller Discovery
2. Brug Screening eller Strategy til idéer
3. Gå til Analyse for dyb vurdering
4. Tilføj relevante aktiver til Watchlist eller Portfolio
""",
    },
]


def _ensure_onboarding_state() -> None:
    if "onboarding_open" not in st.session_state:
        st.session_state["onboarding_open"] = True
    if "onboarding_step" not in st.session_state:
        st.session_state["onboarding_step"] = 0
    if "onboarding_hidden_forever" not in st.session_state:
        st.session_state["onboarding_hidden_forever"] = False


def render_onboarding_guide() -> None:
    _ensure_onboarding_state()

    if st.session_state["onboarding_hidden_forever"]:
        return

    if not st.session_state["onboarding_open"]:
        if st.button("Åbn introduktion", key="open_onboarding_again"):
            st.session_state["onboarding_open"] = True
            st.rerun()
        return

    step = int(st.session_state["onboarding_step"])
    step = max(0, min(step, len(ONBOARDING_STEPS) - 1))
    current = ONBOARDING_STEPS[step]

    with st.container():
        st.markdown("## Introduktion til dashboardet")
        st.caption(f"Trin {step + 1} af {len(ONBOARDING_STEPS)}")

        st.info(current["title"])
        st.markdown(current["body"])

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if st.button("Forrige", key="onboarding_prev", disabled=(step == 0)):
                st.session_state["onboarding_step"] = max(0, step - 1)
                st.rerun()

        with c2:
            if st.button("Næste", key="onboarding_next", disabled=(step >= len(ONBOARDING_STEPS) - 1)):
                st.session_state["onboarding_step"] = min(len(ONBOARDING_STEPS) - 1, step + 1)
                st.rerun()

        with c3:
            if st.button("Luk", key="onboarding_close"):
                st.session_state["onboarding_open"] = False
                st.rerun()

        with c4:
            if st.button("Skjul fremover", key="onboarding_hide_forever"):
                st.session_state["onboarding_hidden_forever"] = True
                st.session_state["onboarding_open"] = False
                st.rerun()