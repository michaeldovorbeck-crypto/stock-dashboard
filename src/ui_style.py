from __future__ import annotations

import streamlit as st


PRO_CSS = """
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.5rem;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 10px 12px;
}

.sd-card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 12px;
    background: rgba(255,255,255,0.025);
}

.sd-card-title {
    font-size: 0.85rem;
    opacity: 0.8;
    margin-bottom: 6px;
}

.sd-card-value {
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.2;
}

.sd-card-sub {
    font-size: 0.82rem;
    opacity: 0.75;
    margin-top: 6px;
}

.sd-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
    border: 1px solid rgba(255,255,255,0.08);
}

.sd-badge-buy {
    background: rgba(0, 180, 90, 0.18);
}

.sd-badge-hold {
    background: rgba(220, 180, 0, 0.18);
}

.sd-badge-sell {
    background: rgba(220, 60, 60, 0.18);
}

.sd-badge-neutral {
    background: rgba(120, 120, 120, 0.18);
}

.sd-section-label {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 8px;
}

.sd-divider {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 10px 0 14px 0;
}
</style>
"""


def apply_pro_style() -> None:
    st.markdown(PRO_CSS, unsafe_allow_html=True)


def render_info_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="sd-card">
            <div class="sd-card-title">{title}</div>
            <div class="sd-card-value">{value}</div>
            <div class="sd-card-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badges(items: list[tuple[str, str]]) -> None:
    """
    items = [(text, kind)]
    kind in: buy, hold, sell, neutral
    """
    chunks = []
    for text, kind in items:
        css = {
            "buy": "sd-badge sd-badge-buy",
            "hold": "sd-badge sd-badge-hold",
            "sell": "sd-badge sd-badge-sell",
            "neutral": "sd-badge sd-badge-neutral",
        }.get(kind, "sd-badge sd-badge-neutral")
        chunks.append(f'<span class="{css}">{text}</span>')

    st.markdown("".join(chunks), unsafe_allow_html=True)