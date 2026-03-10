from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.help_texts import HELP_TEXT


def safe_metric_value(value, fallback="—"):
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    txt = str(value).strip()
    return txt if txt else fallback


def _to_float(value):
    try:
        x = pd.to_numeric(value, errors="coerce")
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def signal_badge(action: str) -> str:
    act = str(action or "").upper()
    if act == "BUY":
        return "🟢 BUY"
    if act == "SELL":
        return "🔴 SELL"
    if act == "HOLD":
        return "🟡 HOLD"
    if act == "KØB":
        return "🟢 KØB"
    if act == "SÆLG":
        return "🔴 SÆLG"
    return "⚪ N/A"


def score_badge(score, good=60, warn=40) -> str:
    s = _to_float(score)
    if s is None:
        return "⚪ —"
    if s >= good:
        return f"🟢 {s:.0f}"
    if s >= warn:
        return f"🟡 {s:.0f}"
    return f"🔴 {s:.0f}"


def confidence_label(diag: dict, analysis: dict) -> str:
    rows = len(diag.get("df", pd.DataFrame()))
    source = str(diag.get("source", "") or "")
    timing = analysis.get("timing", {})
    timing_score = _to_float(timing.get("timing_score"))
    rsi = _to_float(timing.get("rsi"))

    points = 0
    if rows >= 750:
        points += 2
    elif rows >= 250:
        points += 1

    if source in {"Twelve Data", "Yahoo", "Stooq"}:
        points += 1
    if timing_score is not None:
        points += 1
    if rsi is not None:
        points += 1

    if points >= 5:
        return "Høj"
    if points >= 3:
        return "Mellem"
    return "Lav"


def render_source_strip(diag: dict) -> None:
    source = safe_metric_value(diag.get("source"))
    used_symbol = safe_metric_value(diag.get("used_symbol"))
    rows = len(diag.get("df", pd.DataFrame()))

    s1, s2, s3 = st.columns(3)
    s1.metric("Datakilde", source, help=HELP_TEXT["data_source"])
    s2.metric("Brugt ticker", used_symbol, help=HELP_TEXT["used_symbol"])
    s3.metric("Datapunkter", rows)


def render_hero_panel(analysis: dict, diag: dict) -> None:
    record = analysis.get("record", {})
    timing = analysis.get("timing", {})
    returns = analysis.get("returns", {})
    unified = analysis.get("unified_signal", {})

    display_signal = unified.get("technical_signal") or timing.get("action", "")
    display_score = unified.get("technical_score", timing.get("timing_score"))
    quant_score = analysis.get("quant_score")
    confidence = confidence_label(diag, analysis)

    st.markdown("### Hero")

    h1, h2, h3, h4 = st.columns([1.3, 2.2, 1.2, 1.2])
    h1.metric("Ticker", safe_metric_value(analysis.get("ticker")))
    h2.metric("Navn", safe_metric_value(record.get("name")))
    h3.metric("Signal", signal_badge(display_signal))
    h4.metric("Confidence", confidence)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pris", safe_metric_value(analysis.get("last")))
    m2.metric("Teknisk score", score_badge(display_score), help=HELP_TEXT["timing_score"])
    m3.metric("Quant", score_badge(quant_score), help=HELP_TEXT["quant_score"])
    m4.metric("Volatilitet", safe_metric_value(timing.get("atr_pct")), help=HELP_TEXT["atr"])

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("1D %", safe_metric_value(returns.get("1D")))
    r2.metric("1M %", safe_metric_value(returns.get("1M")))
    r3.metric("3M %", safe_metric_value(returns.get("3M")))
    r4.metric("6M %", safe_metric_value(returns.get("6M")))

    st.caption(
        f"Type: {safe_metric_value(record.get('type'))} | "
        f"Sektor: {safe_metric_value(record.get('sector'))} | "
        f"Land: {safe_metric_value(record.get('country'))}"
    )
    st.write(f"**Themes:** {record.get('themes', '') or 'Ingen tema-match endnu'}")
    render_source_strip(diag)


def render_signal_summary_card(analysis: dict) -> None:
    unified = analysis.get("unified_signal", {})
    timing = analysis.get("timing", {})

    action = str(unified.get("technical_signal", "HOLD")).upper()
    trend = safe_metric_value(unified.get("trend_state", timing.get("trend")))
    rsi = _to_float(timing.get("rsi"))
    momentum_1m = _to_float(timing.get("momentum_1m"))
    momentum_3m = _to_float(timing.get("momentum_3m"))

    bullets = []
    bullets.append(
        "Det tekniske signal er bullish"
        if action == "KØB"
        else "Det tekniske signal er bearish"
        if action == "SÆLG"
        else "Det tekniske signal er neutralt"
    )
    bullets.append(f"Trend vurderes som {trend}")

    if rsi is not None:
        if rsi > 70:
            bullets.append("RSI indikerer overkøbt niveau")
        elif rsi < 30:
            bullets.append("RSI indikerer oversolgt niveau")
        else:
            bullets.append("RSI er i neutralt område")

    if momentum_1m is not None:
        bullets.append("1M momentum er positivt" if momentum_1m > 0 else "1M momentum er svagt/negativt")
    if momentum_3m is not None:
        bullets.append("3M momentum er positivt" if momentum_3m > 0 else "3M momentum er svagt/negativt")

    st.markdown("### Signal summary")
    st.info(" • ".join(bullets[:5]))


def render_why_this_matters(analysis: dict) -> None:
    record = analysis.get("record", {})
    unified = analysis.get("unified_signal", {})
    macro = analysis.get("macro", {})
    timing = analysis.get("timing", {})

    why = []

    if unified.get("overall_signal") == "KØB":
        why.append("den samlede indikation peger mod køb")
    elif unified.get("overall_signal") == "SÆLG":
        why.append("den samlede indikation peger mod salg")

    m1 = pd.to_numeric(timing.get("momentum_1m"), errors="coerce")
    if pd.notna(m1) and float(m1) > 0:
        why.append("positivt 1M momentum")

    m3 = pd.to_numeric(timing.get("momentum_3m"), errors="coerce")
    if pd.notna(m3) and float(m3) > 0:
        why.append("positivt 3M momentum")

    if record.get("themes", ""):
        why.append("indgår i relevante temaer")

    if macro.get("regime") == "Risk-on":
        why.append("makroregime understøtter risikable aktiver")

    if not why:
        why = ["ingen tydelig edge endnu", "kræver mere analyse", "se diagnostics og compare"]

    st.markdown("### Why this matters")
    st.info(" / ".join(why[:4]))


def render_quick_stats_block(analysis: dict) -> None:
    timing = analysis.get("timing", {})
    unified = analysis.get("unified_signal", {})

    st.metric("Teknisk signal", safe_metric_value(unified.get("technical_signal")))
    st.metric("Samlet indikation", safe_metric_value(unified.get("overall_signal")))
    st.metric("Teknisk score", safe_metric_value(unified.get("technical_score")), help=HELP_TEXT["timing_score"])
    st.metric("RSI", safe_metric_value(timing.get("rsi")), help=HELP_TEXT["rsi"])
    st.metric("ATR %", safe_metric_value(timing.get("atr_pct")), help=HELP_TEXT["atr"])


def render_factor_view(analysis: dict) -> None:
    unified = analysis.get("unified_signal", {})
    timing = analysis.get("timing", {})
    news_bias_snapshot = analysis.get("news_bias_snapshot", {})

    factors = {
        "Technical score": _to_float(unified.get("technical_score")),
        "Overall score": _to_float(unified.get("overall_score")),
        "News bias": _to_float(unified.get("news_bias")),
        "RSI": _to_float(timing.get("rsi")),
        "1M momentum": _to_float(timing.get("momentum_1m")),
        "3M momentum": _to_float(timing.get("momentum_3m")),
        "ATR %": _to_float(timing.get("atr_pct")),
    }

    rows = [{"Factor": k, "Value": v} for k, v in factors.items()]
    st.markdown("### Mini factor view")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    bucket = news_bias_snapshot.get("bucket", "Neutral")
    count = news_bias_snapshot.get("headline_count", 0)
    st.caption(f"News bias: {bucket} | Headlines analyseret: {count}")


def render_market_now_cards(market: dict) -> None:
    st.markdown("### Marked lige nu")

    c1, c2 = st.columns(2)

    with c1:
        if not market["discovery_df"].empty:
            st.markdown("**Top discovery**")
            cols = [c for c in ["Theme", "Discovery Score", "Stage"] if c in market["discovery_df"].columns]
            if cols:
                st.dataframe(market["discovery_df"][cols].head(5), use_container_width=True, hide_index=True)

        snapshot_df = market.get("snapshot_df", pd.DataFrame())
        if not snapshot_df.empty:
            st.markdown("**Top quant picks**")
            cols = [c for c in ["Ticker", "Quant Score", "Timing Score", "Action"] if c in snapshot_df.columns]
            if cols:
                work = snapshot_df.copy()
                if "Quant Score" in work.columns:
                    work["Quant Score"] = pd.to_numeric(work["Quant Score"], errors="coerce")
                if "Timing Score" in work.columns:
                    work["Timing Score"] = pd.to_numeric(work["Timing Score"], errors="coerce")
                sort_cols = [c for c in ["Quant Score", "Timing Score"] if c in work.columns]
                if sort_cols:
                    work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
                st.dataframe(work[cols].head(5), use_container_width=True, hide_index=True)

    with c2:
        if not market["etf_df"].empty:
            st.markdown("**Top ETF'er**")
            cols = [c for c in ["Ticker", "Strategy Score", "Action"] if c in market["etf_df"].columns]
            if cols:
                st.dataframe(market["etf_df"][cols].head(5), use_container_width=True, hide_index=True)

        leaders_df = market.get("leaders_df", pd.DataFrame())
        if not leaders_df.empty:
            st.markdown("**Top leaders**")
            cols = [c for c in ["Ticker", "Strategy Score", "Action"] if c in leaders_df.columns]
            if cols:
                st.dataframe(leaders_df[cols].head(5), use_container_width=True, hide_index=True)


def _gauge_score(score: float | None) -> int:
    if score is None:
        return 0
    return max(0, min(100, int(round(score))))


def _gauge_bucket(score: float | None) -> str:
    if score is None:
        return "Ukendt"
    if score >= 67:
        return "Høj"
    if score >= 34:
        return "Mellem"
    return "Lav"


def _build_gauge_figure(value: int, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "/100"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#60a5fa"},
                "steps": [
                    {"range": [0, 34], "color": "rgba(239,68,68,0.35)"},
                    {"range": [34, 67], "color": "rgba(245,158,11,0.35)"},
                    {"range": [67, 100], "color": "rgba(34,197,94,0.35)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 4},
                    "thickness": 0.8,
                    "value": value,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )

    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
    )
    return fig


def render_score_gauge_block(label: str, score: float | None, help_text: str = "") -> None:
    val = _gauge_score(score)
    bucket = _gauge_bucket(score)

    st.caption(label if not help_text else f"{label}  ⓘ")
    fig = _build_gauge_figure(val, label)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Niveau: {bucket}")

    if help_text:
        with st.expander(f"Hvad betyder {label}?", expanded=False):
            st.write(help_text)


def render_recommendation_panel(unified: dict) -> None:
    st.markdown("### Samlet anbefaling")
    action = unified.get("overall_signal", "HOLD")
    score = unified.get("overall_score", 50)
    reasons = unified.get("reasons", [])
    news_bias = unified.get("news_bias", 0)

    st.metric("Samlet indikation", action)
    st.progress(max(0, min(100, float(score))) / 100)
    st.caption(f"Samlet score: {score}/100 | News bias: {news_bias:+.1f}")

    if reasons:
        st.info(" / ".join(reasons[:6]))