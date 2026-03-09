"""
Central help texts for Stock Dashboard
"""

PAGE_HELP = {
    "analysis": """
**Analyse**

Her analyserer du et enkelt aktiv.

Du kan:
• søge globalt
• se signaler og momentum
• forstå tema- og makrokontekst
• sammenligne med peers
• se diagnostics og nyheder
""",
    "screening": """
**Screening**

Finder kandidater i et universe baseret på signaler og filtre.

Du kan:
• filtrere på sektor og land
• vælge minimum timing score
• finde stærke kandidater
""",
    "quant": """
**Quant**

Bygger og analyserer et kvantitativt snapshot af markedet.

Bruges til:
• ranking
• snapshot screener
• top picks
""",
    "macro": """
**Macro**

Viser makroregime og nøgletal.

Formålet er at forstå:
• risk-on / risk-off
• inflation
• økonomisk aktivitet
""",
    "themes": """
**Tema**

Viser strukturelle investeringstemaer og hvilke aktier der indgår.
""",
    "discovery": """
**Discovery**

Finder nye trends, temaer og emerging leaders.
""",
    "strategy": """
**Strategy**

Viser de stærkeste ETF'er og leaders.
""",
    "portfolio": """
**Portefølje**

Her registrerer du positioner, ser theme exposure og rebalance-forslag.
""",
}

HELP_TEXT = {
    "timing_score": """
Timing score kombinerer:
• trend
• momentum
• RSI

> 60 = bullish
40-60 = neutral
< 40 = bearish
""",
    "quant_score": """
Quant score rangerer aktier baseret på bl.a.
momentum, trend og relativ styrke.
""",
    "macro_regime": """
Makroregime beskriver det overordnede miljø.

Risk-on = investorer søger risiko
Risk-off = investorer søger sikkerhed
""",
    "rsi": """
RSI (Relative Strength Index)

> 70 = overkøbt
< 30 = oversolgt
""",
    "atr": """
ATR måler volatilitet.
Jo højere ATR, jo større prisudsving.
""",
    "strategy_score": """
Strategy score bruges til at identificere de stærkeste ETF'er og leaders.
""",
    "discovery_score": """
Discovery score måler hvor stærkt en tidlig trend udvikler sig.
""",
    "universe": """
Universe-fil er CSV-listen over tickers, som screeningen analyserer.
""",
    "compare": """
Compare viser relativ udvikling og nøgletal for udvalgte tickers.
""",
    "data_source": """
Viser hvilken datakilde der faktisk blev brugt:
Twelve Data, Yahoo eller Stooq.
""",
    "used_symbol": """
Viser den ticker/symbolvariant, der faktisk gav data.
""",
    "watchlist": """
Watchlist er din egen liste over aktiver, du vil følge tæt.
""",
    "recent_views": """
Senest sete viser de seneste aktiver, du har analyseret.
""",
}

GLOBAL_HELP = """
### Sådan bruges dashboardet

**Analyse**
Dyb analyse af én aktie eller ETF.

**Screening**
Finder nye kandidater i et universe.

**Quant**
Kvantitativ ranking af markedet.

**Macro**
Makroøkonomiske indikatorer.

**Tema**
Tematiske investeringer.

**Discovery**
Finder nye trends.

**Strategy**
Top ETF'er og leaders.

**Portefølje**
Overblik over dine positioner.
"""