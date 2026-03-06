# src/ui_text_da.py

help_text = """
### Hjælp: hvad ser jeg?

Denne app er en gratis “mini-Nordnet” til overblik og screening (ikke finansiel rådgivning).

**Faner / funktioner**
- **EU Europa**: Viser univers (STOXX 600 + DK/SE/DE hvis du har dem som CSV) og giver en “top-liste” baseret på tekniske signaler.
- **US USA (S&P 500)**: Viser US-univers (hvis du har S&P500-listen som CSV i data/universes).
- **Vælg aktie & graf**: Søg på **ticker** eller **navn** og se kursgraf + nøgletal.
- **Portefølje**: Upload din portefølje-CSV og få fordeling (%) og sektorer.

**Signal-forklaring (simpel)**
- **KØB**: momentum op + trend op + RSI i “ok” område (ikke overkøbt)
- **HOLD**: blandet billede / afvent
- **SÆLG/UNDA**: høj risiko (stor drawdown, trend ned, eller meget svag momentum)

**Tip**
Brug top-listen som en shortlist, og tjek derefter:
1) Seneste nyheder  
2) Regnskabsdatoer / guidance  
3) Risiko (drawdown/volatilitet) og positionstørrelse

*(Hvis noget fejler: det er næsten altid fordi en import mangler et navn – eller fordi en CSV ikke ligger hvor app’en forventer.)*
"""
