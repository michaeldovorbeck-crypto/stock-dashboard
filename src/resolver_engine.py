from __future__ import annotations


SYMBOL_ALIASES = {
    "NOVO-B": ["NOVO-B.CO", "NOVO-B:XCSE", "NVO"],
    "NOVO-B.CO": ["NOVO-B.CO", "NOVO-B:XCSE", "NVO"],
    "NOVO-B:XCSE": ["NOVO-B:XCSE", "NOVO-B.CO", "NVO"],
    "NVO": ["NVO", "NOVO-B.CO", "NOVO-B:XCSE"],
    "ASML": ["ASML", "ASML.AS"],
    "ASML.AS": ["ASML.AS", "ASML"],
    "TSM": ["TSM", "2330.TW"],
    "2330.TW": ["2330.TW", "TSM"],
    "SAP": ["SAP", "SAP.DE"],
    "SAP.DE": ["SAP.DE", "SAP"],
    "NESN": ["NESN.SW", "NSRGY"],
    "NESN.SW": ["NESN.SW", "NSRGY"],
}


def normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper().replace(" ", "")


def get_alternative_tickers(symbol: str) -> list[str]:
    sym = normalize_symbol(symbol)
    if not sym:
        return []

    candidates: list[str] = []

    if sym in SYMBOL_ALIASES:
        candidates.extend(SYMBOL_ALIASES[sym])

    if "." in sym:
        base = sym.split(".", 1)[0]
        candidates.extend([sym, base, f"{base}:US"])
    elif ":" in sym:
        base = sym.split(":", 1)[0]
        candidates.extend([sym, base])
    else:
        candidates.extend([sym, f"{sym}:US"])

    seen = set()
    out = []
    for c in candidates:
        c = normalize_symbol(c)
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out