# src/search_engine.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.theme_definitions import THEMES


UNIVERSE_DIR = Path("data") / "universes"


def _load_all_universe_rows() -> pd.DataFrame:
    rows = []

    if not UNIVERSE_DIR.exists():
        return pd.DataFrame()

    for file_path in sorted(UNIVERSE_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue

        if df.empty:
            continue

        df.columns = [c.strip().lower() for c in df.columns]

        if "ticker" not in df.columns and "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ticker"})
        if "name" not in df.columns and "company" in df.columns:
            df = df.rename(columns={"company": "name"})

        if "ticker" not in df.columns:
            continue

        if "name" not in df.columns:
            df["name"] = ""

        if "sector" not in df.columns:
            df["sector"] = ""

        if "country" not in df.columns:
            df["country"] = ""

        df["ticker"] = df["ticker"].astype(str).str.strip()
        df["name"] = df["name"].astype(str).str.strip()
        df["sector"] = df["sector"].astype(str).str.strip()
        df["country"] = df["country"].astype(str).str.strip()
        df["source_file"] = file_path.name
        rows.append(df[["ticker", "name", "sector", "country", "source_file"]])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return out


def build_search_index() -> pd.DataFrame:
    uni = _load_all_universe_rows()

    theme_rows = []
    for theme_name, cfg in THEMES.items():
        proxy = str(cfg.get("proxy", "")).strip()
        benchmark = str(cfg.get("benchmark", "")).strip()
        members = cfg.get("members", [])
        etfs = cfg.get("etfs", [])
        leaders = cfg.get("leaders", [])
        description = str(cfg.get("description", "")).strip()

        if proxy:
            theme_rows.append(
                {
                    "ticker": proxy,
                    "name": f"{theme_name} proxy",
                    "type": "Theme Proxy",
                    "themes": theme_name,
                    "sector": "",
                    "country": "",
                    "description": description,
                }
            )

        if benchmark:
            theme_rows.append(
                {
                    "ticker": benchmark,
                    "name": f"{theme_name} benchmark",
                    "type": "Benchmark",
                    "themes": theme_name,
                    "sector": "",
                    "country": "",
                    "description": description,
                }
            )

        for x in etfs:
            theme_rows.append(
                {
                    "ticker": str(x).strip(),
                    "name": "",
                    "type": "ETF",
                    "themes": theme_name,
                    "sector": "",
                    "country": "",
                    "description": description,
                }
            )

        for x in leaders:
            theme_rows.append(
                {
                    "ticker": str(x).strip(),
                    "name": "",
                    "type": "Leader",
                    "themes": theme_name,
                    "sector": "",
                    "country": "",
                    "description": description,
                }
            )

        for x in members:
            theme_rows.append(
                {
                    "ticker": str(x).strip(),
                    "name": "",
                    "type": "Member",
                    "themes": theme_name,
                    "sector": "",
                    "country": "",
                    "description": description,
                }
            )

    theme_df = pd.DataFrame(theme_rows) if theme_rows else pd.DataFrame(
        columns=["ticker", "name", "type", "themes", "sector", "country", "description"]
    )

    if uni.empty and theme_df.empty:
        return pd.DataFrame()

    if uni.empty:
        out = theme_df.copy()
    else:
        uni = uni.copy()
        uni["type"] = "Asset"
        uni["themes"] = ""
        uni["description"] = ""

        out = uni.merge(
            theme_df.groupby("ticker", as_index=False).agg(
                themes=("themes", lambda s: ", ".join(sorted(set([x for x in s if str(x).strip()])))),
                type=("type", lambda s: ", ".join(sorted(set([x for x in s if str(x).strip()])))),
                description=("description", lambda s: " | ".join(sorted(set([x for x in s if str(x).strip()])))),
            ),
            on="ticker",
            how="left",
            suffixes=("", "_theme"),
        )

        out["themes"] = out["themes"].fillna("")
        out["description"] = out["description"].fillna("")
        out["type"] = out["type_theme"].fillna("Asset")
        out = out.drop(columns=["type_theme"], errors="ignore")

        missing_theme_only = theme_df[~theme_df["ticker"].isin(out["ticker"])].copy()
        if not missing_theme_only.empty:
            missing_theme_only["source_file"] = ""
            out = pd.concat(
                [
                    out[["ticker", "name", "sector", "country", "source_file", "type", "themes", "description"]],
                    missing_theme_only[["ticker", "name", "sector", "country", "source_file", "type", "themes", "description"]],
                ],
                ignore_index=True,
            )

    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["name"] = out["name"].astype(str).fillna("").str.strip()
    out["sector"] = out["sector"].astype(str).fillna("").str.strip()
    out["country"] = out["country"].astype(str).fillna("").str.strip()
    out["type"] = out["type"].astype(str).fillna("").str.strip()
    out["themes"] = out["themes"].astype(str).fillna("").str.strip()
    out["description"] = out["description"].astype(str).fillna("").str.strip()

    out = out[out["ticker"] != ""].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return out


def search_assets(query: str, limit: int = 25) -> pd.DataFrame:
    index_df = build_search_index()
    if index_df.empty:
        return pd.DataFrame()

    q = (query or "").strip().lower()
    if not q:
        return index_df.head(limit).reset_index(drop=True)

    work = index_df.copy()

    score = pd.Series(0, index=work.index, dtype=float)

    score += work["ticker"].astype(str).str.lower().eq(q).astype(float) * 100
    score += work["ticker"].astype(str).str.lower().str.startswith(q).astype(float) * 40
    score += work["ticker"].astype(str).str.lower().str.contains(q, na=False).astype(float) * 20

    score += work["name"].astype(str).str.lower().eq(q).astype(float) * 80
    score += work["name"].astype(str).str.lower().str.contains(q, na=False).astype(float) * 25

    score += work["themes"].astype(str).str.lower().str.contains(q, na=False).astype(float) * 15
    score += work["sector"].astype(str).str.lower().str.contains(q, na=False).astype(float) * 10
    score += work["type"].astype(str).str.lower().str.contains(q, na=False).astype(float) * 10

    work["search_score"] = score
    work = work[work["search_score"] > 0].copy()

    if work.empty:
        return pd.DataFrame()

    work = work.sort_values(["search_score", "ticker"], ascending=[False, True]).head(limit).reset_index(drop=True)
    return work


def find_asset_record(ticker: str) -> dict:
    index_df = build_search_index()
    if index_df.empty:
        return {}

    t = (ticker or "").strip().upper()
    row = index_df[index_df["ticker"].astype(str).str.upper() == t]
    if row.empty:
        return {}

    return row.iloc[0].to_dict()