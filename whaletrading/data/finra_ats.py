"""FINRA ATS (dark pool) weekly volume via the free FINRA Query API.

Endpoint (no auth needed for public datasets, subject to rate limits):
    POST https://api.finra.org/data/group/otcMarket/name/weeklySummary

Each row is one symbol × one week × one tier, summed across all ATSs.
Published on a ~2-4 week delay, so this is a confirmation layer, not a
timing signal.
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

log = logging.getLogger(__name__)

API_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
PAGE_LIMIT = 5000


def fetch_weekly(tickers: list[str], weeks: int = 26) -> pd.DataFrame:
    """Return weekly ATS share volume per ticker (columns match ats_weekly table)."""
    wanted = sorted({t.upper() for t in tickers})
    frames = []
    for batch_start in range(0, len(wanted), 50):
        batch = wanted[batch_start : batch_start + 50]
        payload = {
            "limit": PAGE_LIMIT,
            "compareFilters": [
                {
                    "compareType": "EQUAL",
                    "fieldName": "summaryTypeCode",
                    "fieldValue": "ATS_W_SMBL",
                }
            ],
            "domainFilters": [
                {"fieldName": "issueSymbolIdentifier", "values": batch}
            ],
        }
        try:
            resp = requests.post(
                API_URL,
                json=payload,
                headers={"Accept": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            rows = resp.json()
        except Exception as exc:
            log.warning("FINRA ATS API failed (%s) — skipping dark-pool weekly data", exc)
            return pd.DataFrame()
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    col_symbol = _first_col(df, ["issueSymbolIdentifier", "symbol"])
    col_week = _first_col(df, ["weekStartDate", "summaryStartDate"])
    col_shares = _first_col(df, ["totalWeeklyShareQuantity", "shareQuantity"])
    col_trades = _first_col(df, ["totalWeeklyTradeCount", "tradeCount"])
    if not (col_symbol and col_week and col_shares):
        log.warning("FINRA ATS API returned unexpected schema: %s", list(df.columns))
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "ticker": df[col_symbol].astype(str).str.upper(),
            "week_start": pd.to_datetime(df[col_week]).dt.strftime("%Y-%m-%d"),
            "total_shares": pd.to_numeric(df[col_shares], errors="coerce"),
            "total_trades": pd.to_numeric(df[col_trades], errors="coerce")
            if col_trades
            else None,
        }
    ).dropna(subset=["total_shares"])
    # A symbol can appear once per tier — collapse to one row per week.
    out = (
        out.groupby(["ticker", "week_start"], as_index=False)
        .agg(total_shares=("total_shares", "sum"), total_trades=("total_trades", "sum"))
    )
    cutoff = pd.Timestamp.today() - pd.Timedelta(weeks=weeks)
    return out[pd.to_datetime(out["week_start"]) >= cutoff]


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None
