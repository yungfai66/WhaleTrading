"""Daily OHLCV via yfinance (free, no key). Weekly/monthly are resampled."""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def fetch_daily(ticker: str, lookback_years: int = 5) -> pd.DataFrame:
    """Return daily OHLCV indexed by date, or an empty frame on failure."""
    try:
        df = yf.download(
            ticker,
            period=f"{lookback_years}y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # network, delisted ticker, etc.
        log.warning("yfinance failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        log.warning("yfinance returned no data for %s (invalid/delisted ticker?)", ticker)
        return pd.DataFrame()

    # yfinance returns a (field, ticker) MultiIndex even for one symbol.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.dropna(subset=["close"])


def resample(daily: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample daily OHLCV to 'W' or 'M' bars ('D' returns as-is)."""
    if timeframe == "D" or daily.empty:
        return daily
    rule = {"W": "W-FRI", "M": "ME"}[timeframe]
    out = daily.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["close"])


def fetch_quote(ticker: str) -> float | None:
    """Best-effort current quote (Yahoo's public feed, ~15 min delayed).

    Distinct from fetch_daily()'s end-of-day bar: this can reflect today's
    price even mid-session, before the daily bar finalizes at the close.
    Returns None on any failure so callers can fall back to the daily close.
    """
    try:
        fast_info = yf.Ticker(ticker).fast_info
        price = fast_info.get("last_price") if hasattr(fast_info, "get") else fast_info["last_price"]
        return float(price) if price else None
    except Exception as exc:
        log.warning("yfinance quote failed for %s: %s", ticker, exc)
        return None


def company_name(ticker: str) -> str | None:
    """Best-effort issuer name (used to match 13F holdings without an alias)."""
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("shortName") or info.get("longName")
    except Exception:
        return None
