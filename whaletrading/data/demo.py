"""Synthetic fixtures for offline/demo mode (WHALETRADING_DEMO=1).

Generates deterministic, plausible-looking data per ticker so the pipeline and
dashboard can be exercised without network access. Not real market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rng(ticker: str) -> np.random.Generator:
    return np.random.default_rng(abs(hash(("whaletrading", ticker))) % (2**32))


def demo_prices(ticker: str, lookback_years: int = 5) -> pd.DataFrame:
    rng = _rng(ticker)
    dates = pd.bdate_range(
        end=pd.Timestamp.today().normalize(), periods=lookback_years * 252
    )
    n = len(dates)
    # Regime-switching drift so ribbons/candles/signals get realistic phases.
    regime_len = rng.integers(60, 180)
    drifts = []
    while len(drifts) < n:
        drifts.extend([rng.normal(0.0005, 0.0015)] * int(regime_len))
        regime_len = rng.integers(60, 180)
    drift = np.array(drifts[:n])
    vol = rng.uniform(0.015, 0.035)
    rets = drift + rng.normal(0, vol, n)
    close = 20 * np.exp(np.cumsum(rets)) * rng.uniform(0.5, 8.0)

    spread = np.abs(rng.normal(0, vol, n)) + 0.004
    high = close * (1 + spread * rng.uniform(0.4, 1.0, n))
    low = close * (1 - spread * rng.uniform(0.4, 1.0, n))
    open_ = low + (high - low) * rng.uniform(0.1, 0.9, n)

    base_volume = rng.uniform(2e6, 6e7)
    volume = base_volume * np.exp(rng.normal(0, 0.4, n))
    # Whale days: sparse large-volume days that push price with the trend.
    whale_days = rng.random(n) < 0.06
    volume[whale_days] *= rng.uniform(2.0, 4.0, whale_days.sum())

    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([open_, close, high]),
            "low": np.minimum.reduce([open_, close, low]),
            "close": close,
            "volume": volume.astype(int),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def demo_short_volume(ticker: str, prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    rng = _rng(ticker + ":shv")
    n = len(prices)
    # Off-exchange share ~40% of consolidated volume; short ratio mean-reverting.
    off_exchange = (prices["volume"] * rng.uniform(0.3, 0.5)).astype(int)
    ratio = 0.45 + np.cumsum(rng.normal(0, 0.01, n))
    ratio = 0.45 + (ratio - ratio.mean()) * 0.3
    ratio = np.clip(ratio, 0.25, 0.7)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": prices.index.strftime("%Y-%m-%d"),
            "short_volume": (off_exchange * ratio).astype(int),
            "short_exempt_volume": (off_exchange * 0.005).astype(int),
            "total_volume": off_exchange,
        }
    )


def demo_13f(ticker: str, managers: list[dict]) -> pd.DataFrame:
    rng = _rng(ticker + ":13f")
    quarter_ends = pd.date_range(end=pd.Timestamp.today(), periods=3, freq="QE")[:2]
    rows = []
    for mgr in managers[:5]:
        base = int(rng.uniform(1e6, 5e7))
        for i, q in enumerate(quarter_ends):
            drift = rng.uniform(-0.15, 0.25)
            shares = int(base * (1 + drift * i))
            rows.append(
                {
                    "ticker": ticker,
                    "report_period": q.strftime("%Y-%m-%d"),
                    "manager_cik": int(mgr["cik"]),
                    "manager_name": mgr.get("name", str(mgr["cik"])),
                    "shares": shares,
                    "value_usd": shares * 100,
                }
            )
    return pd.DataFrame(rows)


def demo_ats_weekly(ticker: str, prices: pd.DataFrame, weeks: int = 26) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    rng = _rng(ticker + ":ats")
    weekly_vol = prices["volume"].resample("W-MON").sum().tail(weeks)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "week_start": weekly_vol.index.strftime("%Y-%m-%d"),
            "total_shares": (weekly_vol * rng.uniform(0.1, 0.2)).astype(int),
            "total_trades": (weekly_vol / rng.uniform(150, 400)).astype(int),
        }
    )
