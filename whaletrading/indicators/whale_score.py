"""Composite "whale accumulation" score (0-100) and its retail counterpart.

No free feed labels trades institutional vs retail, so the score blends
free *proxies* (weights come from config and are renormalized when a
source is missing):

  big_money_volume — days whose volume z-score exceeds a threshold are
      treated as big-money days; each is classified accumulation vs
      distribution by where the close lands in the bar's range (Chaikin
      close-location value), then netted over a rolling window.
  dark_pool — FINRA daily off-exchange short-volume ratio vs its own
      baseline. Persistent above-baseline readings are commonly read as
      hidden accumulation (market makers shorting to fill large buyers).
  inst_13f — QoQ change in shares held by the tracked 13F managers
      (quarterly, 45-day lag: a slow confirmation layer).

The retail score mirrors big_money_volume but nets the *low*-volume days,
where up-moves on thin volume suggest retail chasing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clv(ohlc: pd.DataFrame) -> pd.Series:
    """Close-location value in [-1, 1]: +1 close at high, -1 close at low."""
    rng = (ohlc["high"] - ohlc["low"]).replace(0, np.nan)
    return (((ohlc["close"] - ohlc["low"]) - (ohlc["high"] - ohlc["close"])) / rng).fillna(0)


def _to_score(x: pd.Series | float, scale: float = 1.0):
    """Map a roughly-unit-scale signal to 0-100 via tanh (50 = neutral)."""
    return 50.0 * (1.0 + np.tanh(np.asarray(x, dtype=float) * scale))


def big_money_scores(
    prices: pd.DataFrame,
    baseline_window: int = 60,
    zscore_threshold: float = 1.25,
    flow_window: int = 20,
) -> pd.DataFrame:
    """Whale + retail flow scores from OHLCV alone. Index matches `prices`."""
    vol = prices["volume"].astype(float)
    mean = vol.rolling(baseline_window, min_periods=baseline_window // 3).mean()
    std = vol.rolling(baseline_window, min_periods=baseline_window // 3).std()
    vol_z = ((vol - mean) / std).fillna(0)

    clv = _clv(prices)
    signed_flow = clv * vol

    whale_flow = signed_flow.where(vol_z >= zscore_threshold, 0.0)
    retail_flow = signed_flow.where(vol_z < 0, 0.0)

    denom = vol.rolling(flow_window, min_periods=flow_window // 2).sum()
    whale_net = (whale_flow.rolling(flow_window, min_periods=flow_window // 2).sum() / denom).fillna(0)
    retail_net = (retail_flow.rolling(flow_window, min_periods=flow_window // 2).sum() / denom).fillna(0)

    return pd.DataFrame(
        {
            "big_money_score": _to_score(whale_net, scale=4.0),
            "retail_score": _to_score(retail_net, scale=4.0),
            "volume_zscore": vol_z,
        },
        index=prices.index,
    )


def dark_pool_score(
    short_volume: pd.DataFrame, baseline_window: int = 60
) -> pd.Series | None:
    """Score from FINRA daily off-exchange short-volume ratio deviations."""
    if short_volume is None or short_volume.empty:
        return None
    sv = short_volume.sort_index()
    ratio = (sv["short_volume"] / sv["total_volume"].replace(0, np.nan)).dropna()
    if len(ratio) < baseline_window // 3:
        return None
    mean = ratio.rolling(baseline_window, min_periods=baseline_window // 3).mean()
    std = ratio.rolling(baseline_window, min_periods=baseline_window // 3).std().replace(0, np.nan)
    z = ((ratio - mean) / std).clip(-4, 4)
    smoothed = z.ewm(span=5, adjust=False).mean().fillna(0)
    return pd.Series(_to_score(smoothed, scale=0.6), index=ratio.index)


def inst_13f_score(holdings: pd.DataFrame) -> tuple[float | None, float | None]:
    """(score, qoq_pct_change) from the last two report periods, else (None, None)."""
    if holdings is None or holdings.empty:
        return None, None
    by_period = holdings.groupby("report_period")["shares"].sum().sort_index()
    if len(by_period) < 2:
        return None, None
    prev, last = float(by_period.iloc[-2]), float(by_period.iloc[-1])
    if prev <= 0:
        return None, None
    pct = (last - prev) / prev
    return float(_to_score(pct, scale=10.0)), pct


def composite_whale_score(
    prices: pd.DataFrame,
    short_volume: pd.DataFrame | None,
    holdings_13f: pd.DataFrame | None,
    weights: dict,
    baseline_window: int = 60,
    zscore_threshold: float = 1.25,
    flow_window: int = 20,
) -> pd.DataFrame:
    """Daily whale_score / retail_score plus per-component columns."""
    bm = big_money_scores(prices, baseline_window, zscore_threshold, flow_window)
    components = pd.DataFrame(index=prices.index)
    components["big_money_volume"] = bm["big_money_score"]

    dp = dark_pool_score(short_volume, baseline_window)
    if dp is not None:
        components["dark_pool"] = dp.reindex(prices.index).ffill()

    score_13f, pct_13f = inst_13f_score(holdings_13f)
    if score_13f is not None:
        components["inst_13f"] = score_13f

    # Weighted blend, renormalizing over whichever components exist per row.
    weight_row = pd.DataFrame(
        {c: components[c].notna() * float(weights.get(c, 0)) for c in components.columns},
        index=components.index,
    )
    total_w = weight_row.sum(axis=1).replace(0, np.nan)
    whale = (components.fillna(0) * weight_row).sum(axis=1) / total_w

    out = components.copy()
    out["whale_score"] = whale.fillna(50.0).clip(0, 100)
    out["retail_score"] = bm["retail_score"].clip(0, 100)
    out["volume_zscore"] = bm["volume_zscore"]
    out.attrs["pct_13f"] = pct_13f
    return out
