"""Market-wide "Fear & Greed" composite, modeled on CNN's index
(https://edition.cnn.com/markets/fear-and-greed) but built entirely from
free Yahoo Finance data — no scraping of CNN's page or its private API.

CNN blends 7 indicators; we compute 6 free proxies for them and openly
document what's missing:

  momentum    — S&P 500 vs its 125-day average (CNN: same idea, exact match)
  volatility  — VIX vs its 50-day average, inverted (CNN: same idea)
  safe_haven  — SPY 20d return minus TLT 20d return (CNN: stocks vs Treasuries)
  junk_bond   — HYG 20d return minus LQD 20d return, as a spread proxy
                (CNN: real high-yield/investment-grade spread — we don't have
                a free daily spread feed, so ETF total-return divergence
                stands in for it)
  strength    — share of a large-cap basket near its 52-week high minus the
                share near its 52-week low (CNN: full NYSE new-highs/lows)
  breadth     — McClellan-style summation of net advancing volume across the
                same basket (CNN: full NYSE advance/decline volume)

  put/call    — OMITTED. No free daily put/call history exists; CNN's 7th
                indicator has no proxy here. The composite is a 6-indicator
                average, not 7.

Each raw signal is z-scored against its own trailing ~1-year history, then
mapped to 0-100 with the same tanh squash (`_to_score`) the whale score uses
— 50 is neutral, same normalization idiom as whale_score.py so the whole app
speaks one statistical language.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .whale_score import _to_score

# Index/ETF symbols the six indicators are built from.
INDEX_SYMBOLS = ("^GSPC", "^VIX", "SPY", "TLT", "HYG", "LQD")

# Large-cap basket used only for the Strength / Breadth proxies — a fixed
# market cross-section, deliberately independent of the user's watchlist so
# the index doesn't drift as tickers are added/removed. Spread across
# sectors (tech, financials, healthcare, energy, consumer, industrials,
# utilities, communications) so it approximates "the market", not one
# sector's mood.
BASKET = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL", "CRM", "ADBE",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP",
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK",
    "XOM", "CVX", "COP",
    "WMT", "PG", "KO", "PEP", "MCD", "HD", "NKE",
    "CAT", "BA", "HON", "UPS", "GE",
    "NEE", "DUK",
    "DIS", "NFLX", "VZ", "T",
)
BASKET_MIN = 20  # minimum basket tickers with data before trusting strength/breadth

REQUIRED_SYMBOLS = INDEX_SYMBOLS + BASKET

# Score bands — single source of truth for the page, the header chip, and
# the history-chart shading.
FG_BANDS = [
    (0, 25, "Extreme Fear"),
    (25, 45, "Fear"),
    (45, 55, "Neutral"),
    (55, 75, "Greed"),
    (75, 101, "Extreme Greed"),
]

INDICATOR_INFO = {
    "momentum": ("Market Momentum", "S&P 500 vs. its 125-day average — above is bullish."),
    "volatility": ("Market Volatility", "VIX vs. its 50-day average — a calmer-than-usual VIX reads as greed."),
    "strength": ("Stock Price Strength", "Large-caps near 52-week highs vs. near 52-week lows."),
    "breadth": ("Stock Price Breadth", "Volume-weighted advance/decline summation across large-caps."),
    "safe_haven": ("Safe Haven Demand", "Stocks (SPY) vs. Treasuries (TLT) over 20 days — Treasuries winning is fear."),
    "junk_bond": ("Junk Bond Demand", "High-yield (HYG) vs. investment-grade (LQD) bonds over 20 days — a narrowing gap is greed."),
}


def label(score: float) -> str:
    for lo, hi, name in FG_BANDS:
        if lo <= score < hi:
            return name
    return FG_BANDS[0][2] if score < FG_BANDS[0][0] else FG_BANDS[-1][2]


def _score_from_raw(raw: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score of `raw` against its own trailing `window`, tanh-squashed
    to 0-100. Same shape as whale_score.dark_pool_score's normalization.

    The rolling std is floored at a small fraction of the series' own
    overall scale. Without this, a stretch where `raw` goes nearly flat
    (e.g. momentum during a very steady, low-noise trend) drives std toward
    zero, and dividing by a near-zero std turns ordinary floating-point
    noise into a wildly whipsawing z-score — verified this actually happens
    with a smooth synthetic trend before adding the floor.
    """
    raw = raw.dropna()
    mean = raw.rolling(window, min_periods=window // 3).mean()
    std = raw.rolling(window, min_periods=window // 3).std()
    floor = raw.abs().median() * 0.05 + 1e-9
    std = std.clip(lower=floor)
    z = ((raw - mean) / std).clip(-4, 4).fillna(0)
    return pd.Series(_to_score(z, scale=0.6), index=raw.index)


def _momentum_raw(gspc: pd.DataFrame) -> pd.Series:
    close = gspc["close"]
    sma = close.rolling(125, min_periods=125).mean()
    return (close / sma - 1.0).dropna()


def _volatility_raw(vix: pd.DataFrame) -> pd.Series:
    close = vix["close"]
    sma = close.rolling(50, min_periods=50).mean()
    return (-(close / sma - 1.0)).dropna()


def _safe_haven_raw(spy: pd.DataFrame, tlt: pd.DataFrame) -> pd.Series:
    spy_ret = spy["close"].pct_change(20)
    tlt_ret = tlt["close"].pct_change(20)
    return (spy_ret - tlt_ret).dropna()


def _junk_bond_raw(hyg: pd.DataFrame, lqd: pd.DataFrame) -> pd.Series:
    hyg_ret = hyg["close"].pct_change(20)
    lqd_ret = lqd["close"].pct_change(20)
    return (hyg_ret - lqd_ret).dropna()


def _strength_raw(basket: dict[str, pd.DataFrame]) -> pd.Series:
    near_high, near_low = [], []
    for df in basket.values():
        close = df["close"]
        roll_max = close.rolling(252, min_periods=252 // 2).max()
        roll_min = close.rolling(252, min_periods=252 // 2).min()
        near_high.append(close >= roll_max * 0.98)
        near_low.append(close <= roll_min * 1.02)
    share_high = pd.concat(near_high, axis=1).mean(axis=1, skipna=True)
    share_low = pd.concat(near_low, axis=1).mean(axis=1, skipna=True)
    return (share_high - share_low).dropna()


def _breadth_raw(basket: dict[str, pd.DataFrame]) -> pd.Series:
    net_parts = [np.sign(df["close"].diff()) * df["volume"].astype(float) for df in basket.values()]
    total_parts = [df["volume"].astype(float) for df in basket.values()]
    net_vol = pd.concat(net_parts, axis=1).sum(axis=1, skipna=True)
    total_vol = pd.concat(total_parts, axis=1).sum(axis=1, skipna=True).replace(0, np.nan)
    ratio = (net_vol / total_vol).dropna()
    oscillator = ratio.ewm(span=19, adjust=False).mean() - ratio.ewm(span=39, adjust=False).mean()
    return oscillator.cumsum().dropna()


def compute(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long-format result: columns ['date', 'indicator', 'score', 'raw'].

    `frames` maps symbol -> daily OHLCV (as returned by prices.fetch_daily /
    fetch_daily_batch), for whichever of REQUIRED_SYMBOLS were fetched
    successfully. Missing symbols simply drop that indicator — the
    'composite' row is the mean of whatever indicators exist that day,
    renormalizing over missing sources the same way composite_whale_score
    does.
    """
    raws: dict[str, pd.Series] = {}

    gspc, vix = frames.get("^GSPC"), frames.get("^VIX")
    if gspc is not None and not gspc.empty:
        raws["momentum"] = _momentum_raw(gspc)
    if vix is not None and not vix.empty:
        raws["volatility"] = _volatility_raw(vix)

    spy, tlt = frames.get("SPY"), frames.get("TLT")
    if spy is not None and tlt is not None and not spy.empty and not tlt.empty:
        raws["safe_haven"] = _safe_haven_raw(spy, tlt)

    hyg, lqd = frames.get("HYG"), frames.get("LQD")
    if hyg is not None and lqd is not None and not hyg.empty and not lqd.empty:
        raws["junk_bond"] = _junk_bond_raw(hyg, lqd)

    basket_frames = {t: frames[t] for t in BASKET if t in frames and frames[t] is not None and not frames[t].empty}
    if len(basket_frames) >= BASKET_MIN:
        raws["strength"] = _strength_raw(basket_frames)
        raws["breadth"] = _breadth_raw(basket_frames)

    if not raws:
        return pd.DataFrame(columns=["date", "indicator", "score", "raw"])

    score_df = pd.DataFrame({name: _score_from_raw(raw) for name, raw in raws.items()})
    score_df["composite"] = score_df.mean(axis=1, skipna=True)
    raw_df = pd.DataFrame(raws).reindex(score_df.index)

    long_rows = []
    for indicator in score_df.columns:
        sub = pd.DataFrame(
            {"score": score_df[indicator], "raw": raw_df.get(indicator)}
        ).dropna(subset=["score"])
        sub["indicator"] = indicator
        long_rows.append(sub)

    out = pd.concat(long_rows)
    out.index.name = "date"
    return out.reset_index()
