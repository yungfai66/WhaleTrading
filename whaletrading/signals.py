"""Buy/sell rules translated from the strategy notes.

A buy needs one of:
  1. red candle + blue (bearish) ribbon tightening + whale accumulation rising
     (or retail falling)
  2. red candle as the ribbon turns bullish ("red ribbon forming") + the same
     accumulation confirmation
  3. MACD golden cross + whale accumulation rising

Sell / trim warnings:
  1. whale accumulation falling while retail accumulation rises (the reversal
     pattern called out in the notes)
  2. yellow candle + falling whale accumulation
"""

from __future__ import annotations

import pandas as pd

from .indicators.candles import classify_candles
from .indicators.macd import compute_macd
from .indicators.ribbons import compute_ribbon

# Minimum score-point move over the confirmation lookback to call it a trend.
DELTA_EPS = 2.0
DELTA_BARS = 3


def evaluate(frame: pd.DataFrame) -> pd.DataFrame:
    """Add indicator + signal columns to a bar frame.

    `frame` needs open/high/low/close/volume plus whale_score / retail_score,
    all on the same (already-resampled) index.
    """
    out = frame.copy()
    out = out.join(classify_candles(out))
    out = out.join(compute_ribbon(out["close"]))
    out = out.join(compute_macd(out["close"]))

    whale_delta = out["whale_score"].diff(DELTA_BARS)
    retail_delta = out["retail_score"].diff(DELTA_BARS)
    whale_rising = whale_delta > DELTA_EPS
    whale_falling = whale_delta < -DELTA_EPS
    retail_rising = retail_delta > DELTA_EPS
    retail_falling = retail_delta < -DELTA_EPS
    accumulation_ok = whale_rising | retail_falling

    ribbon_turning_bullish = out["ribbon_bullish"] & ~out["ribbon_bullish"].shift(
        1, fill_value=False
    )

    buy_dip = out["red_candle"] & out["ribbon_bearish"] & out["ribbon_tightening"] & accumulation_ok
    buy_turn = out["red_candle"] & ribbon_turning_bullish & accumulation_ok
    buy_macd = out["macd_golden_cross"] & whale_rising

    # The notes describe the whale→retail hand-off as a *top* pattern, so only
    # flag it while the ribbon is still bullish — otherwise it fires all the
    # way down a decline.
    sell_shift = whale_falling & retail_rising & out["ribbon_bullish"]
    sell_candle = out["yellow_candle"] & whale_falling

    out["whale_delta"] = whale_delta
    out["retail_delta"] = retail_delta
    out["buy_signal"] = buy_dip | buy_turn | buy_macd
    out["sell_signal"] = sell_shift | sell_candle
    out["buy_reason"] = _reasons(
        {"dip reversal": buy_dip, "ribbon turn": buy_turn, "MACD cross": buy_macd}
    )
    out["sell_reason"] = _reasons(
        {"whale→retail shift": sell_shift, "yellow candle": sell_candle}
    )
    return out


def zone_label(score: float, thresholds: dict) -> str:
    """Threshold badge for a whale score, honoring per-ticker overrides."""
    if score >= thresholds.get("soar", 75):
        return "soar"
    if score >= thresholds.get("rise", 50):
        return "rise"
    if score >= thresholds.get("momentum", 35):
        return "momentum"
    return "weak"


def _reasons(rules: dict[str, pd.Series]) -> pd.Series:
    names = list(rules)
    combined = pd.concat(rules.values(), axis=1)
    combined.columns = names
    return combined.apply(
        lambda row: ", ".join(n for n in names if row[n]), axis=1
    )
