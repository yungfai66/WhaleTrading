"""EMA ribbon: trend direction ("red"/bullish vs "blue"/bearish) and tightening."""

from __future__ import annotations

import pandas as pd

RIBBON_SPANS = (8, 13, 21, 34, 55)


def compute_ribbon(close: pd.Series, spans: tuple[int, ...] = RIBBON_SPANS) -> pd.DataFrame:
    """Return EMA columns plus ribbon state.

    - ribbon_bullish: every faster EMA above every slower one (a "red ribbon")
    - ribbon_bearish: full bearish stacking (a "blue ribbon")
    - ribbon_width: (max EMA - min EMA) / close — small width = tight ribbon
    - ribbon_tightening: width shrinking vs. 5 bars ago
    """
    emas = pd.DataFrame(
        {f"ema_{s}": close.ewm(span=s, adjust=False).mean() for s in spans},
        index=close.index,
    )
    ordered_cols = [f"ema_{s}" for s in spans]
    stacked_bull = pd.Series(True, index=close.index)
    stacked_bear = pd.Series(True, index=close.index)
    for fast, slow in zip(ordered_cols, ordered_cols[1:]):
        stacked_bull &= emas[fast] >= emas[slow]
        stacked_bear &= emas[fast] <= emas[slow]

    width = (emas.max(axis=1) - emas.min(axis=1)) / close
    out = emas
    out["ribbon_bullish"] = stacked_bull
    out["ribbon_bearish"] = stacked_bear
    out["ribbon_width"] = width
    out["ribbon_tightening"] = width < width.shift(5)
    return out
