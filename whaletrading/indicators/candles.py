"""Candle classification in the spirit of the strategy notes.

"Red candle"    = bullish reversal marker: a strong close (upper part of the
                  bar's range) appearing after a decline. Valid only when it
                  coincides with rising whale accumulation — that check lives
                  in signals.py, not here.
"Yellow candle" = bearish warning: a weak close after an advance.
"""

from __future__ import annotations

import pandas as pd


def classify_candles(ohlc: pd.DataFrame, trend_window: int = 20) -> pd.DataFrame:
    """Return red_candle / yellow_candle booleans per bar."""
    rng = (ohlc["high"] - ohlc["low"]).replace(0, pd.NA)
    close_location = ((ohlc["close"] - ohlc["low"]) / rng).fillna(0.5)
    sma = ohlc["close"].rolling(trend_window, min_periods=trend_window // 2).mean()

    after_decline = ohlc["close"].shift(1) < sma.shift(1)
    after_advance = ohlc["close"].shift(1) > sma.shift(1)
    bullish_bar = (ohlc["close"] > ohlc["open"]) & (close_location > 0.6)
    bearish_bar = (ohlc["close"] < ohlc["open"]) & (close_location < 0.4)

    return pd.DataFrame(
        {
            "red_candle": bullish_bar & after_decline,
            "yellow_candle": bearish_bar & after_advance,
            "close_location": close_location,
        },
        index=ohlc.index,
    )
