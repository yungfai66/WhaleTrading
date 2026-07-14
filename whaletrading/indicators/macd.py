"""Standard MACD (12/26/9) with golden/death cross flags."""

from __future__ import annotations

import pandas as pd


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    macd = (
        close.ewm(span=fast, adjust=False).mean()
        - close.ewm(span=slow, adjust=False).mean()
    )
    sig = macd.ewm(span=signal, adjust=False).mean()
    above = macd > sig
    return pd.DataFrame(
        {
            "macd": macd,
            "macd_signal": sig,
            "macd_hist": macd - sig,
            "macd_golden_cross": above & ~above.shift(1, fill_value=False),
            "macd_death_cross": ~above & above.shift(1, fill_value=True),
        },
        index=close.index,
    )
