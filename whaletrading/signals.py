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


# Plain-English translations of the internal signal reason labels, for
# non-trader-facing UI text.
REASON_PLAIN = {
    "dip reversal": "price bounced back up (an up candle) while it was in a downtrend and whale buying was increasing",
    "ribbon turn": "the price trend just turned upward (bullish) while whale buying was increasing",
    "MACD cross": "momentum turned upward (a MACD golden cross) while whale buying was increasing",
    "whale→retail shift": "whale buying is falling while regular-investor buying is rising — a pattern often seen near a price peak",
    "yellow candle": "a weak candle formed (price closed near its low for the period) while whale buying was falling",
}

# Verdicts ordered most-actionable-first (used to sort the overview table).
ACTION_ORDER = ["BUY", "TRIM", "WATCH", "HOLD", "WAIT"]
ACTION_LABELS = {
    "BUY": "🟢 Buy",
    "TRIM": "🟠 Trim",
    "WATCH": "🟡 Watch",
    "HOLD": "🔵 Hold",
    "WAIT": "⚪ Wait",
}


def current_action(frame: pd.DataFrame, thresholds: dict) -> dict:
    """Plain-language verdict for "what do I do right now" from an evaluated
    weekly frame (output of `evaluate`).

    Returns {action, label, severity, headline, detail, invalidation}.
    Priority: fresh sell warning > fresh buy setup > setup forming > hold > wait.
    """
    if frame is None or frame.empty:
        return {
            "action": "WAIT",
            "label": ACTION_LABELS["WAIT"],
            "severity": "neutral",
            "headline": "⚪ NO SIGNAL — not enough data yet",
            "detail": "There isn't enough data for this stock yet.",
            "invalidation": "",
        }

    latest = frame.iloc[-1]
    fresh = frame.tail(2)  # this week + last week
    whale = float(latest["whale_score"])
    delta = float(latest.get("whale_delta") or 0)
    zone = zone_label(whale, thresholds)
    trend_word = "rising" if delta > DELTA_EPS else ("falling" if delta < -DELTA_EPS else "steady")
    score_text = (
        f"Whale score (estimated big-investor buying, 0-100) is {whale:.0f} "
        f"— {zone} level — and {trend_word}."
    )

    def _plain(reason_col: str) -> str:
        parts: list[str] = []
        for reasons in fresh[reason_col].dropna():
            for r in str(reasons).split(", "):
                plain = REASON_PLAIN.get(r)
                if plain and plain not in parts:
                    parts.append(plain)
        return "; ".join(parts).capitalize() if parts else ""

    fresh_sell = fresh["sell_signal"].any()
    fresh_buy = fresh["buy_signal"].any()

    if fresh_sell:
        when = "this week" if bool(frame["sell_signal"].iloc[-1]) else "last week"
        return {
            "action": "TRIM",
            "label": ACTION_LABELS["TRIM"],
            "severity": "warning",
            "headline": "🔴 SELL SIGNAL — consider taking some profit",
            "detail": f"A sell signal appeared {when}: {_plain('sell_reason') or 'a pattern suggesting big investors may be selling'}. {score_text}",
            "invalidation": "This would change if whale buying turns back up without the same pattern repeating.",
        }
    if fresh_buy:
        when = "this week" if bool(frame["buy_signal"].iloc[-1]) else "last week"
        return {
            "action": "BUY",
            "label": ACTION_LABELS["BUY"],
            "severity": "success",
            "headline": "🟢 BUY SIGNAL — consider buying gradually, not all at once",
            "detail": f"A buy signal appeared {when}: {_plain('buy_reason') or 'a price reversal backed by rising whale buying'}. {score_text}",
            "invalidation": "This would be undone if whale buying starts falling, or price drops back below where the signal appeared.",
        }
    if delta > DELTA_EPS and zone != "weak" and bool(
        latest.get("ribbon_tightening") or latest.get("ribbon_bearish")
    ):
        return {
            "action": "WATCH",
            "label": ACTION_LABELS["WATCH"],
            "severity": "info",
            "headline": "⚪ NO SIGNAL YET — but conditions may be building toward one",
            "detail": f"{score_text} Prices are moving into a tighter range, which sometimes comes right before a bigger move — but that's not a signal by itself.",
            "invalidation": "Wait for an actual 🟢 buy signal before acting — this is only an early hint.",
        }
    if bool(latest.get("ribbon_bullish")) and delta > -DELTA_EPS:
        return {
            "action": "HOLD",
            "label": ACTION_LABELS["HOLD"],
            "severity": "info",
            "headline": "⚪ NO SIGNAL — price trend still looks positive",
            "detail": f"Price has generally been rising and whale buying isn't falling. {score_text} If you already own this stock, nothing here suggests selling.",
            "invalidation": "Watch for whale buying falling while regular-investor buying rises — that's the sell-signal pattern.",
        }
    return {
        "action": "WAIT",
        "label": ACTION_LABELS["WAIT"],
        "severity": "neutral",
        "headline": "⚪ NO SIGNAL — nothing stands out right now",
        "detail": f"Neither a buy nor a sell signal is active. {score_text}",
        "invalidation": "Re-check when whale buying starts clearly rising or falling.",
    }


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
