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
    "dip reversal": "a strong bounce candle appeared during the downtrend while whale accumulation was rising",
    "ribbon turn": "the trend just flipped bullish with whale accumulation behind it",
    "MACD cross": "momentum turned up (MACD golden cross) while whales were buying",
    "whale→retail shift": "whales appear to be handing shares to retail buyers near the top",
    "yellow candle": "a weak candle formed while whale accumulation was falling",
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
            "headline": "Wait — no data",
            "detail": "No data available for this ticker yet.",
            "invalidation": "",
        }

    latest = frame.iloc[-1]
    fresh = frame.tail(2)  # this week + last week
    whale = float(latest["whale_score"])
    delta = float(latest.get("whale_delta") or 0)
    zone = zone_label(whale, thresholds)
    trend_word = "rising" if delta > DELTA_EPS else ("falling" if delta < -DELTA_EPS else "flat")
    score_text = f"Whale score is {whale:.0f} ({zone} zone) and {trend_word}."

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
            "headline": "Trim / take profit — whales look like they're selling",
            "detail": f"A sell warning fired {when}: {_plain('sell_reason') or 'distribution pattern'}. {score_text}",
            "invalidation": "Reconsider if whale accumulation turns back up without the pattern repeating.",
        }
    if fresh_buy:
        when = "this week" if bool(frame["buy_signal"].iloc[-1]) else "last week"
        return {
            "action": "BUY",
            "label": ACTION_LABELS["BUY"],
            "severity": "success",
            "headline": "Buy setup active — consider dollar-cost averaging in",
            "detail": f"A buy signal fired {when}: {_plain('buy_reason') or 'reversal pattern with whale support'}. {score_text}",
            "invalidation": "Setup is invalidated if whale accumulation starts falling or price breaks below the signal candle's low.",
        }
    if delta > DELTA_EPS and zone != "weak" and bool(
        latest.get("ribbon_tightening") or latest.get("ribbon_bearish")
    ):
        return {
            "action": "WATCH",
            "label": ACTION_LABELS["WATCH"],
            "severity": "info",
            "headline": "Setup forming — whales accumulating, no entry candle yet",
            "detail": f"{score_text} The trend ribbon is compressing, which often precedes a move.",
            "invalidation": "Wait for a bullish candle or MACD cross to confirm before buying.",
        }
    if bool(latest.get("ribbon_bullish")) and delta > -DELTA_EPS:
        return {
            "action": "HOLD",
            "label": ACTION_LABELS["HOLD"],
            "severity": "info",
            "headline": "Hold — uptrend intact, whales still in",
            "detail": f"The trend ribbon is bullish and whale accumulation is holding. {score_text}",
            "invalidation": "Watch for whale accumulation falling while retail rises — that's the trim warning.",
        }
    return {
        "action": "WAIT",
        "label": ACTION_LABELS["WAIT"],
        "severity": "neutral",
        "headline": "Wait — no edge right now",
        "detail": f"No active setup. {score_text}",
        "invalidation": "Re-check when whale accumulation starts rising or a buy signal fires.",
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
