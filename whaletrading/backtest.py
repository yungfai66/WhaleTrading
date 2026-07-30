"""Event-study backtest for the buy/sell signals in signals.py.

Measures forward returns after each fired signal vs. the ticker's own
baseline (and vs. SPY), broken out per signal reason, with an empirical
significance test. Deliberately NOT a trade/equity-curve backtest — see
docs/superpowers/specs/2026-07-31-backtester-design.md for the reasoning.

Run:  python -m whaletrading.backtest            # full watchlist
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import signals
from .config import load_config
from .data import prices as prices_mod
from .data import store

HORIZONS = (4, 8, 13)
N_PERMUTATIONS = 1000
MIN_EVENTS_FOR_PVALUE = 20
SIGNAL_REASONS = ("dip reversal", "ribbon turn", "MACD cross")
SELL_REASONS = ("whale→retail shift", "yellow candle")


@dataclass
class BacktestResult:
    events: pd.DataFrame
    summary: pd.DataFrame
    coverage: pd.DataFrame


def _reconstruct_causal_score(metrics: pd.DataFrame) -> pd.DataFrame:
    """Re-blend whale_score from stored per-bar components, dropping inst_13f.

    metrics.components is the JSON dict written by pipeline.py's
    _recompute_metrics, e.g. {"big_money_volume": 41.2, "dark_pool": 55.0,
    "inst_13f": 60.0}. inst_13f is a single scalar stamped onto every
    historical row (see whale_score.py::inst_13f_score) — using it here would
    leak future 13F filings into past bars. Dropping it and renormalizing
    over whatever remains keeps every value strictly backward-looking.

    Weights are fixed at 1.0 per available component (equal-weight
    renormalization) rather than reusing config.whale_weights: the config
    weights were tuned assuming all three components are present, and this
    module has no access to what config produced a given historical row.
    """
    if metrics.empty or "components" not in metrics:
        return pd.DataFrame()

    def _parse(raw) -> dict:
        try:
            return json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            return {}

    parsed = metrics["components"].apply(_parse)
    big_money = parsed.apply(lambda d: d.get("big_money_volume"))
    dark_pool = parsed.apply(lambda d: d.get("dark_pool"))

    components = pd.DataFrame(
        {"big_money_volume": big_money, "dark_pool": dark_pool}, index=metrics.index
    ).apply(pd.to_numeric, errors="coerce")

    present = components.notna()
    total_present = present.sum(axis=1).replace(0, np.nan)
    causal_whale = (components.fillna(0) * present).sum(axis=1) / total_present

    out = pd.DataFrame(index=metrics.index)
    out["whale_score"] = causal_whale.fillna(50.0).clip(0, 100)
    out["retail_score"] = metrics["retail_score"]
    out["has_dark_pool"] = present["dark_pool"]
    out["has_big_money"] = present["big_money_volume"]
    return out


def _load_causal_frame(conn, ticker: str, timeframe: str) -> pd.DataFrame:
    """OHLCV + causally-reconstructed scores + signals, resampled to timeframe.

    Mirrors app.py::load_ticker_frame's join/resample shape, but rebuilds the
    score from stored components instead of trusting the (partially
    non-causal) stored whale_score/retail_score columns directly.
    """
    daily = store.load_prices(conn, ticker)
    metrics = store.load_metrics(conn, ticker)
    if daily.empty or metrics.empty:
        return pd.DataFrame()

    causal = _reconstruct_causal_score(metrics)
    bars = prices_mod.resample(daily, timeframe)
    if timeframe != "D":
        rule = {"W": "W-FRI", "M": "ME"}[timeframe]
        causal = causal.resample(rule).mean()
        causal["has_dark_pool"] = causal["has_dark_pool"] > 0
        causal["has_big_money"] = causal["has_big_money"] > 0

    frame = bars.join(causal, how="left").ffill().dropna(subset=["whale_score"])
    if frame.empty:
        return frame
    return signals.evaluate(frame)


def _forward_returns(frame: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """close[t+1+h] / open[t+1] - 1 for each bar t and horizon h.

    Entry is always the bar *after* t (never t's own close) so a signal
    computed from bar t's completed data can't be filled at a price that bar
    implies but doesn't yet guarantee. Bars without enough trailing history
    for a given horizon get NaN for that horizon, not a padded value.
    """
    entry = frame["open"].shift(-1)
    out = pd.DataFrame(index=frame.index)
    for h in horizons:
        exit_close = frame["close"].shift(-(1 + h))
        out[f"fwd_ret_{h}"] = exit_close / entry - 1
    return out


def _baseline_means(fwd: pd.DataFrame, horizons: tuple[int, ...]) -> dict[int, float]:
    return {h: float(fwd[f"fwd_ret_{h}"].mean()) for h in horizons}


def _permutation_pvalue(
    all_values: np.ndarray,
    n_events: int,
    observed: float,
    rng: np.random.Generator,
    tail: str = "greater",
) -> float | None:
    """Empirical p-value from random n_events-sized draws vs. the observed
    excess return. None if there isn't enough data to draw from.

    tail="greater": fraction of random draws >= observed — the right test
        for buy reasons, where a real edge shows up as unusually high excess
        return.
    tail="less": fraction of random draws <= observed — the right test for
        sell reasons, where a real edge shows up as unusually *negative*
        excess return. Using "greater" there would report p~1 (looks
        insignificant) for a strong, real negative effect.
    """
    pool = all_values[~np.isnan(all_values)]
    if n_events == 0 or len(pool) < n_events:
        return None
    baseline = pool.mean()
    draws = rng.choice(pool, size=(N_PERMUTATIONS, n_events), replace=True)
    draw_means = draws.mean(axis=1) - baseline
    if tail == "less":
        return float((draw_means <= observed).mean())
    return float((draw_means >= observed).mean())


def _events_for_ticker(
    ticker: str, frame: pd.DataFrame, fwd: pd.DataFrame, horizons: tuple[int, ...]
) -> pd.DataFrame:
    rows = []
    for reason_col, reasons in (("buy_reason", SIGNAL_REASONS), ("sell_reason", SELL_REASONS)):
        for reason in reasons:
            fired = frame[reason_col].fillna("").apply(lambda s: reason in s.split(", "))
            for date in frame.index[fired]:
                row = {
                    "ticker": ticker,
                    "date": date,
                    "reason": reason,
                    "has_dark_pool": bool(frame.loc[date, "has_dark_pool"]),
                }
                for h in horizons:
                    row[f"fwd_ret_{h}"] = fwd.loc[date, f"fwd_ret_{h}"]
                rows.append(row)
    return pd.DataFrame(rows)


def _summarize(
    events: pd.DataFrame, baselines: dict[str, dict[int, float]], horizons: tuple[int, ...],
    all_fwd: dict[str, pd.DataFrame], rng: np.random.Generator,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=["reason", "horizon", "era", "n", "mean_excess_return", "spy_excess_return", "p_value"]
        )

    spy_baseline = baselines.get("SPY", {})
    rows = []
    for reason in sorted(events["reason"].unique()):
        for era, subset in (
            ("all", events[events["reason"] == reason]),
            ("dark_pool_present", events[(events["reason"] == reason) & events["has_dark_pool"]]),
        ):
            for h in horizons:
                col = f"fwd_ret_{h}"
                vals = subset[col].dropna()
                n = len(vals)
                if n == 0:
                    continue
                # Excess vs. each event's own ticker baseline (not one global
                # number) so a mix of tickers with different drift is handled
                # correctly per-event before averaging.
                excess = subset.loc[vals.index].apply(
                    lambda r: r[col] - baselines.get(r["ticker"], {}).get(h, np.nan), axis=1
                ).dropna()
                spy_excess = vals.mean() - spy_baseline.get(h, np.nan) if spy_baseline else np.nan

                pooled_all = np.concatenate(
                    [f[col].to_numpy() for f in all_fwd.values()]
                ) if all_fwd else np.array([])
                tail = "less" if reason in SELL_REASONS else "greater"
                p = (
                    _permutation_pvalue(pooled_all, n, excess.mean(), rng, tail=tail)
                    if n >= MIN_EVENTS_FOR_PVALUE and len(excess)
                    else None
                )
                rows.append(
                    {
                        "reason": reason,
                        "horizon": h,
                        "era": era,
                        "n": n,
                        "mean_excess_return": round(float(excess.mean()), 4) if len(excess) else None,
                        "spy_excess_return": round(float(spy_excess), 4) if not np.isnan(spy_excess) else None,
                        "p_value": round(p, 4) if p is not None else None,
                    }
                )
    return pd.DataFrame(rows)


def run_event_study(
    tickers: list[str],
    timeframe: str = "W",
    horizons: tuple[int, ...] = HORIZONS,
    db_path=None,
    seed: int = 0,
) -> BacktestResult:
    conn = store.connect(db_path)
    rng = np.random.default_rng(seed)

    all_events = []
    all_fwd: dict[str, pd.DataFrame] = {}
    baselines: dict[str, dict[int, float]] = {}
    coverage_rows = []

    for ticker in tickers:
        frame = _load_causal_frame(conn, ticker, timeframe)
        if frame.empty:
            coverage_rows.append(
                {"ticker": ticker, "bars": 0, "events": 0, "start": None, "end": None,
                 "dark_pool_coverage": 0.0}
            )
            continue

        fwd = _forward_returns(frame, horizons)
        all_fwd[ticker] = fwd
        baselines[ticker] = _baseline_means(fwd, horizons)

        events = _events_for_ticker(ticker, frame, fwd, horizons)
        all_events.append(events)

        coverage_rows.append(
            {
                "ticker": ticker,
                "bars": len(frame),
                "events": len(events),
                "start": frame.index.min().strftime("%Y-%m-%d"),
                "end": frame.index.max().strftime("%Y-%m-%d"),
                "dark_pool_coverage": round(float(frame["has_dark_pool"].mean()), 3),
            }
        )

    # SPY as its own "ticker" purely to get a forward-return baseline for the
    # spy_excess_return column — it never contributes events since it isn't
    # run through signals.evaluate() with a whale score of its own here.
    if "SPY" not in tickers:
        spy_daily = store.load_prices(conn, "SPY")
        if not spy_daily.empty:
            spy_bars = prices_mod.resample(spy_daily, timeframe)
            baselines["SPY"] = _baseline_means(_forward_returns(spy_bars, horizons), horizons)
    conn.close()

    events_df = (
        pd.concat(all_events, ignore_index=True) if all_events and any(len(e) for e in all_events)
        else pd.DataFrame(columns=["ticker", "date", "reason", "has_dark_pool"] + [f"fwd_ret_{h}" for h in horizons])
    )
    summary_df = _summarize(events_df, baselines, horizons, all_fwd, rng)
    coverage_df = pd.DataFrame(coverage_rows)

    return BacktestResult(events=events_df, summary=summary_df, coverage=coverage_df)


def main() -> int:
    # Reason strings include non-ASCII characters (e.g. "whale→retail
    # shift"); Windows consoles default to cp1252, which can't encode them.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cfg = load_config()
    result = run_event_study(cfg.all_tickers)
    print("=== Coverage ===")
    print(result.coverage.to_string(index=False))
    print("\n=== Summary (per reason x horizon x era) ===")
    if result.summary.empty:
        print("No events fired across the watchlist — nothing to summarize.")
    else:
        print(result.summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
