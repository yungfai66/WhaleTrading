"""Refresh pipeline: fetch each source → compute metrics → persist to SQLite.

Every source degrades gracefully: if FINRA / EDGAR / Yahoo is unreachable the
affected component is dropped and the composite renormalizes over what's left.

Run:  python -m whaletrading.pipeline            # live data
      WHALETRADING_DEMO=1 python -m whaletrading.pipeline   # synthetic demo data
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta

import pandas as pd

from .config import Config, load_config
from .data import demo, finra_ats, finra_short_volume, prices, sec_13f, store
from .indicators.whale_score import composite_whale_score

log = logging.getLogger(__name__)


def refresh_all(
    cfg: Config | None = None, db_path=None, tickers: list[str] | None = None
) -> dict:
    """Refresh the given tickers (default: cfg.watchlist); returns a
    per-source summary for the UI/CLI.

    `tickers` lets a caller refresh a different (e.g. session-customized)
    watchlist than the one in config/watchlist.yaml — used by the app's
    "add a ticker" feature so a newly added symbol actually gets fetched.
    """
    cfg = cfg or load_config()
    tickers = list(tickers) if tickers else list(cfg.watchlist)
    conn = store.connect(db_path)
    summary: dict = {"tickers": {}, "demo_mode": cfg.demo_mode}

    price_frames = _refresh_prices(conn, cfg, summary, tickers)
    _refresh_short_volume(conn, cfg, summary, price_frames, tickers)
    _refresh_ats(conn, cfg, summary, price_frames, tickers)
    _refresh_13f(conn, cfg, summary, tickers)
    _recompute_metrics(conn, cfg, summary, price_frames, tickers)

    store.mark_refreshed(conn, "pipeline")
    conn.close()
    return summary


def _refresh_prices(conn, cfg: Config, summary: dict, tickers: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = (
            demo.demo_prices(ticker, cfg.price_lookback_years)
            if cfg.demo_mode
            else prices.fetch_daily(ticker, cfg.price_lookback_years)
        )
        if df.empty:
            summary["tickers"].setdefault(ticker, []).append("prices: FAILED")
            # fall back to whatever is already cached
            cached = store.load_prices(conn, ticker)
            if not cached.empty:
                frames[ticker] = cached
                summary["tickers"][ticker].append("prices: using cache")
            continue
        frames[ticker] = df
        rows = df.reset_index()
        rows["ticker"] = ticker
        rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
        store.upsert_df(
            conn, "prices", rows[["ticker", "date", "open", "high", "low", "close", "volume"]]
        )
        summary["tickers"].setdefault(ticker, []).append(f"prices: {len(df)} days")
    store.mark_refreshed(conn, "prices")
    return frames


def _refresh_short_volume(conn, cfg: Config, summary: dict, price_frames, tickers: list[str]) -> None:
    if cfg.demo_mode:
        for ticker, pf in price_frames.items():
            df = demo.demo_short_volume(ticker, pf)
            store.upsert_df(conn, "short_volume", df)
        store.mark_refreshed(conn, "short_volume")
        return

    # Incremental: only fetch days FINRA has that we don't (per whole-batch dates).
    cached_dates = {
        row[0]
        for row in conn.execute("SELECT DISTINCT date FROM short_volume").fetchall()
    }
    start = date.today() - timedelta(days=cfg.finra_short_volume_days)
    df = finra_short_volume.fetch_range(tickers, start=start, skip_dates=cached_dates)
    if df.empty and not cached_dates:
        summary["sources_failed"] = summary.get("sources_failed", []) + ["finra_short_volume"]
        log.warning("no FINRA short-volume data fetched (network blocked?)")
        return
    store.upsert_df(conn, "short_volume", df)
    store.mark_refreshed(conn, "short_volume")


def _refresh_ats(conn, cfg: Config, summary: dict, price_frames, tickers: list[str]) -> None:
    if cfg.demo_mode:
        for ticker, pf in price_frames.items():
            store.upsert_df(conn, "ats_weekly", demo.demo_ats_weekly(ticker, pf))
        store.mark_refreshed(conn, "ats_weekly")
        return
    df = finra_ats.fetch_weekly(tickers)
    if df.empty:
        summary["sources_failed"] = summary.get("sources_failed", []) + ["finra_ats"]
        return
    store.upsert_df(conn, "ats_weekly", df)
    store.mark_refreshed(conn, "ats_weekly")


def _refresh_13f(conn, cfg: Config, summary: dict, tickers: list[str]) -> None:
    if cfg.demo_mode:
        for ticker in tickers:
            store.upsert_df(conn, "inst_13f", demo.demo_13f(ticker, cfg.managers_13f))
        store.mark_refreshed(conn, "inst_13f")
        return
    if not cfg.managers_13f:
        return
    # Matched by config/watchlist.yaml's issuer_aliases, not by `tickers` —
    # a ticker added via the UI without a corresponding alias entry simply
    # won't get 13F data, same as any ticker missing an alias today.
    df = sec_13f.fetch_13f_holdings(
        cfg.managers_13f, cfg.issuer_aliases, cfg.sec_user_agent
    )
    if df.empty:
        summary["sources_failed"] = summary.get("sources_failed", []) + ["sec_13f"]
        return
    store.upsert_df(conn, "inst_13f", df)
    store.mark_refreshed(conn, "inst_13f")


def _recompute_metrics(conn, cfg: Config, summary: dict, price_frames, tickers: list[str]) -> None:
    for ticker in tickers:
        pf = price_frames.get(ticker)
        if pf is None or pf.empty:
            continue
        scores = composite_whale_score(
            prices=pf,
            short_volume=store.load_short_volume(conn, ticker),
            holdings_13f=store.load_13f(conn, ticker),
            weights=cfg.whale_weights,
            baseline_window=cfg.baseline_window,
            zscore_threshold=cfg.volume_zscore_threshold,
            flow_window=cfg.flow_window,
        )
        component_cols = [
            c for c in ("big_money_volume", "dark_pool", "inst_13f") if c in scores
        ]
        rows = pd.DataFrame(
            {
                "ticker": ticker,
                "date": scores.index.strftime("%Y-%m-%d"),
                "whale_score": scores["whale_score"].round(2),
                "retail_score": scores["retail_score"].round(2),
                "components": [
                    json.dumps({c: (None if pd.isna(r[c]) else round(float(r[c]), 2)) for c in component_cols})
                    for _, r in scores.iterrows()
                ],
            }
        )
        store.upsert_df(conn, "metrics", rows)
        summary["tickers"].setdefault(ticker, []).append(
            f"whale score: {scores['whale_score'].iloc[-1]:.1f} "
            f"(components: {', '.join(component_cols)})"
        )
    store.mark_refreshed(conn, "metrics")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    if cfg.demo_mode:
        print("DEMO MODE — generating synthetic data (unset WHALETRADING_DEMO for live).")
    summary = refresh_all(cfg)
    failed = summary.get("sources_failed", [])
    for ticker, notes in summary["tickers"].items():
        print(f"{ticker:6s} {' | '.join(notes)}")
    if failed:
        print(f"\nWARNING — sources unavailable this run: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
