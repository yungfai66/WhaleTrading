"""Regression test for the FINRA short-volume incremental-cache bug.

Bug: _refresh_short_volume used to compute its "already fetched" date set
from ANY ticker's rows (a plain `SELECT DISTINCT date FROM short_volume`).
Once one ticker set's refresh populated a date range, every later refresh
for a *different*, disjoint ticker set saw those dates as already covered
and silently skipped fetching them -- so newly-added tickers never got any
short_volume rows for the whole cached window, with no error or warning.

Fix: a date only counts as cached once EVERY currently-requested ticker
already has a row for it.
"""

from __future__ import annotations

import pandas as pd

from whaletrading import pipeline
from whaletrading.config import Config
from whaletrading.data import store


def _fake_fetch_range(tickers, start, end=None, skip_dates=None):
    """Stand-in for finra_short_volume.fetch_range: returns rows for
    `tickers` on every date in the fixed 3-day window not in skip_dates,
    mimicking a real FINRA response without any network access."""
    all_dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    skip_dates = skip_dates or set()
    wanted_dates = [d for d in all_dates if d not in skip_dates]
    if not wanted_dates:
        return pd.DataFrame()
    rows = [
        {"ticker": t, "date": d, "short_volume": 50, "short_exempt_volume": 1, "total_volume": 500}
        for t in tickers
        for d in wanted_dates
    ]
    return pd.DataFrame(rows)


def test_new_ticker_gets_short_volume_even_when_dates_already_cached_for_others(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = store.connect(db_path)

    # Simulate a prior refresh that only ever covered AAPL for these dates.
    store.upsert_df(
        conn,
        "short_volume",
        pd.DataFrame(
            {
                "ticker": ["AAPL"] * 3,
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "short_volume": [100, 200, 300],
                "short_exempt_volume": [1, 2, 3],
                "total_volume": [1000, 2000, 3000],
            }
        ),
    )

    calls = []

    def tracking_fetch_range(tickers, start, end=None, skip_dates=None):
        calls.append(set(skip_dates or set()))
        return _fake_fetch_range(tickers, start, end, skip_dates)

    monkeypatch.setattr(pipeline.finra_short_volume, "fetch_range", tracking_fetch_range)

    cfg = Config(watchlists={"test": ["MSFT"]}, default_watchlist="test", finra_short_volume_days=10)
    summary = {}

    # MSFT is brand new and disjoint from the AAPL-only prior refresh --
    # under the bug, every one of these 3 dates would be in cached_dates
    # (because AAPL has them) and MSFT would get nothing.
    pipeline._refresh_short_volume(conn, cfg, summary, price_frames={}, tickers=["MSFT"])

    msft_rows = store.load_short_volume(conn, "MSFT")
    assert not msft_rows.empty, "MSFT should get short_volume rows on its first refresh"
    assert len(msft_rows) == 3
    assert calls == [set()], "nothing should be skipped for a ticker with zero existing coverage"

    conn.close()


def test_fully_covered_dates_are_skipped_on_repeat_refresh(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = store.connect(db_path)

    calls = []

    def tracking_fetch_range(tickers, start, end=None, skip_dates=None):
        calls.append(set(skip_dates or set()))
        return _fake_fetch_range(tickers, start, end, skip_dates)

    monkeypatch.setattr(pipeline.finra_short_volume, "fetch_range", tracking_fetch_range)

    cfg = Config(watchlists={"test": ["MSFT"]}, default_watchlist="test", finra_short_volume_days=10)

    # First refresh populates MSFT for all 3 dates.
    pipeline._refresh_short_volume(conn, cfg, {}, price_frames={}, tickers=["MSFT"])
    # Second refresh for the SAME ticker should now skip all 3 dates --
    # confirms the fix doesn't just always re-fetch everything.
    pipeline._refresh_short_volume(conn, cfg, {}, price_frames={}, tickers=["MSFT"])

    assert calls[0] == set()
    assert calls[1] == {"2024-01-02", "2024-01-03", "2024-01-04"}

    conn.close()
