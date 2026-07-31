"""Regression tests for the scheduled prefetch entry point (whaletrading.prefetch).

Covers what a cron run depends on: with no argument it refreshes every
watchlist's tickers in one deduped pass (a ticker shared by several lists is
only fetched once) and records a last-refresh timestamp for *each* of them;
with a watchlist name it refreshes just that one. Either way the timestamp is
recorded the same way the app's manual "Refresh data" button does -- so the
sidebar's "Last refresh" caption reflects a scheduled run too, not just a
manual click.
"""

from __future__ import annotations

from whaletrading import prefetch
from whaletrading.config import Config
from whaletrading.data import store


def test_prefetch_refreshes_the_named_watchlist(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(store, "DB_PATH", db_path)

    cfg = Config(
        watchlists={"Special Watchlist": ["AAPL"], "US Bought": ["MSFT", "TSLA"]},
        default_watchlist="Special Watchlist",
    )
    monkeypatch.setattr(prefetch, "load_config", lambda: cfg)

    calls = []

    def fake_refresh_all(cfg_arg, tickers):
        calls.append(list(tickers))
        return {"tickers": {t: [] for t in tickers}}

    monkeypatch.setattr(prefetch, "refresh_all", fake_refresh_all)
    monkeypatch.setattr(prefetch.sys, "argv", ["prefetch", "US Bought"])

    assert prefetch.main() == 0
    assert calls == [["MSFT", "TSLA"]]

    conn = store.connect(db_path)
    assert store.get_meta(conn, "last_refresh:pipeline:US Bought") is not None
    # Only the requested watchlist is touched -- not every list in config.
    assert store.get_meta(conn, "last_refresh:pipeline:Special Watchlist") is None
    conn.close()


def test_prefetch_with_no_argument_refreshes_every_watchlist_deduped(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(store, "DB_PATH", db_path)

    cfg = Config(
        watchlists={
            "Special Watchlist": ["AAPL", "MSFT"],
            "US Bought": ["MSFT", "TSLA"],
        },
        default_watchlist="US Bought",
    )
    monkeypatch.setattr(prefetch, "load_config", lambda: cfg)

    calls = []
    monkeypatch.setattr(
        prefetch, "refresh_all", lambda cfg_arg, tickers: calls.append(list(tickers)) or {"tickers": {}}
    )
    monkeypatch.setattr(prefetch.sys, "argv", ["prefetch"])

    assert prefetch.main() == 0
    # AAPL, MSFT, TSLA fetched once each, not MSFT twice for appearing in
    # both lists.
    assert calls == [["AAPL", "MSFT", "TSLA"]]

    conn = store.connect(db_path)
    assert store.get_meta(conn, "last_refresh:pipeline:Special Watchlist") is not None
    assert store.get_meta(conn, "last_refresh:pipeline:US Bought") is not None
    conn.close()


def test_prefetch_rejects_unknown_watchlist(tmp_path, monkeypatch, capsys):
    cfg = Config(watchlists={"US Bought": ["MSFT"]}, default_watchlist="US Bought")
    monkeypatch.setattr(prefetch, "load_config", lambda: cfg)
    monkeypatch.setattr(prefetch.sys, "argv", ["prefetch", "Nonexistent List"])

    called = []
    monkeypatch.setattr(prefetch, "refresh_all", lambda *a, **k: called.append(1))

    assert prefetch.main() == 1
    assert not called
    assert "Unknown watchlist" in capsys.readouterr().out
