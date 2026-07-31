"""Scheduled prefetch: refresh one watchlist ahead of anyone opening the app.

The app itself no longer auto-refreshes on page load (see app.py) -- only
this scheduled run and the "Refresh data" button in the sidebar touch the
network. Wired to Windows Task Scheduler, weekdays ~05:00 SGT (shortly after
the US regular session closes) -- this machine's local time is already SGT,
so a plain 05:00 weekday trigger needs no timezone conversion.

Run:  python -m whaletrading.prefetch                # config's default_watchlist
      python -m whaletrading.prefetch "US Bought"     # a specific watchlist
"""

from __future__ import annotations

import logging
import sys

from .config import load_config
from .data import store
from .pipeline import refresh_all

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cfg = load_config()
    name = sys.argv[1] if len(sys.argv) > 1 else cfg.default_watchlist
    if name not in cfg.watchlists:
        print(f"Unknown watchlist: {name!r} (have: {', '.join(cfg.watchlists)})")
        return 1

    tickers = cfg.watchlists[name]
    summary = refresh_all(cfg, tickers=tickers)

    conn = store.connect()
    store.mark_refreshed(conn, f"pipeline:{name}")
    conn.close()

    for ticker, notes in summary["tickers"].items():
        print(f"{ticker:6s} {' | '.join(notes)}")
    failed = summary.get("sources_failed", [])
    if failed:
        print(f"\nWARNING — sources unavailable this run: {', '.join(failed)}")
    print(f"\nPrefetched {len(tickers)} tickers for {name!r}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
