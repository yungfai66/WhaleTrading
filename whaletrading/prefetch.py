"""Scheduled prefetch: refresh every watchlist ahead of anyone opening the app.

The app itself no longer auto-refreshes on page load (see app.py) -- only a
scheduled run of this script and the "Refresh data" button in the sidebar
touch the network. Wired to a GitHub Actions cron job
(.github/workflows/prefetch.yml), weekdays ~05:00 SGT (shortly after the US
regular session closes), which publishes the resulting data/whaletrading.db
and data/snapshot_meta.json to the repo's `data-cache` branch --
whaletrading.data.snapshot_sync downloads that from the Streamlit Cloud side.

With no argument, every watchlist in config/watchlist.yaml is refreshed in a
single pass over the deduped union of their tickers (cfg.all_tickers) -- a
ticker shared by several lists (e.g. TSLA) is only fetched once, not once per
list. Pass a specific watchlist name to refresh just that one instead (handy
for manual/local testing).

Run:  python -m whaletrading.prefetch                # every watchlist
      python -m whaletrading.prefetch "US Bought"     # a specific watchlist
"""

from __future__ import annotations

import json
import logging
import sys

from .config import DATA_DIR, load_config
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
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is not None:
        if arg not in cfg.watchlists:
            print(f"Unknown watchlist: {arg!r} (have: {', '.join(cfg.watchlists)})")
            return 1
        names = [arg]
        tickers = cfg.watchlists[arg]
    else:
        names = list(cfg.watchlists)
        tickers = cfg.all_tickers

    summary = refresh_all(cfg, tickers=tickers)

    # Companion file for snapshot_sync.py: lets the Cloud app decide whether
    # to download the (much larger) db without fetching it first. Keyed by
    # watchlist name -- every name refreshed this run shares the same
    # refreshed_at, since they were all pulled from the one combined fetch
    # above; existing entries for lists not covered by this run are kept.
    conn = store.connect()
    for name in names:
        store.mark_refreshed(conn, f"pipeline:{name}")
    meta_path = DATA_DIR / "snapshot_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for name in names:
        meta[name] = store.get_meta(conn, f"last_refresh:pipeline:{name}")
    conn.close()
    meta_path.write_text(json.dumps(meta, indent=2))

    for ticker, notes in summary["tickers"].items():
        print(f"{ticker:6s} {' | '.join(notes)}")
    failed = summary.get("sources_failed", [])
    if failed:
        print(f"\nWARNING — sources unavailable this run: {', '.join(failed)}")
    print(f"\nPrefetched {len(tickers)} tickers across {len(names)} watchlist(s): {', '.join(names)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
