"""Pull a pre-fetched data snapshot from the repo's `data-cache` branch
(published weekdays by .github/workflows/prefetch.yml) instead of Streamlit
Cloud live-fetching FINRA/EDGAR/Yahoo on a cold container. Same fails-open
contract as gist_store: any error here just means the caller falls back to
its existing bootstrap/live-fetch path, never crashes.

Only downloads when the remote snapshot is *newer* than whatever's already
loaded locally, so a manual "Refresh data" click during a session isn't
clobbered by a now-stale-by-comparison snapshot later in the same session --
mirrors the same last_refresh:pipeline:<watchlist> meta key the app's own
manual refresh already writes (see app.py's _mark_watchlist_refreshed).
"""

from __future__ import annotations

import logging
from datetime import datetime

import requests

from ..config import DATA_DIR

log = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/yungfai66/WhaleTrading/data-cache"
META_URL = f"{RAW_BASE}/data/snapshot_meta.json"
DB_URL = f"{RAW_BASE}/data/whaletrading.db"
TIMEOUT = 20


def sync_if_newer(watchlist_name: str, local_last_refresh: str | None) -> bool:
    """Download the data-cache branch's db over the local one if its
    snapshot_meta.json shows a newer refresh for `watchlist_name` than
    `local_last_refresh` (an ISO timestamp, or None if nothing local yet).

    Returns True if it downloaded a fresher snapshot -- caller must reopen
    any open DB connection and clear its own read caches (st.cache_data)
    afterward, since the underlying file just changed under it.
    """
    try:
        meta = requests.get(META_URL, timeout=TIMEOUT)
        meta.raise_for_status()
        remote_iso = meta.json().get(watchlist_name)
        if not remote_iso:
            return False

        if local_last_refresh:
            remote_dt = datetime.fromisoformat(remote_iso)
            local_dt = datetime.fromisoformat(local_last_refresh)
            if remote_dt <= local_dt:
                return False

        db = requests.get(DB_URL, timeout=TIMEOUT)
        db.raise_for_status()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "whaletrading.db").write_bytes(db.content)
        return True
    except Exception as exc:
        log.warning("snapshot_sync: failed: %s", exc)
        return False
