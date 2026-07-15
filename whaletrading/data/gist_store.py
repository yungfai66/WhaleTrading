"""Optional cross-device sync for the session watchlist (pin/order) via a
private GitHub Gist. Free (personal GitHub API use), opt-in — the app works
exactly as before if the two secrets below aren't set.

Setup (see README): create a private Gist with one file
`watchlist_state.json` containing `{}`, note its ID, and a GitHub Personal
Access Token scoped to `gist` only. Set GITHUB_GIST_TOKEN / GITHUB_GIST_ID
as Streamlit secrets.
"""

from __future__ import annotations

import json
import logging

import requests

log = logging.getLogger(__name__)

GIST_API = "https://api.github.com/gists"
GIST_FILENAME = "watchlist_state.json"
TIMEOUT = 10


def load_watchlist_state(token: str, gist_id: str) -> dict | None:
    """{"watchlist_order": [...], "pinned_tickers": [...]}, or None on any
    failure (missing/invalid creds, network error, malformed content, empty
    file) — callers must fall back to config/watchlist.yaml, never crash."""
    try:
        resp = requests.get(
            f"{GIST_API}/{gist_id}",
            headers=_headers(token),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        files = resp.json().get("files", {})
        content = files.get(GIST_FILENAME, {}).get("content")
        if not content:
            return None
        data = json.loads(content)
        if not isinstance(data, dict) or "watchlist_order" not in data:
            return None
        return data
    except Exception as exc:
        log.warning("gist_store: load failed: %s", exc)
        return None


def save_watchlist_state(token: str, gist_id: str, order: list[str], pinned: list[str]) -> bool:
    """Overwrite the gist's watchlist_state.json. Returns False on any
    failure — caller shows a non-blocking warning and keeps going with
    session state as the source of truth for the rest of that session."""
    payload = {
        "files": {
            GIST_FILENAME: {
                "content": json.dumps(
                    {"watchlist_order": order, "pinned_tickers": sorted(pinned)}, indent=2
                )
            }
        }
    }
    try:
        resp = requests.patch(
            f"{GIST_API}/{gist_id}",
            headers=_headers(token),
            json=payload,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        log.warning("gist_store: save failed: %s", exc)
        return False


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
