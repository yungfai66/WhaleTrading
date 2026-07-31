"""Regression tests for whaletrading.data.snapshot_sync's freshness gate --
the logic deciding whether to download the GitHub Actions-published
snapshot (data-cache branch) over whatever's already loaded locally.
"""

from __future__ import annotations

from whaletrading.data import snapshot_sync


class _FakeResponse:
    def __init__(self, json_data=None, content=b""):
        self._json = json_data
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def test_downloads_when_no_local_data(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot_sync, "DATA_DIR", tmp_path)
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        if url == snapshot_sync.META_URL:
            return _FakeResponse(json_data={"US Bought": "2026-08-03T05:00:00+00:00"})
        return _FakeResponse(content=b"fake-db-bytes")

    monkeypatch.setattr(snapshot_sync.requests, "get", fake_get)

    assert snapshot_sync.sync_if_newer("US Bought", None) is True
    assert (tmp_path / "whaletrading.db").read_bytes() == b"fake-db-bytes"
    assert calls == [snapshot_sync.META_URL, snapshot_sync.DB_URL]


def test_skips_download_when_remote_is_not_newer(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot_sync, "DATA_DIR", tmp_path)
    db_calls = []

    def fake_get(url, timeout):
        if url == snapshot_sync.META_URL:
            return _FakeResponse(json_data={"US Bought": "2026-08-03T05:00:00+00:00"})
        db_calls.append(url)
        return _FakeResponse(content=b"should-not-be-downloaded")

    monkeypatch.setattr(snapshot_sync.requests, "get", fake_get)

    # Local is already as fresh as (or fresher than) the remote snapshot.
    result = snapshot_sync.sync_if_newer("US Bought", "2026-08-03T09:00:00+00:00")

    assert result is False
    assert db_calls == []
    assert not (tmp_path / "whaletrading.db").exists()


def test_no_entry_for_watchlist_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot_sync, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        snapshot_sync.requests, "get", lambda url, timeout: _FakeResponse(json_data={"US Bought": "2026-08-03T05:00:00+00:00"})
    )

    assert snapshot_sync.sync_if_newer("Special Watchlist", None) is False


def test_fails_open_on_network_error(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot_sync, "DATA_DIR", tmp_path)

    def raise_error(url, timeout):
        raise ConnectionError("no network")

    monkeypatch.setattr(snapshot_sync.requests, "get", raise_error)

    assert snapshot_sync.sync_if_newer("US Bought", None) is False
