"""FINRA Reg SHO daily short-sale volume files (free, no key, no login).

One pipe-delimited file per trading day, published after the close:
    https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt
Columns: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

This covers consolidated off-exchange (dark pool / internalizer) volume — the
raw material for the dark-pool pressure component of the whale score.
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

URL_TEMPLATE = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"


def fetch_day(day: date, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch one day's file. Empty frame for weekends/holidays (404) or errors."""
    url = URL_TEMPLATE.format(yyyymmdd=day.strftime("%Y%m%d"))
    sess = session or requests
    try:
        resp = sess.get(url, timeout=30)
    except requests.RequestException as exc:
        log.warning("FINRA short-volume fetch failed for %s: %s", day, exc)
        return pd.DataFrame()
    if resp.status_code == 404:  # non-trading day
        return pd.DataFrame()
    if resp.status_code != 200:
        log.warning("FINRA short-volume HTTP %s for %s", resp.status_code, day)
        return pd.DataFrame()
    return parse_file(resp.text)


def parse_file(text: str) -> pd.DataFrame:
    """Parse a Reg SHO daily file into columns matching the short_volume table."""
    try:
        df = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    except Exception as exc:
        log.warning("could not parse FINRA file: %s", exc)
        return pd.DataFrame()

    df.columns = [c.strip().lower() for c in df.columns]
    required = {"date", "symbol", "shortvolume", "totalvolume"}
    if not required.issubset(df.columns):
        log.warning("FINRA file missing expected columns, got %s", list(df.columns))
        return pd.DataFrame()

    # The last line is a "trailer" record count — drop rows with unparseable dates.
    df = df[df["date"].astype(str).str.fullmatch(r"\d{8}", na=False)].copy()
    out = pd.DataFrame(
        {
            "ticker": df["symbol"].astype(str).str.strip().str.upper(),
            "date": pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d"),
            "short_volume": pd.to_numeric(df["shortvolume"], errors="coerce"),
            "short_exempt_volume": pd.to_numeric(
                df.get("shortexemptvolume"), errors="coerce"
            ),
            "total_volume": pd.to_numeric(df["totalvolume"], errors="coerce"),
        }
    )
    return out.dropna(subset=["short_volume", "total_volume"])


def fetch_range(
    tickers: list[str],
    start: date,
    end: date | None = None,
    skip_dates: set[str] | None = None,
) -> pd.DataFrame:
    """Fetch files from `start` to `end`, filtered to `tickers`.

    `skip_dates` (YYYY-MM-DD strings) lets the pipeline skip days already cached.
    """
    end = end or date.today()
    wanted = {t.upper() for t in tickers}
    skip = skip_dates or set()
    session = requests.Session()
    frames = []
    day = start
    while day <= end:
        if day.weekday() < 5 and day.isoformat() not in skip:
            df = fetch_day(day, session)
            if not df.empty:
                frames.append(df[df["ticker"].isin(wanted)])
        day += timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
