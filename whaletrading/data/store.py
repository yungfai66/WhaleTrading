"""SQLite cache so refreshes are incremental and the dashboard loads fast."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR

DB_PATH = DATA_DIR / "whaletrading.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS short_volume (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    short_volume INTEGER, short_exempt_volume INTEGER, total_volume INTEGER,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS ats_weekly (
    ticker TEXT NOT NULL,
    week_start TEXT NOT NULL,
    total_shares INTEGER, total_trades INTEGER,
    PRIMARY KEY (ticker, week_start)
);
CREATE TABLE IF NOT EXISTS inst_13f (
    ticker TEXT NOT NULL,
    report_period TEXT NOT NULL,   -- quarter end, YYYY-MM-DD
    manager_cik INTEGER NOT NULL,
    manager_name TEXT,
    shares INTEGER,
    value_usd INTEGER,
    PRIMARY KEY (ticker, report_period, manager_cik)
);
CREATE TABLE IF NOT EXISTS metrics (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    whale_score REAL, retail_score REAL,
    components TEXT,               -- JSON breakdown of the composite
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS sentiment (
    date TEXT NOT NULL,
    indicator TEXT NOT NULL,   -- 'composite' | 'momentum' | 'volatility' | 'strength' | 'breadth' | 'safe_haven' | 'junk_bond'
    score REAL,                -- 0-100, 50 = neutral
    raw REAL,                  -- underlying signal value (None for 'composite')
    PRIMARY KEY (date, indicator)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def upsert_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """Insert-or-replace all rows of `df` (columns must match the table)."""
    if df is None or df.empty:
        return 0
    cols = list(df.columns)
    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    rows = [tuple(None if pd.isna(v) else v for v in row) for row in df.itertuples(index=False)]
    with conn:
        conn.executemany(sql, rows)
    return len(rows)


def read_df(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def load_prices(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = read_df(
        conn,
        "SELECT date, open, high, low, close, volume FROM prices WHERE ticker=? ORDER BY date",
        (ticker,),
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def load_short_volume(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = read_df(
        conn,
        "SELECT date, short_volume, short_exempt_volume, total_volume "
        "FROM short_volume WHERE ticker=? ORDER BY date",
        (ticker,),
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def load_13f(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    return read_df(
        conn,
        "SELECT report_period, manager_cik, manager_name, shares, value_usd "
        "FROM inst_13f WHERE ticker=? ORDER BY report_period",
        (ticker,),
    )


def load_metrics(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = read_df(
        conn,
        "SELECT date, whale_score, retail_score, components FROM metrics WHERE ticker=? ORDER BY date",
        (ticker,),
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def load_sentiment(conn: sqlite3.Connection) -> pd.DataFrame:
    """Fear & Greed history, wide: date index, one 'score' column per
    indicator plus 'composite', e.g. df['composite'], df['momentum']."""
    df = read_df(conn, "SELECT date, indicator, score FROM sentiment ORDER BY date")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="indicator", values="score")


def load_sentiment_raw(conn: sqlite3.Connection) -> pd.DataFrame:
    """Same shape as load_sentiment but the underlying raw signal values —
    used as the small subtitle on each indicator card."""
    df = read_df(conn, "SELECT date, indicator, raw FROM sentiment ORDER BY date")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="indicator", values="raw")


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )


def get_meta(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def mark_refreshed(conn: sqlite3.Connection, source: str) -> None:
    set_meta(conn, f"last_refresh:{source}", datetime.now(timezone.utc).isoformat())


def source_freshness(conn: sqlite3.Connection) -> dict[str, str | None]:
    """Latest data point per source (YYYY-MM-DD or None if the table is empty).

    This is the *data date*, not the fetch time — e.g. a 13F row dated
    2026-03-31 was filed up to 45 days later and fetched later still.
    """
    queries = {
        "prices": "SELECT MAX(date) FROM prices",
        "short_volume": "SELECT MAX(date) FROM short_volume",
        "ats_weekly": "SELECT MAX(week_start) FROM ats_weekly",
        "inst_13f": "SELECT MAX(report_period) FROM inst_13f",
        "sentiment": "SELECT MAX(date) FROM sentiment WHERE indicator='composite'",
    }
    return {name: conn.execute(sql).fetchone()[0] for name, sql in queries.items()}
