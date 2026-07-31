"""WhaleTrading dashboard.

Run:  streamlit run app.py
Demo: WHALETRADING_DEMO=1 streamlit run app.py   (synthetic data, no network)
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_sortables import sort_items

from whaletrading import signals
from whaletrading.config import load_config
from whaletrading.data import gist_store
from whaletrading.data import prices as prices_mod
from whaletrading.data import store
from whaletrading.indicators import fear_greed
from whaletrading.pipeline import refresh_all

# Streamlit Cloud secrets don't auto-populate os.environ — bridge the ones we
# read that way. No-op locally (no secrets.toml) or off Cloud.
for _key in ("WHALETRADING_DEMO", "GITHUB_GIST_TOKEN", "GITHUB_GIST_ID"):
    try:
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
    except Exception:
        pass


def gist_configured() -> tuple[str, str] | None:
    """(token, gist_id) if cross-device watchlist sync is set up, else None —
    optional feature, app works session-only without it (see README)."""
    token = os.environ.get("GITHUB_GIST_TOKEN")
    gist_id = os.environ.get("GITHUB_GIST_ID")
    return (token, gist_id) if token and gist_id else None

# Validated palette (dataviz reference, light mode).
C = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "whale": "#e34948",     # red bars = whale accumulation (strategy convention)
    "retail": "#008300",    # green bars = retail accumulation
    "up": "#1baf7a",
    "down": "#e34948",
    "blue": "#2a78d6",      # bearish ("blue") ribbon / MACD line
    "orange": "#eb6834",    # MACD signal line
    "violet": "#4a3aa7",
    "price_line": "#a91b1b",  # close-price line — darker than candle red so it doesn't blend into down candles
    "buy": "#0ca30c",
    "sell": "#d03b3b",
    "neutral": "#c3c2b7",
}
RIBBON_STEPS = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"]

TIMEFRAMES = {"Daily": "D", "Weekly": "W", "Monthly": "M"}

# Data-source metadata for the freshness/validity panel: display name,
# inherent publication delay, and how old (calendar days) the newest row may
# be before we flag the source as stale.
SOURCE_INFO = {
    "prices": {
        "name": "Prices & volume (Yahoo Finance)",
        "delay": "End of day — daily bars finalize after the US close; intraday quotes are ~15 min delayed",
        "stale_after": 5,
    },
    "short_volume": {
        "name": "Dark-pool short volume (FINRA daily)",
        "delay": "Same evening (~6pm ET) — reflects that day's off-exchange trading",
        "stale_after": 5,
    },
    "ats_weekly": {
        "name": "Dark-pool ATS volume (FINRA weekly)",
        "delay": "2-week publication delay (4 weeks for smaller stocks)",
        "stale_after": 35,
    },
    "inst_13f": {
        "name": "Institutional holdings (SEC 13F)",
        "delay": "Quarterly; filed up to 45 days after quarter end — positions are 45–135 days old",
        "stale_after": 140,
    },
    "sentiment": {
        "name": "Market Fear & Greed inputs (Yahoo Finance)",
        "delay": "End of day — same cadence as prices",
        "stale_after": 5,
    },
}

# Date-range presets for the ticker detail chart: label -> calendar days back
# (None = show everything). Default preset differs by bar granularity so a
# fresh page load looks like a reasonable default window.
RANGE_PRESETS = {
    "1M": 30, "3M": 91, "6M": 182, "1Y": 365, "2Y": 730, "5Y": 1826, "All": None,
}
DEFAULT_RANGE = {"D": "1Y", "W": "2Y", "M": "All"}

# Left-panel nav: key -> button label. Order here is the order they render in.
# Labels are kept short (full names, if any, live in the page itself) so
# they never wrap inside the nav button, even with the sidebar dragged
# narrower than its default width.
PAGES = {
    "overview": "📊 Watchlist",
    "feargreed": "😱 Fear & Greed",
    "detail": "📈 Ticker detail",
    "guide": "📖 How to read this",
}

ZONE_RANK = {"weak": 0, "momentum": 1, "rise": 2, "soar": 3}

# Compact legend shown above the watchlist table — full explanations live in
# the hover tooltip (title=) rather than as printed text, to save space.
ACTION_GUIDE = [
    ("🟢 Buy", "This IS a buy signal — consider buying gradually (DCA, Dollar-Cost Averaging) rather than all at once."),
    ("🟠 Trim", "This IS a sell signal — big investors look like they're selling, consider taking some profit."),
    ("🟡 Watch", "No signal yet, but big-investor buying is rising and conditions may be building toward a buy signal."),
    ("🔵 Hold", "No signal, price trend still looks positive."),
    ("⚪ Wait", "No signal, nothing stands out right now."),
]


def get_watchlist_state(cfg) -> dict[str, dict]:
    """Every watchlist's order/pinned, keyed by name — the full multi-
    watchlist state. Seeded once per session from a synced GitHub Gist when
    cross-device sync is configured (see gist_configured / README),
    otherwise from config/watchlist.yaml — session-scoped only and resets
    on page reload without Gist sync, since the app's filesystem is
    ephemeral on Streamlit Cloud and silently rewriting
    config/watchlist.yaml would look like it worked locally but lose the
    change on every redeploy there.

    Only watchlist *names* present in the current config are kept — a name
    that only exists in an older synced Gist (e.g. after a config rename)
    is dropped rather than surfaced as a phantom, unselectable entry.
    """
    if "watchlists" not in st.session_state:
        synced = None
        creds = gist_configured()
        if creds:
            synced = gist_store.load_state(*creds)
        synced_lists = (synced or {}).get("watchlists") or {}
        st.session_state["watchlists"] = {
            name: {
                "order": list(synced_lists[name]["order"]) if name in synced_lists else list(tickers),
                "pinned": set(synced_lists[name].get("pinned", [])) if name in synced_lists else set(),
            }
            for name, tickers in cfg.watchlists.items()
        }
        synced_active = (synced or {}).get("active_watchlist")
        st.session_state["active_watchlist"] = (
            synced_active if synced_active in cfg.watchlists else cfg.default_watchlist
        )
    if st.session_state.get("active_watchlist") not in st.session_state["watchlists"]:
        st.session_state["active_watchlist"] = cfg.default_watchlist
    return st.session_state["watchlists"]


def get_working_watchlist(cfg) -> list[str]:
    """The ticker list actually shown for the *active* watchlist, after
    pins/adds/removes/reordering."""
    state = get_watchlist_state(cfg)
    return state[st.session_state["active_watchlist"]]["order"]


def get_active_pinned(cfg) -> set[str]:
    state = get_watchlist_state(cfg)
    return state[st.session_state["active_watchlist"]]["pinned"]


def sync_watchlist_state() -> None:
    """Push every watchlist's order/pins, plus which one is active, to the
    Gist when sync is configured. Fails open — a sync error never blocks
    the pin/add/remove/reorder/paste-import action itself, it just means
    this particular change won't show up on other devices until the next
    successful save."""
    creds = gist_configured()
    if not creds:
        return
    payload = {
        name: {"order": v["order"], "pinned": sorted(v["pinned"])}
        for name, v in st.session_state["watchlists"].items()
    }
    if not gist_store.save_state(*creds, st.session_state["active_watchlist"], payload):
        st.warning("Couldn't sync to your other devices right now — this change is still saved for this session.")


def _table_height(n_rows: int) -> int:
    """Size the dataframe container so every row is visible with no internal
    scrollbar — the page scrolls instead, never a scrollbar nested inside a
    scrollbar. Deliberately generous (not an exact-fit calculation): actual
    row height depends on the viewer's font size/zoom/OS text scaling, which
    we can't know in advance, so this overestimates. A little empty space
    at the bottom of the table is fine; a stray inner scrollbar is not."""
    return 150 + 46 * max(n_rows, 1)


def _format_singapore(iso_ts: str | None) -> str:
    """Format a stored UTC ISO timestamp (see store.mark_refreshed) in
    Singapore time, since that's the viewer's timezone."""
    if not iso_ts:
        return "never"
    dt = datetime.fromisoformat(iso_ts).astimezone(ZoneInfo("Asia/Singapore"))
    return dt.strftime("%Y-%m-%d %H:%M") + " SGT"


def _mark_watchlist_refreshed(watchlist_name: str) -> None:
    """Records a per-watchlist last-refresh timestamp (distinct from
    pipeline.py's own global "last_refresh:pipeline", which just means "a
    refresh ran for *some* ticker set") so the sidebar caption reflects
    whichever watchlist is actually active, not whichever was refreshed
    most recently overall."""
    conn = store.connect()
    store.mark_refreshed(conn, f"pipeline:{watchlist_name}")
    conn.close()


st.set_page_config(page_title="WhaleTrading", page_icon="🐋", layout="wide")

# Sidebar gets a minimum width wide enough for its longest nav label
# ("😱 Fear & Greed") so it never wraps at the default drag position —
# still user-resizable, this only sets the floor. Also trims Streamlit's
# large default top padding and divider margins, which otherwise stack up
# into a noticeable gap between the title and the "Watchlist" section below
# the nav row.
st.markdown(
    """
    <style>
    /* Streamlit's fixed header is ~60px (3.75rem) tall with a high z-index —
       padding-top must clear it or content underneath gets visually hidden
       behind it (looks broken: e.g. a button's colored fill shows but its
       text doesn't, since the text sits under the opaque header layer). */
    .block-container { padding-top: 4rem !important; padding-bottom: 1.5rem !important; }
    hr { margin: 0.4rem 0 !important; }
    [data-testid="stAlert"] { padding: 0.5rem 0.9rem !important; margin-bottom: 0.4rem !important; }
    [data-testid="stExpander"] { margin-bottom: 0.4rem !important; }
    [data-testid="stCaptionContainer"] p { margin-bottom: 0 !important; }
    div[data-testid="stMarkdownContainer"] > p { margin-bottom: 0.3rem !important; }
    h1, h2, h3 { margin-top: 0.2rem !important; margin-bottom: 0.3rem !important; }
    div[data-testid="stRadio"] > label { margin-bottom: 0.1rem !important; }
    div[data-testid="stRadio"] div[role="radiogroup"] { gap: 0.4rem !important; }

    /* Taller tap target, less horizontal padding stealing label width — the
       watchlist-table and sidebar-nav rules below tighten padding further
       still, this is just the app-wide floor. */
    .stButton button { padding: 0.3rem 0.6rem !important; line-height: 1.2 !important; }

    /* Sidebar: nav-button labels never wrap — ellipsis instead — and read
       left-aligned like a nav list rather than centered like an action
       button. min-width keeps the default sidebar width wide enough for
       the longest label; users can still drag it narrower or wider. */
    section[data-testid="stSidebar"] { min-width: 230px; }
    section[data-testid="stSidebar"] .stButton button {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        justify-content: flex-start;
        text-align: left;
        display: block;
    }

    /* Ticker + bar-size controls nested under the "Ticker detail" nav
       button in the left panel — indented so they read as sub-controls. */
    .st-key-ticker_detail_nav { margin-left: 0.9rem; margin-bottom: 0.3rem; }

    /* Watchlist table: bordered cells, minimal row/column spacing. Scoped
       to this one container (Streamlit stamps a stable st-key-<key> class
       via st.container(key=...)) so it doesn't affect other tables/columns
       elsewhere in the app. font-variant-numeric keeps digits a fixed width
       so Close/Whale/Δ20d/Retail line up column-wise instead of drifting.

       Colors below are unconditional, not gated on
       @media(prefers-color-scheme: dark) — that tracks the OS/browser
       theme, not Streamlit's own in-app theme toggle, and this app always
       runs in Streamlit's dark theme regardless of OS setting. The gated
       version silently never applied, so the row/header highlight fell
       back to this near-white default against the dark theme's near-white
       text — unreadable. The hover/header-band tints use color-mix()
       against currentColor so they self-adjust to whatever text color is
       actually active instead of a second hardcoded guess. */
    .st-key-watchlist_table { font-variant-numeric: tabular-nums; }
    .st-key-watchlist_table div[data-testid="stHorizontalBlock"] {
        border-bottom: 1px solid #33363f;
        gap: 0.3rem !important;
    }
    .st-key-watchlist_table div[data-testid="stHorizontalBlock"]:hover {
        background: color-mix(in srgb, currentColor 16%, transparent);
    }
    .st-key-watchlist_table div[data-testid="column"] {
        border-right: 1px solid #33363f;
        padding: 0.05rem 0.4rem !important;
    }
    .st-key-watchlist_table div[data-testid="column"]:last-child { border-right: none; }
    /* Close / Whale / Δ20d / Retail — right-align the plain-text data
       cells (header buttons keep their own centered layout, unaffected by
       a parent text-align). */
    .st-key-watchlist_table div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(6),
    .st-key-watchlist_table div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(7),
    .st-key-watchlist_table div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(8),
    .st-key-watchlist_table div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(9) {
        text-align: right;
    }
    /* Header row: tinted band so it reads as a header, not just another row. */
    .st-key-watchlist_header_row {
        background: color-mix(in srgb, currentColor 8%, transparent);
        border-radius: 4px 4px 0 0;
    }
    /* Header/data buttons: never wrap, ellipsis instead, tight padding, and
       a font size that tracks viewport width — the original bug (a full
       column-width label + sort arrow at a fixed 0.78rem font wrapped to
       two lines and broke row alignment). Background/text/border colors are
       also overridden here: Streamlit's own secondary-button style renders
       a fixed white pill with dark text regardless of the app's dark theme
       — the actual source of the "header button contrast, ugly" complaint,
       separate from (and not fixed by) the row/header-band coloring above. */
    .st-key-watchlist_table .stButton button {
        padding: 0.05rem 0.3rem !important;
        font-size: clamp(0.68rem, 0.78vw, 0.82rem) !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        width: 100%;
        background-color: #2c2f37 !important;
        color: #e8e8ec !important;
        border-color: #454850 !important;
    }
    .st-key-watchlist_table .stButton button:hover {
        background-color: #383c45 !important;
        border-color: #5a5e68 !important;
        color: #ffffff !important;
    }

    /* Watchlist % bar (Whale column) — a real filled track, not the old
       ▓░ ASCII characters, so it renders as a bar at any font/zoom. */
    .wt-bar-track {
        position: relative;
        display: inline-block;
        width: 60px;
        height: 8px;
        background: #33363f;
        border-radius: 4px;
        vertical-align: middle;
        margin-right: 0.4rem;
    }
    .wt-bar-fill {
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        border-radius: 4px;
        background: #e34948;
    }

    /* App/header background: Streamlit's default dark theme (#0e1117)
       reads as harsh near-black. Softer dark grey instead; Streamlit's own
       sidebar background (#262730) is untouched since it isn't part of the
       complaint. Unconditional for the same reason as above — the OS media
       query this used to be gated on never matched this app's actual
       (always-dark) theme. */
    .stApp, [data-testid="stHeader"] { background: #20232a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def bootstrap_data(_cfg) -> bool:
    """Populate the SQLite cache on a fresh container so visitors never see an
    empty dashboard. Runs at most once per container lifecycle (cache_resource).

    Scoped to just the default watchlist, not every watchlist combined —
    with several imported lists totaling 100+ tickers, fetching everything
    on first load would make a cold start far slower than necessary. Other
    watchlists populate on demand, the first time someone switches to one
    and clicks 🔄 Refresh data (or edits it), same as this scoped bootstrap
    covers the default one.
    """
    conn = store.connect()
    has_data = conn.execute("SELECT 1 FROM metrics LIMIT 1").fetchone() is not None
    conn.close()
    if has_data:
        return False
    refresh_all(_cfg, tickers=_cfg.watchlists[_cfg.default_watchlist])
    _mark_watchlist_refreshed(_cfg.default_watchlist)
    return True


# How stale the cache must be (minutes since last pipeline refresh) before a
# new page load/reload triggers an automatic refresh. Session-scoped (not
# st.cache_resource), so it fires once per browser session/reload rather than
# once per container lifetime — without this a re-opened tab could sit on a
# refresh from hours ago until someone remembers to click the button.
AUTO_REFRESH_STALE_MINUTES = 30


def auto_refresh_if_stale(cfg) -> None:
    """Fires at most once per session, scoped to whichever watchlist is
    active at that moment (almost always the default one, at page load).
    Switching to a *different*, never-yet-refreshed watchlist later in the
    same session doesn't re-trigger this — the existing "no data — run
    refresh" row messaging covers that; 🔄 Refresh data handles it in one
    click without a second automatic full-app refresh happening in the
    background."""
    if st.session_state.get("_auto_refresh_done"):
        return
    st.session_state["_auto_refresh_done"] = True
    active_name = st.session_state["active_watchlist"]
    conn = store.connect()
    last = store.get_meta(conn, f"last_refresh:pipeline:{active_name}")
    conn.close()
    if last:
        age_min = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() / 60
        if age_min < AUTO_REFRESH_STALE_MINUTES:
            return
    with st.spinner("Refreshing data…"):
        refresh_all(cfg, tickers=get_working_watchlist(cfg))
        _mark_watchlist_refreshed(active_name)
    st.cache_data.clear()


@st.cache_data(ttl=60)
def load_quote(ticker: str) -> float | None:
    """Short-TTL cache (60s) so this stays meaningfully "current", separate
    from the 5-minute cache on the daily-bar data used for charting."""
    return prices_mod.fetch_quote(ticker)


@st.cache_data(ttl=60 * 60 * 24)
def load_company_name(ticker: str) -> str | None:
    """Company/issuer name, e.g. "NVIDIA Corporation" for NVDA. Long TTL —
    this practically never changes. Best-effort: None on any failure."""
    return prices_mod.company_name(ticker)


@st.cache_data(ttl=300)
def load_latest_daily(ticker: str) -> pd.Series | None:
    """The single most recent completed daily bar, independent of whatever
    bar-size (D/W/M) the user has selected for the chart — fixes "Close"
    looking stale when Weekly/Monthly bars are selected."""
    conn = store.connect()
    daily = store.load_prices(conn, ticker)
    conn.close()
    return daily.iloc[-1] if not daily.empty else None


@st.cache_data(ttl=300)
def load_ticker_frame(ticker: str, timeframe: str) -> pd.DataFrame:
    """OHLCV + whale/retail scores resampled to the timeframe, with signals."""
    conn = store.connect()
    daily = store.load_prices(conn, ticker)
    metrics = store.load_metrics(conn, ticker)
    conn.close()
    if daily.empty or metrics.empty:
        return pd.DataFrame()
    bars = prices_mod.resample(daily, timeframe)
    scores = metrics[["whale_score", "retail_score"]]
    if timeframe != "D":
        rule = {"W": "W-FRI", "M": "ME"}[timeframe]
        scores = scores.resample(rule).mean()
    frame = bars.join(scores, how="left").ffill().dropna(subset=["whale_score"])
    return signals.evaluate(frame)


@st.cache_data(ttl=300)
def load_overview(watchlist: tuple[str, ...]) -> pd.DataFrame:
    conn = store.connect()
    cfg = get_config()
    rows = []
    for ticker in watchlist:
        metrics = store.load_metrics(conn, ticker)
        daily = store.load_prices(conn, ticker)
        if metrics.empty or daily.empty:
            rows.append({"Ticker": ticker, "Status": "no data — run refresh"})
            continue
        weekly = load_ticker_frame(ticker, "W")
        last = metrics.iloc[-1]
        thr = cfg.thresholds_for(ticker)
        act = signals.current_action(weekly, thr)
        company = None if cfg.demo_mode else load_company_name(ticker)

        # act['detail'] already states "A buy/sell signal appeared this week: ..."
        # when the latest bar just fired one — no need to build a separate message.
        signal_icon = "—"
        signal_detail = ""
        if not weekly.empty:
            latest_bar = weekly.iloc[-1]
            if bool(latest_bar["sell_signal"]):
                signal_icon, signal_detail = "🔴", act["detail"]
            elif bool(latest_bar["buy_signal"]):
                signal_icon, signal_detail = "🟢", act["detail"]

        zone = signals.zone_label(float(last["whale_score"]), thr)
        rows.append(
            {
                "Ticker": ticker,
                "Company": company or "",
                "Action": act["label"],
                "Signal": signal_icon,
                "Close": round(float(daily["close"].iloc[-1]), 2),
                "Whale %": round(float(last["whale_score"]), 1),
                "Δ 20d": round(
                    float(
                        metrics["whale_score"].iloc[-1]
                        - metrics["whale_score"].iloc[-21]
                    ),
                    1,
                )
                if len(metrics) > 21
                else None,
                "Retail %": round(float(last["retail_score"]), 1),
                "Zone": zone,
                "Status": "ok",
                "_sort": signals.ACTION_ORDER.index(act["action"]),
                "_signal_detail": signal_detail,
                "_zone_rank": ZONE_RANK[zone],
                "_severity": act["severity"],
            }
        )
    conn.close()
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_sentiment() -> pd.DataFrame:
    """Fear & Greed history: date index, one score column per indicator plus
    'composite'. Empty (no columns) if nothing's been refreshed yet."""
    conn = store.connect()
    df = store.load_sentiment(conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_sentiment_raw() -> pd.DataFrame:
    """Same shape as load_sentiment but the underlying raw signal values —
    shown as each indicator card's subtitle."""
    conn = store.connect()
    df = store.load_sentiment_raw(conn)
    conn.close()
    return df


# Band name -> color, reusing the existing palette so the meaning matches
# the rest of the app: red = bearish/fear (C['sell']/C['down']), green =
# bullish/greed (C['up']/C['buy']), gray = neutral.
FG_BAND_COLORS = {
    "Extreme Fear": C["sell"],
    "Fear": C["down"],
    "Neutral": C["neutral"],
    "Greed": C["up"],
    "Extreme Greed": C["buy"],
}

# Position-based lookback for "prior readings" (trading days back from the
# latest row) — same idiom as the Δ 20d column (metrics.iloc[-21] above).
PRIOR_READINGS = [("Previous close", 1), ("1 week ago", 5), ("1 month ago", 21), ("1 year ago", 252)]


# Shown as a native Plotly hover tooltip on each panel's title (see
# `captureevents`/`hovertext` below) — order matches `subplot_titles`.
PANEL_HELP = (
    "Candlesticks show price.<br>The trend ribbon (EMA lines) shows<br>"
    "trend direction — red/bullish, blue/bearish.<br>"
    "The bold red line is the closing price.<br>"
    "🟢/🔴 triangles mark exactly where<br>a buy/sell signal fired.",
    "MACD tracks momentum.<br>Blue = MACD line, orange = signal line.<br>"
    "Blue crossing above orange = golden cross<br>"
    "(momentum turning up); below = death cross<br>"
    "(turning down). A golden cross only counts<br>"
    "toward a buy signal when whale buying<br>is also rising.",
    "Red bars = estimated whale (big-investor)<br>"
    "buying; green bars = estimated retail buying.<br>"
    "Both 0-100, 50 = neutral.<br>"
    "Dashed lines mark the momentum / rise / soar<br>zone thresholds.",
    "Trading volume per bar, colored to match<br>"
    "the trend ribbon — red = bullish trend,<br>"
    "blue = bearish trend, gray = mixed/neutral.",
)


def four_panel_figure(frame: pd.DataFrame, ticker: str, thresholds: dict) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.45, 0.18, 0.25, 0.12],
        # ⓘ hints that each title is hoverable (see PANEL_HELP / captureevents
        # below) for a plain-language explanation of that panel.
        subplot_titles=(
            "Price · EMA (Exponential Moving Average) ribbon · signals ⓘ",
            "MACD (Moving Average Convergence Divergence) ⓘ",
            "Whale (red) vs retail (green) accumulation ⓘ",
            "Volume (trend-colored) ⓘ",
        ),
    )

    # ── Panel 1: candles + EMA ribbon + close price + signal markers ────
    # Ribbon lines drawn first (bottom layer), then the close-price line in a
    # single bold, unmistakable color (so it's never confused with the 5
    # thinner ribbon lines), then candles on top so their wicks stay visible.
    ema_cols = [c for c in frame.columns if c.startswith("ema_")]
    for i, (color, col) in enumerate(zip(RIBBON_STEPS, ema_cols)):
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[col],
                mode="lines",
                line=dict(color=color, width=1),
                name="Trend ribbon (EMAs)",
                legendgroup="ribbon",
                hoverinfo="skip",
                showlegend=(i == 0),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["close"],
            mode="lines",
            line=dict(color=C["price_line"], width=1.6),
            name="Close price",
            opacity=0.9,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Candle",
            increasing_line_color=C["up"],
            increasing_fillcolor=C["up"],
            decreasing_line_color=C["down"],
            decreasing_fillcolor=C["down"],
        ),
        row=1,
        col=1,
    )
    buys = frame[frame["buy_signal"]]
    sells = frame[frame["sell_signal"]]
    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys["low"] * 0.98,
            mode="markers",
            marker=dict(symbol="triangle-up", size=11, color=C["buy"],
                        line=dict(color=C["surface"], width=2)),
            name="Buy signal",
            customdata=buys["buy_reason"],
            hovertemplate="BUY %{x|%Y-%m-%d}<br>%{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells["high"] * 1.02,
            mode="markers",
            marker=dict(symbol="triangle-down", size=11, color=C["sell"],
                        line=dict(color=C["surface"], width=2)),
            name="Sell warning",
            customdata=sells["sell_reason"],
            hovertemplate="SELL %{x|%Y-%m-%d}<br>%{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # ── Panel 4: volume colored by ribbon trend ─────────────────────────
    trend_color = pd.Series(C["neutral"], index=frame.index)
    trend_color[frame["ribbon_bullish"]] = C["whale"]   # "red ribbon" = bullish
    trend_color[frame["ribbon_bearish"]] = C["blue"]    # "blue ribbon" = bearish
    fig.add_trace(
        go.Bar(
            x=frame.index,
            y=frame["volume"],
            marker_color=trend_color.tolist(),
            name="Volume",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>vol %{y:,.0f}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    # The bar itself is one trace colored per-point, so it can't produce
    # per-color legend entries on its own — add 3 invisible marker traces
    # just to give this panel's legend real swatches for what the colors mean.
    for label, color in (("Uptrend", C["whale"]), ("Downtrend", C["blue"]), ("Mixed", C["neutral"])):
        fig.add_trace(
            go.Scatter(
                x=[frame.index[0]],
                y=[None],
                mode="markers",
                marker=dict(symbol="square", size=9, color=color),
                name=label,
                legend="legend4",
                hoverinfo="skip",
            ),
            row=4,
            col=1,
        )

    # ── Panel 3: whale vs retail bars + threshold guides ────────────────
    fig.add_trace(
        go.Bar(
            x=frame.index,
            y=frame["whale_score"],
            marker_color=C["whale"],
            name="Whale accumulation",
            legend="legend3",
            hovertemplate="%{x|%Y-%m-%d}<br>whale %{y:.1f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=frame.index,
            y=frame["retail_score"],
            marker_color=C["retail"],
            name="Retail accumulation",
            legend="legend3",
            hovertemplate="%{x|%Y-%m-%d}<br>retail %{y:.1f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    for key, dash in (("momentum", "dot"), ("rise", "dash"), ("soar", "solid")):
        fig.add_hline(
            y=thresholds[key],
            line=dict(color=C["muted"], width=1, dash=dash),
            annotation_text=f"{key} {thresholds[key]}",
            annotation_font=dict(color=C["ink2"], size=10),
            row=3,
            col=1,
        )

    # ── Panel 2: MACD ───────────────────────────────────────────────────
    hist_colors = [C["up"] if v >= 0 else C["down"] for v in frame["macd_hist"]]
    fig.add_trace(
        go.Bar(
            x=frame.index,
            y=frame["macd_hist"],
            marker_color=hist_colors,
            name="MACD hist",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index, y=frame["macd"], mode="lines",
            line=dict(color=C["blue"], width=2), name="MACD",
            legend="legend2",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index, y=frame["macd_signal"], mode="lines",
            line=dict(color=C["orange"], width=2), name="Signal",
            legend="legend2",
        ),
        row=2,
        col=1,
    )
    golden = frame[frame["macd_golden_cross"]]
    death = frame[frame["macd_death_cross"]]
    fig.add_trace(
        go.Scatter(
            x=golden.index, y=golden["macd"], mode="markers",
            marker=dict(symbol="triangle-up", size=9, color=C["buy"],
                        line=dict(color=C["surface"], width=1)),
            name="Golden cross",
            legend="legend2",
            hovertemplate="Golden cross %{x|%Y-%m-%d}<br>momentum turning up<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=death.index, y=death["macd"], mode="markers",
            marker=dict(symbol="triangle-down", size=9, color=C["muted"],
                        line=dict(color=C["surface"], width=1)),
            name="Death cross",
            legend="legend2",
            hovertemplate="Death cross %{x|%Y-%m-%d}<br>momentum turning down (not wired to a verdict)<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # A bordered box around each panel makes it unambiguous which title (e.g.
    # "MACD") belongs to which chart below it, especially once the shared
    # legend above pushes things closer together.
    for r in range(1, 5):
        fig.add_shape(
            type="rect",
            xref="x domain",
            yref="y domain",
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            line=dict(color=C["grid"], width=1.3),
            fillcolor="rgba(0,0,0,0)",
            layer="below",
            row=r,
            col=1,
        )

    # Each panel gets its own small legend, anchored just outside the right
    # edge at that panel's own row height (row y-domains are fixed by the
    # row_heights/vertical_spacing above — computed once and hardcoded here
    # rather than looked up, since those inputs never change at runtime).
    legend_style = dict(
        orientation="v", xanchor="left", yanchor="top", x=1.01,
        font=dict(size=9), bgcolor=C["surface"], bordercolor=C["grid"], borderwidth=1,
    )
    fig.update_layout(
        height=980,
        barmode="group",
        bargap=0.25,
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(color=C["ink2"], family='system-ui, -apple-system, "Segoe UI", sans-serif'),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["surface"], bordercolor=C["grid"], font=dict(size=11, color=C["ink2"])),
        legend=dict(y=1.0, **legend_style),
        legend2=dict(y=0.5675, **legend_style),
        legend3=dict(y=0.3645, **legend_style),
        legend4=dict(y=0.102, **legend_style),
        margin=dict(l=40, r=150, t=68, b=30),
        xaxis_rangeslider_visible=False,
    )
    # Quarterly gridlines/ticks (Jan/Apr/Jul/Oct 1st), regardless of the
    # selected time range or Plotly's own auto-spacing, as a consistent
    # reference grid across all 4 panels.
    fig.update_xaxes(gridcolor=C["grid"], zeroline=False, dtick="M3", tick0="2020-01-01")
    # shared_xaxes hides date labels on every row but the bottom one by
    # default — show them on the price panel too so you don't have to look
    # all the way down to the MACD panel to tell what date you're looking at.
    # Placed on top (rather than the panel's own bottom edge) so they don't
    # collide with the MACD panel's title sitting just below this one.
    fig.update_xaxes(showticklabels=True, side="top", tickfont=dict(size=9), row=1, col=1)
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    # The top-side date labels just added sit at the same height Plotly's
    # automatic subplot-title annotation uses by default — nudge panel 1's
    # title (always the first of the 4 subplot_titles annotations) further
    # up so the two don't overlap.
    fig.layout.annotations[0].update(yshift=18)
    # Hovering any panel title shows a plain-language explanation of that
    # panel — captureevents=True is what makes an otherwise-static text
    # annotation respond to mouse hover at all.
    for i, help_text in enumerate(PANEL_HELP):
        fig.layout.annotations[i].update(hovertext=help_text, captureevents=True)
    return fig


@st.cache_data(ttl=300)
def load_freshness() -> dict:
    conn = store.connect()
    fresh = store.source_freshness(conn)
    conn.close()
    return fresh


def _freshness_section(demo_mode: bool) -> None:
    """Data validity: latest data point + inherent delay + stale flag per source."""
    with st.expander("🕐 Data freshness & validity — how current is what you're seeing?", expanded=False):
        fresh = load_freshness()
        if demo_mode:
            st.caption("Demo mode: dates below are synthetic, not real market data.")
        today = pd.Timestamp.today().normalize()
        rows = []
        for key, info in SOURCE_INFO.items():
            latest = fresh.get(key)
            if latest is None:
                status = "❌ no data"
            else:
                age = (today - pd.Timestamp(latest)).days
                status = "✅ current" if age <= info["stale_after"] else f"⚠️ stale ({age}d old)"
            rows.append(
                {
                    "Source": info["name"],
                    "Latest data point": latest or "—",
                    "Inherent delay": info["delay"],
                    "Status": status,
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=_table_height(len(rows)),
            column_config={
                "Source": st.column_config.TextColumn(
                    "Source",
                    help=(
                        "ATS = Alternative Trading System (a dark pool). "
                        "FINRA = Financial Industry Regulatory Authority. "
                        "SEC 13F = quarterly institutional-holdings filing "
                        "with the Securities and Exchange Commission."
                    ),
                ),
                "Latest data point": st.column_config.TextColumn(
                    "Latest data point", help="The newest date this source's cached data covers."
                ),
                "Inherent delay": st.column_config.TextColumn(
                    "Inherent delay", help="How old the information is even immediately after a fresh pull — a property of the source, not this app."
                ),
                "Status": st.column_config.TextColumn(
                    "Status", help="✅ within the source's normal update cycle · ⚠️ older than expected, refresh recommended."
                ),
            },
        )
        st.caption(
            "⚠️ means the cache is older than the source's normal publication "
            "cycle — hit **Refresh data** in the left panel. \"Inherent delay\" "
            "is how old the information is even right after a refresh: prices "
            "and dark-pool volume describe **yesterday/today**, weekly ATS "
            "(Alternative Trading System) describes **2–4 weeks ago**, and 13F "
            "(SEC Form 13F, quarterly institutional-holdings filing) holdings "
            "describe **last quarter**. Signals weight the fast sources most, "
            "and 13F least. FINRA = Financial Industry Regulatory Authority "
            "(publishes the dark-pool data); SEC = Securities and Exchange "
            "Commission (publishes 13F)."
        )


def _bulk_import_ui(active: str, working: list[str], pinned: set[str]) -> None:
    """Paste-replace the active watchlist's entire ticker list in one action
    — the practical way to keep a large imported list (e.g. from Yahoo
    Finance) in sync: copy your list from wherever it lives, paste it here,
    done. One-by-one Add/Remove below still works for small tweaks."""
    with st.expander("📋 Paste-import a ticker list (e.g. re-sync from Yahoo)"):
        st.caption(
            "Paste tickers separated by commas, spaces, or newlines — this "
            "**replaces** the whole list below. Pins on tickers no longer "
            "present are dropped; everything else (other watchlists) is "
            "untouched."
        )
        pasted = st.text_area(
            "Paste tickers", key=f"bulk_import_text_{active}",
            label_visibility="collapsed", height=100,
            placeholder="AAPL, MSFT, NVDA\nTSLA\n...",
        )
        if st.button("Replace with pasted list", key=f"bulk_import_apply_{active}") and pasted.strip():
            new_order, seen = [], set()
            for raw_sym in re.split(r"[\s,]+", pasted.strip()):
                sym = raw_sym.strip().upper()
                if sym and sym not in seen:
                    seen.add(sym)
                    new_order.append(sym)
            if not new_order:
                st.warning("No tickers found in the pasted text.")
            else:
                working[:] = new_order
                pinned &= set(new_order)
                # Can't reassign st.session_state["detail_ticker"] directly
                # here — the sidebar's Ticker selectbox (same key) has
                # already been instantiated earlier in this run, and
                # Streamlit forbids modifying a widget's key after that.
                # Stash it the same way go_to_ticker() does and apply it at
                # the top of the *next* run, before that widget renders.
                if st.session_state.get("detail_ticker") not in new_order:
                    st.session_state["pending_nav"] = ("overview", new_order[0])
                st.session_state.pop("ov_sort", None)
                sync_watchlist_state()
                st.success(f"Replaced “{active}” with {len(new_order)} tickers.")
                st.rerun()


def _watchlist_edit_ui(cfg, working: list[str], pinned: set[str]) -> None:
    """Add/delete/pin/drag-reorder tickers — shown in place of the scored
    table while Edit mode is on. `working`/`pinned` are the live list/set
    objects inside st.session_state["watchlists"][active_watchlist] — every
    mutation below (append/remove/&=) writes straight through to session
    state, same object, no reassignment needed except where the whole list
    is replaced wholesale (drag-reorder, paste-import)."""
    active = st.session_state["active_watchlist"]

    # A widget's session_state value can't be reassigned after that widget
    # has already been instantiated in the same run — so clearing the text
    # input after "Add" needs the stash-then-rerun pattern, applied *before*
    # the widget below is created.
    if st.session_state.pop("_clear_add_ticker_input", False):
        st.session_state["add_ticker_input"] = ""

    if gist_configured():
        st.caption("Changes here **sync across your devices/browsers** via the configured GitHub Gist.")
    else:
        st.caption(
            "Changes here apply only to **this browser session** and reset on "
            "reload — see README for free cross-device sync."
        )
    add_col1, add_col2 = st.columns([4, 1], gap="small")
    new_ticker = add_col1.text_input(
        "Add a ticker symbol",
        key="add_ticker_input",
        label_visibility="collapsed",
        placeholder="Add a ticker, e.g. GOOG",
    )
    if add_col2.button("➕ Add", use_container_width=True) and new_ticker.strip():
        sym = new_ticker.strip().upper()
        if sym in working:
            st.warning(f"{sym} is already in “{active}”.")
        else:
            working.append(sym)
            st.session_state["_clear_add_ticker_input"] = True
            st.session_state.pop("ov_sort", None)
            sync_watchlist_state()
            st.rerun()

    _bulk_import_ui(active, working, pinned)

    if not working:
        st.info("This watchlist is empty — add a ticker above.")
        return

    def _label(t: str) -> str:
        if cfg.demo_mode:
            return t
        name = load_company_name(t)
        return f"{t} — {name}" if name else t

    labels = [_label(t) for t in working]
    st.caption("⠿ Drag to reorder:")
    # Key derived from the active watchlist + its current ticker set/order
    # (not a fixed string): a bidirectional component only calls
    # setComponentValue() in response to a real user drag, so after any
    # *other* change to `working` (add/delete/switch watchlist), an
    # unchanged key would make it echo back its last cached value — the
    # pre-change list — which we'd then wrongly treat as a new drag and use
    # to clobber the just-made change. A content-derived key forces a
    # remount (fresh `default=labels`) whenever the set/order actually
    # changes underneath it, so the returned value only ever differs from
    # `labels` when the user genuinely dragged something.
    drag_key = "watchlist_drag_" + active + "|" + "|".join(working)
    new_labels = sort_items(labels, direction="vertical", key=drag_key)
    if new_labels != labels:
        new_order = [working[labels.index(lbl)] for lbl in new_labels]
        working[:] = new_order
        st.session_state.pop("ov_sort", None)
        sync_watchlist_state()
        st.rerun()

    st.caption("Pin / remove:")
    pinned_before = set(pinned)
    for ticker in working:
        row = st.columns([0.5, 2.5, 0.6], gap="small")
        is_pinned = row[0].checkbox(
            "Pin", value=ticker in pinned, key=f"pin_{active}_{ticker}", label_visibility="collapsed"
        )
        if is_pinned:
            pinned.add(ticker)
        else:
            pinned.discard(ticker)
        row[1].write(f"**{ticker}**")
        if row[2].button("🗑️", key=f"del_{active}_{ticker}", help=f"Remove {ticker} from “{active}”"):
            working.remove(ticker)
            pinned.discard(ticker)
            # Same deferred-assignment reasoning as the paste-import handler
            # above — can't set detail_ticker directly this late in the run.
            if st.session_state.get("detail_ticker") == ticker:
                st.session_state["pending_nav"] = ("overview", working[0] if working else None)
            st.session_state.pop("ov_sort", None)
            sync_watchlist_state()
            st.rerun()

    # Pin checkboxes trigger Streamlit's own implicit rerun (no explicit
    # st.rerun() above to hang a save call off), so sync here instead, once
    # per render, only when a pin actually changed this run.
    if pinned != pinned_before:
        sync_watchlist_state()


# Column key -> (header label, tooltip, dataframe field to sort by). Pin and
# the chart-link column aren't sortable (no dataframe field), so they render
# as plain labels rather than clickable header buttons.
#
# Labels are kept short — the full meaning lives in the tooltip (`help=` on
# the header button), not the label itself. A full-length label ("Retail %")
# in a narrow column at a fixed font size used to wrap to two lines and
# break row alignment; short labels + the nowrap/ellipsis CSS above make
# that impossible even at the narrowest realistic column width.
TABLE_COLUMNS = [
    ("pin", "📌", "Pinned in Edit mode — always sorted to the top, regardless of other sorting. Click to sort.", "_pinned"),
    ("ticker", "Ticker", "The stock symbol. Click to sort.", "Ticker"),
    ("company", "Company", "Company / issuer name. Click to sort.", "Company"),
    ("action", "Action", "🟢 Buy · 🟠 Trim · 🟡 Watch · 🔵 Hold · ⚪ Wait — see legend above. Click to sort.", "_sort"),
    ("signal", "Sig", "Signal — 🔴/🟢 = a buy/sell signal fired this week (hover for detail). Click to sort.", "_has_signal"),
    ("close", "Close", "Last COMPLETED daily close (not a live quote). Click to sort.", "Close"),
    ("whale", "Whale", "Whale % — 0-100 estimate of big-investor buying. 50 = neutral. Click to sort.", "Whale %"),
    ("delta", "Δ20d", "Change in Whale % over ~20 trading days. Click to sort.", "Δ 20d"),
    ("retail", "Retail", "Retail % — 0-100 estimate of regular/individual-investor buying. Click to sort.", "Retail %"),
    ("zone", "Zone", "weak <35 · momentum 35-50 · rise 50-75 · soar >75. Click to sort.", "_zone_rank"),
    ("chart", "📈", "Open this stock's full chart.", None),
]
# Widths roughly proportional to what each column actually needs: icon-only
# columns (pin, chart) are narrow, Company/Whale (bar + number) get the most
# room. Sums to a set of fractions, not pixels — Streamlit distributes the
# row's actual width across them, and the CSS above ellipses any label that
# still doesn't fit rather than wrapping it.
ROW_COLS = [0.3, 0.55, 1.3, 0.8, 0.4, 0.65, 1.3, 0.55, 0.65, 0.7, 0.35]


def _cycle_sort(col_key: str) -> None:
    cur = st.session_state.get("ov_sort")
    if cur is None or cur[0] != col_key:
        st.session_state["ov_sort"] = (col_key, True)
    elif cur[1]:
        st.session_state["ov_sort"] = (col_key, False)
    else:
        st.session_state["ov_sort"] = None
    st.rerun()


def overview_page(cfg):
    working = get_working_watchlist(cfg)
    pinned = get_active_pinned(cfg)

    st.markdown(
        " · ".join(
            f'<span title="{html.escape(tip)}" style="cursor:help;">{label}</span>'
            for label, tip in ACTION_GUIDE
        ),
        unsafe_allow_html=True,
    )

    top = st.columns([5, 1], gap="small")
    top[0].subheader(f"Watchlist — {st.session_state['active_watchlist']}")
    edit_mode = st.session_state.get("watchlist_edit_mode", False)
    if top[1].button("✓ Done" if edit_mode else "✏️ Edit", use_container_width=True):
        st.session_state["watchlist_edit_mode"] = not edit_mode
        st.rerun()

    if edit_mode:
        _watchlist_edit_ui(cfg, working, pinned)
        return

    if not working:
        st.warning("Your watchlist is empty. Click ✏️ Edit to add a ticker.")
        return

    df = load_overview(tuple(working))
    ok = df[df["Status"] == "ok"].copy()
    ok["_pinned"] = ok["Ticker"].isin(pinned)
    ok["_has_signal"] = ok["_signal_detail"].astype(bool)
    missing = df[df["Status"] != "ok"]
    if not ok.empty:
        sort_state = st.session_state.get("ov_sort")
        if sort_state:
            col_key, asc = sort_state
            sort_field = next(f for k, _, _, f in TABLE_COLUMNS if k == col_key)
            # Pinned rows stay on top no matter which column is sorted (even
            # when that column is Pin itself) — same _pinned-first tiebreak
            # the default branch below already uses.
            display = ok.sort_values(
                ["_pinned", sort_field], ascending=[False, asc], na_position="last"
            ).reset_index(drop=True)
        else:
            display = ok.sort_values(
                ["_pinned", "_sort", "Whale %"], ascending=[False, True, False]
            ).reset_index(drop=True)

        # A native Streamlit row list, not st.dataframe: avoids the built-in
        # row-selection checkbox (confusing — looked like a "pin" toggle) and
        # guarantees no nested scrollbar, since plain elements never scroll
        # internally the way a data-grid does. Wrapped in a keyed container
        # so the scoped CSS above can add cell borders + compact spacing.
        with st.container(key="watchlist_table"):
            with st.container(key="watchlist_header_row"):
                hdr = st.columns(ROW_COLS, gap="small")
                for c, (col_key, label, tip, sort_field) in zip(hdr, TABLE_COLUMNS):
                    if sort_field is None:
                        c.markdown(f'<span title="{html.escape(tip)}" style="font-size:0.78rem;color:#898781;">{label}</span>', unsafe_allow_html=True)
                    else:
                        arrow = ""
                        if sort_state and sort_state[0] == col_key:
                            arrow = " ▲" if sort_state[1] else " ▼"
                        if c.button(label + arrow, key=f"hdr_{col_key}", help=tip, use_container_width=True):
                            _cycle_sort(col_key)
            for _, r in display.iterrows():
                cells = st.columns(ROW_COLS, gap="small")
                cells[0].write("📌" if r["_pinned"] else "")
                cells[1].markdown(f"**{r['Ticker']}**")
                company = r["Company"] or "—"
                cells[2].markdown(
                    f'<span title="{html.escape(company)}" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;font-size:0.85rem;">{html.escape(company)}</span>',
                    unsafe_allow_html=True,
                )
                cells[3].write(r["Action"])
                if r["_signal_detail"]:
                    cells[4].markdown(f'<span title="{html.escape(r["_signal_detail"])}" style="cursor:help;">{r["Signal"]}</span>', unsafe_allow_html=True)
                else:
                    cells[4].write("—")
                cells[5].write(f"{r['Close']:,.2f}")
                wv = float(r["Whale %"])
                pct = max(0.0, min(100.0, wv))
                cells[6].markdown(
                    f'<span class="wt-bar-track"><span class="wt-bar-fill" style="width:{pct:.0f}%;"></span></span>{wv:.1f}',
                    unsafe_allow_html=True,
                )
                d20 = r["Δ 20d"]
                if d20 is None or pd.isna(d20):
                    cells[7].write("—")
                else:
                    delta_color = C["up"] if d20 >= 0 else C["down"]
                    cells[7].markdown(f'<span style="color:{delta_color};">{d20:+.1f}</span>', unsafe_allow_html=True)
                cells[8].write(f"{float(r['Retail %']):.1f}")
                cells[9].write(r["Zone"])
                if cells[10].button("📈", key=f"chart_{r['Ticker']}", help=f"Open {r['Ticker']}'s chart"):
                    go_to_ticker(r["Ticker"])
    if not missing.empty:
        st.warning(
            "No cached data for: "
            + ", ".join(missing["Ticker"])
            + ". Click 🔄 Refresh data (left panel), or check the ticker symbol."
        )


BANNER_BY_SEVERITY = {"success": st.success, "warning": st.warning}


def verdict_banner(ticker: str, thr: dict) -> None:
    """Plain-language "what do I do now" verdict, always from the weekly view
    so it matches the overview table regardless of the selected bar size.

    The headline (from signals.current_action) always states outright
    whether this IS a buy signal, a sell signal, or no signal at all —
    e.g. "🟢 BUY SIGNAL — ..." — rendered as a header inside a colored
    box so it can't be missed or misread as trader jargon.
    """
    weekly = load_ticker_frame(ticker, "W")
    act = signals.current_action(weekly, thr)
    banner = BANNER_BY_SEVERITY.get(act["severity"], st.info)
    banner(f"### {act['headline']}\n\n{act['detail']}")
    caption_bits = []
    if act["invalidation"]:
        caption_bits.append(f"↳ This changes if: {act['invalidation']}")
    if not weekly.empty:
        caption_bits.append(
            f"Verdict computed from data through **{weekly.index.max():%Y-%m-%d}** "
            "(see 📖 How to read this in the left panel for per-source delays)."
        )
    if caption_bits:
        st.caption("  \n".join(caption_bits))


def _fg_chip_html() -> str:
    """Small band-colored pill next to the page title — e.g. "😱 62 · Greed"
    — so the market-wide reading is visible from every page, not just the
    Fear & Greed page itself. Empty string (renders nothing) until the
    first refresh has populated the sentiment table."""
    scores = load_sentiment()
    if scores.empty or "composite" not in scores.columns:
        return ""
    composite = scores["composite"].dropna()
    if composite.empty:
        return ""
    val = float(composite.iloc[-1])
    band = fear_greed.label(val)
    color = FG_BAND_COLORS[band]
    return (
        f'<span title="Market Fear &amp; Greed Index — {html.escape(band)}, '
        f'updated {composite.index[-1]:%Y-%m-%d}. See the 😱 Fear &amp; Greed '
        f'page in the left panel." '
        f'style="cursor:help;font-size:0.8rem;font-weight:600;padding:0.15rem 0.6rem;'
        f'border-radius:999px;background:{color};color:#fff;margin-left:0.6rem;'
        f'vertical-align:middle;">😱 {val:.0f} · {html.escape(band)}</span>'
    )


def _fg_gauge_figure(score: float) -> go.Figure:
    steps = [
        {"range": [lo, min(hi, 100)], "color": FG_BAND_COLORS[name]}
        for lo, hi, name in fear_greed.FG_BANDS
    ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 38, "color": C["ink"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": C["muted"], "tickfont": {"size": 9}},
                "bar": {"color": C["ink"], "thickness": 0.22},
                "bgcolor": C["surface"],
                "borderwidth": 0,
                "steps": steps,
            },
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=25, r=25, t=25, b=5),
        paper_bgcolor=C["surface"],
        font=dict(color=C["ink"], family="sans-serif"),
    )
    return fig


def _fg_history_figure(composite: pd.Series) -> go.Figure:
    hist = composite.tail(252)
    fig = go.Figure()
    for lo, hi, name in fear_greed.FG_BANDS:
        fig.add_hrect(y0=lo, y1=min(hi, 100), fillcolor=FG_BAND_COLORS[name], opacity=0.10, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=hist.index, y=hist.values, mode="lines",
            line=dict(color=C["ink"], width=1.6), hovertemplate="%{y:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=35, r=15, t=10, b=25),
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(color=C["ink2"], size=10),
        yaxis=dict(range=[0, 100], gridcolor=C["grid"]),
        xaxis=dict(gridcolor=C["grid"]),
        showlegend=False,
    )
    return fig


def _fg_sparkline_figure(series: pd.Series, color: str) -> go.Figure:
    hist = series.dropna().tail(90)
    fig = go.Figure(
        go.Scatter(x=hist.index, y=hist.values, mode="lines", line=dict(color=color, width=1.5))
    )
    fig.update_layout(
        height=64,
        margin=dict(l=0, r=0, t=2, b=2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def _fg_empty_state() -> None:
    st.warning(
        "No sentiment data cached yet. Click 🔄 Refresh data in the left panel — "
        "the first refresh fetches ~46 symbols (index/ETF proxies + a large-cap "
        "basket) and can take a little longer than a normal refresh."
    )


def fear_greed_page(cfg) -> None:
    """Market-wide sentiment gauge modeled on CNN's Fear & Greed Index
    (edition.cnn.com/markets/fear-and-greed), computed from free Yahoo data —
    see whaletrading/indicators/fear_greed.py for the full methodology and
    what's a documented proxy vs. an exact match."""
    st.subheader("😱 Fear & Greed Index")
    st.caption(
        "A market-wide, contrarian sentiment gauge — not a buy/sell signal for "
        "any one stock. Extreme fear has historically been a buying "
        "opportunity; extreme greed, a warning sign. Modeled on CNN's Fear & "
        "Greed Index using 6 of its 7 indicators (Put/Call options demand has "
        "no free daily data source, so it's omitted)."
    )

    scores = load_sentiment()
    if scores.empty or "composite" not in scores.columns:
        _fg_empty_state()
        return
    composite = scores["composite"].dropna()
    if composite.empty:
        _fg_empty_state()
        return

    if cfg.demo_mode:
        st.info("🎭 Demo mode — this sentiment history is synthetic, not a real market reading.")

    latest = float(composite.iloc[-1])
    latest_band = fear_greed.label(latest)

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.plotly_chart(
            _fg_gauge_figure(latest), use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown(
            f'<div style="text-align:center;font-size:1.35rem;font-weight:700;'
            f'color:{FG_BAND_COLORS[latest_band]};">{latest_band}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"As of {composite.index[-1]:%Y-%m-%d} · updates with 🔄 Refresh data")
    with right:
        cols = st.columns(4, gap="small")
        for col, (readable, offset) in zip(cols, PRIOR_READINGS):
            val = None if len(composite) <= offset else float(composite.iloc[-1 - offset])
            if val is None:
                col.metric(readable, "—")
            else:
                col.metric(readable, f"{val:.0f}", help=fear_greed.label(val))
        st.plotly_chart(
            _fg_history_figure(composite), use_container_width=True,
            config={"displayModeBar": False},
        )

    st.divider()
    st.markdown("##### What's driving it")
    available = [k for k in fear_greed.INDICATOR_INFO if k in scores.columns and not scores[k].dropna().empty]
    for row_start in range(0, len(available), 3):
        row_cols = st.columns(3, gap="medium")
        for col, key in zip(row_cols, available[row_start : row_start + 3]):
            series = scores[key].dropna()
            name, desc = fear_greed.INDICATOR_INFO[key]
            val = float(series.iloc[-1])
            band = fear_greed.label(val)
            color = FG_BAND_COLORS[band]
            with col.container(border=True):
                st.markdown(f"**{name}**")
                st.markdown(
                    f'<span style="font-size:1.3rem;font-weight:700;color:{color};">{val:.0f}</span> '
                    f'<span style="color:{color};font-weight:600;">{band}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(desc)
                st.plotly_chart(
                    _fg_sparkline_figure(series, color), use_container_width=True,
                    config={"displayModeBar": False},
                )
    missing = [name for key, (name, _) in fear_greed.INDICATOR_INFO.items() if key not in available]
    caveat = (
        "⚠️ Strength and Breadth are computed from a fixed ~40-stock large-cap "
        "basket, not full NYSE breadth — a documented proxy, not an exact "
        "match to CNN's numbers. Put/Call options demand is omitted entirely "
        "(no free daily data source). Treat this as directional context, not "
        "a precise reading."
    )
    if missing:
        caveat += " Currently missing (source unavailable this refresh): " + ", ".join(missing) + "."
    st.caption(caveat)


# Static reference diagram for the "How to read this" guide: the buy/sell
# signal logic (OR of AND-paths) from signals.py, rendered once here rather
# than regenerated per view. Update this by hand if signals.py's rules change.
SIGNAL_DIAGRAM_SVG = """
<div style="max-width:900px;margin:0 auto;">
<svg viewBox="0 0 900 742" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="dtitle ddesc" style="width:100%;height:auto;display:block;">
  <title id="dtitle">WhaleTrading buy/sell signal composition</title>
  <desc id="ddesc">The buy signal is an OR of three AND-conditions (dip reversal, ribbon turn, MACD cross). The sell signal is an OR of two AND-conditions (whale-to-retail shift, yellow candle).</desc>
  <style>
    .wtdiag-root {
      --card-bg: #f3f4f6; --card-border: #d1d5db;
      --chip-bg: #ffffff; --chip-border: #cbd5e1;
      --text: #1f2937; --muted: #6b7280;
      --line: #9ca3af;
      --buy: #16a34a; --sell: #dc2626;
      --header-buy: #dcfce7; --header-sell: #fee2e2;
    }
    @media (prefers-color-scheme: dark) {
      .wtdiag-root {
        --card-bg: #1f2937; --card-border: #374151;
        --chip-bg: #111827; --chip-border: #374151;
        --text: #e5e7eb; --muted: #9ca3af;
        --line: #6b7280;
        --buy: #22c55e; --sell: #f87171;
        --header-buy: #14532d; --header-sell: #7f1d1d;
      }
    }
    .wtdiag-root text { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--text); }
    .wtdiag-root .section { font-size: 15px; font-weight: 600; }
    .wtdiag-root .legend { font-size: 12px; fill: var(--muted); }
    .wtdiag-root .cardhdr { font-size: 12px; font-weight: 700; }
    .wtdiag-root .chip { font-size: 12px; }
    .wtdiag-root .andlbl { font-size: 10px; fill: var(--muted); font-weight: 600; letter-spacing: 0.5px; }
    .wtdiag-root .orlbl { font-size: 13px; fill: var(--muted); font-weight: 700; }
    .wtdiag-root .outlbl { font-size: 15px; font-weight: 700; fill: #fff; }
    .wtdiag-root .card { fill: var(--card-bg); stroke: var(--card-border); stroke-width: 1; rx: 8; }
    .wtdiag-root .chipbox { fill: var(--chip-bg); stroke: var(--chip-border); stroke-width: 1; rx: 6; }
    .wtdiag-root .conn { stroke: var(--line); stroke-width: 1.5; fill: none; }
    .wtdiag-root .arrow { stroke: var(--line); stroke-width: 1.5; fill: none; marker-end: url(#wtdiag-arrowhead); }
  </style>
  <defs>
    <marker id="wtdiag-arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="var(--line)"/>
    </marker>
  </defs>
  <g class="wtdiag-root">

  <text x="450" y="24" text-anchor="middle" class="legend">Green = triggers a buy · Red = triggers a sell · each card's rows are AND'd; cards combine via OR</text>

  <!-- BUY SECTION -->
  <text x="90" y="52" class="section">BUY signal — fires if ANY of these 3 paths is true</text>

  <!-- Card A: Dip reversal (4 conditions) -->
  <rect x="90" y="64" width="220" height="210" class="card"/>
  <rect x="90" y="64" width="220" height="26" fill="var(--header-buy)" rx="8"/>
  <rect x="90" y="82" width="220" height="8" fill="var(--header-buy)"/>
  <text x="200" y="81" text-anchor="middle" class="cardhdr">Path 1 — Dip reversal</text>
  <rect x="102" y="98" width="196" height="32" class="chipbox"/>
  <text x="200" y="118" text-anchor="middle" class="chip">Up-close candle after a decline</text>
  <text x="200" y="139" text-anchor="middle" class="andlbl">AND</text>
  <rect x="102" y="146" width="196" height="32" class="chipbox"/>
  <text x="200" y="166" text-anchor="middle" class="chip">Ribbon still bearish (downtrend)</text>
  <text x="200" y="187" text-anchor="middle" class="andlbl">AND</text>
  <rect x="102" y="194" width="196" height="32" class="chipbox"/>
  <text x="200" y="214" text-anchor="middle" class="chip">Ribbon tightening (narrowing)</text>
  <text x="200" y="235" text-anchor="middle" class="andlbl">AND</text>
  <rect x="102" y="242" width="196" height="26" class="chipbox"/>
  <text x="200" y="259" text-anchor="middle" class="chip">Whale rising OR retail falling</text>

  <!-- Card B: Ribbon turn (3 conditions) -->
  <rect x="340" y="64" width="220" height="166" class="card"/>
  <rect x="340" y="64" width="220" height="26" fill="var(--header-buy)" rx="8"/>
  <rect x="340" y="82" width="220" height="8" fill="var(--header-buy)"/>
  <text x="450" y="81" text-anchor="middle" class="cardhdr">Path 2 — Ribbon turn</text>
  <rect x="352" y="98" width="196" height="32" class="chipbox"/>
  <text x="450" y="118" text-anchor="middle" class="chip">Up-close candle</text>
  <text x="450" y="139" text-anchor="middle" class="andlbl">AND</text>
  <rect x="352" y="146" width="196" height="32" class="chipbox"/>
  <text x="450" y="163" text-anchor="middle" class="chip">Ribbon just turned bullish</text>
  <text x="450" y="176" text-anchor="middle" class="chip" font-size="10">("red ribbon forming")</text>
  <text x="450" y="195" text-anchor="middle" class="andlbl">AND</text>
  <rect x="352" y="200" width="196" height="26" class="chipbox"/>
  <text x="450" y="217" text-anchor="middle" class="chip">Whale rising OR retail falling</text>

  <!-- Card C: MACD cross (2 conditions) -->
  <rect x="590" y="64" width="220" height="122" class="card"/>
  <rect x="590" y="64" width="220" height="26" fill="var(--header-buy)" rx="8"/>
  <rect x="590" y="82" width="220" height="8" fill="var(--header-buy)"/>
  <text x="700" y="81" text-anchor="middle" class="cardhdr">Path 3 — MACD cross</text>
  <rect x="602" y="98" width="196" height="32" class="chipbox"/>
  <text x="700" y="118" text-anchor="middle" class="chip">MACD golden cross</text>
  <text x="700" y="139" text-anchor="middle" class="andlbl">AND</text>
  <rect x="602" y="146" width="196" height="32" class="chipbox"/>
  <text x="700" y="166" text-anchor="middle" class="chip">Whale score rising</text>

  <!-- Converge to OR -->
  <path d="M200,274 L200,290" class="conn"/>
  <path d="M450,230 L450,290" class="conn"/>
  <path d="M700,186 L700,290" class="conn"/>
  <path d="M200,290 L700,290" class="conn"/>
  <text x="450" y="303" text-anchor="middle" class="orlbl">OR</text>
  <path d="M450,290 L450,320" class="arrow"/>

  <rect x="350" y="322" width="200" height="42" rx="8" fill="var(--buy)"/>
  <text x="450" y="349" text-anchor="middle" class="outlbl">BUY signal</text>

  <!-- divider -->
  <line x1="60" y1="392" x2="840" y2="392" stroke="var(--card-border)" stroke-width="1"/>

  <!-- SELL SECTION -->
  <text x="90" y="420" class="section">SELL / trim signal — fires if EITHER of these 2 paths is true</text>

  <!-- Card D: whale-&gt;retail shift (3 conditions) -->
  <rect x="215" y="432" width="220" height="166" class="card"/>
  <rect x="215" y="432" width="220" height="26" fill="var(--header-sell)" rx="8"/>
  <rect x="215" y="450" width="220" height="8" fill="var(--header-sell)"/>
  <text x="325" y="449" text-anchor="middle" class="cardhdr">Path 1 — Whale→retail shift</text>
  <rect x="227" y="466" width="196" height="32" class="chipbox"/>
  <text x="325" y="486" text-anchor="middle" class="chip">Whale score falling</text>
  <text x="325" y="507" text-anchor="middle" class="andlbl">AND</text>
  <rect x="227" y="514" width="196" height="32" class="chipbox"/>
  <text x="325" y="534" text-anchor="middle" class="chip">Retail score rising</text>
  <text x="325" y="555" text-anchor="middle" class="andlbl">AND</text>
  <rect x="227" y="562" width="196" height="26" class="chipbox"/>
  <text x="325" y="579" text-anchor="middle" class="chip">Ribbon still bullish</text>

  <!-- Card E: yellow candle (2 conditions) -->
  <rect x="465" y="432" width="220" height="122" class="card"/>
  <rect x="465" y="432" width="220" height="26" fill="var(--header-sell)" rx="8"/>
  <rect x="465" y="450" width="220" height="8" fill="var(--header-sell)"/>
  <text x="575" y="449" text-anchor="middle" class="cardhdr">Path 2 — Yellow candle</text>
  <rect x="477" y="466" width="196" height="32" class="chipbox"/>
  <text x="575" y="486" text-anchor="middle" class="chip">Down-close candle after a rally</text>
  <text x="575" y="507" text-anchor="middle" class="andlbl">AND</text>
  <rect x="477" y="514" width="196" height="32" class="chipbox"/>
  <text x="575" y="534" text-anchor="middle" class="chip">Whale score falling</text>

  <!-- Converge to OR -->
  <path d="M325,598 L325,614" class="conn"/>
  <path d="M575,554 L575,614" class="conn"/>
  <path d="M325,614 L575,614" class="conn"/>
  <text x="450" y="627" text-anchor="middle" class="orlbl">OR</text>
  <path d="M450,614 L450,644" class="arrow"/>

  <rect x="350" y="646" width="200" height="42" rx="8" fill="var(--sell)"/>
  <text x="450" y="673" text-anchor="middle" class="outlbl">SELL / trim signal</text>

  <text x="450" y="712" text-anchor="middle" class="legend">"Whale rising/falling" and "retail rising/falling" = 3-bar score delta beyond a ±2-point threshold (steady otherwise).</text>
  <text x="450" y="730" text-anchor="middle" class="legend">A ticker can show BOTH a buy and a sell in the same week if unrelated paths on each side happen to fire together.</text>
  </g>
</svg>
</div>
"""


def guide_page(demo_mode: bool) -> None:
    _freshness_section(demo_mode)
    st.divider()
    st.markdown("##### 📖 How to read this — plain-language guide (start here if you're new to investing)")
    st.markdown(
        """
**Is there a signal right now?** Look at the colored box above — it always
states outright: **🟢 BUY SIGNAL**, **🔴 SELL SIGNAL**, or **⚪ NO SIGNAL**.
Everything below explains how that's decided, in plain English — no prior
trading knowledge assumed.

---

**A few terms used on this page:**
- **Candle** — one time period's price movement (a day, week, or month,
  depending on the "Bar size" you pick), drawn as a small bar. Its color
  shows whether price went up or down that period.
- **Bullish** = a sign price may rise. **Bearish** = a sign price may fall.
- **Uptrend** = price has generally been rising lately. **Downtrend** =
  price has generally been falling lately.
- **Trend ribbon** — a band drawn from several moving averages (a moving
  average is just yesterday's noise smoothed out). When the ribbon gets
  tight/"compressing," it means those averages are converging — which
  often happens right before a bigger price move.
- **Whale score** — our estimate (0-100) of how much big investors
  ("whales") are buying a stock, built from free public data. 50 = neutral.
- **Retail score** — the same idea, but estimating regular/individual
  investor buying instead.

---

**The five things you might see:**

| Icon | What it means | What to do |
|---|---|---|
| 🟢 Buy | **This IS a buy signal** — it fired in the last 2 weeks | Consider buying gradually (a little at a time) rather than all at once |
| 🟠 Trim | **This IS a sell signal** — big investors look like they're selling | Consider selling some of your position if you own it |
| 🟡 Watch | **No signal yet** — but conditions may be building toward a buy | Just watch for now — don't buy on this alone |
| 🔵 Hold | **No signal** — price trend still looks positive | If you own it, nothing here suggests selling |
| ⚪ Wait | **No signal** — nothing stands out either way | Do nothing; check back later |

Only 🟢 Buy and 🟠 Trim are actual signals. 🟡🔵⚪ all mean **no signal is
active right now** — they just differ in what's happening in the background.

**What triggers a 🟢 BUY SIGNAL** (any one of these — always also requires
the whale score to be rising):
- price bounces back up after falling, while the trend ribbon is tight
- the trend ribbon flips from a downtrend to an uptrend
- a "MACD golden cross" — a sign momentum is turning upward (see Panel 2 below)

**What triggers a 🔴 SELL SIGNAL (Trim warning):**
- the whale score falls while the retail score rises during an uptrend —
  a pattern often seen near a price peak
- a weak candle forms (price closes near its low) while the whale score is falling

Each bullet above is its own independent path — **any one** firing is enough
to trigger the signal, but every condition *within* a path has to line up at
the same time. The diagram below shows the full logic:
"""
    )
    st.markdown(SIGNAL_DIAGRAM_SVG, unsafe_allow_html=True)
    st.markdown(
        """
**The whale-score zones** (thresholds configurable per stock): below 35 =
weak, above 35 = momentum, above 50 = rise, above 75 = soar.

---

**How to read the chart, panel by panel:**

**Panel 1 — Price & trend ribbon:** candles (defined above) with the trend
ribbon overlaid.
- **Blue** ribbon = downtrend. **Red** ribbon = uptrend.
- 🟢/🔴 triangles mark exactly where a buy/sell signal fired.

**Panel 2 — MACD (Moving Average Convergence Divergence):** a separate
momentum indicator (momentum = whether price is speeding up or slowing down).
- **MACD line** (blue) and **Signal line** (orange) — when the blue line
  crosses above the orange one, that's a **golden cross** (momentum turning
  up); crossing below is a **death cross** (momentum turning down).
- A golden cross only counts toward a 🟢 buy signal when the whale score is
  *also* rising at the same time — momentum alone is never enough on its
  own. The death cross is shown for reference but doesn't currently trigger
  a 🟠 sell signal by itself.

**Panel 3 — Whale vs retail score:** the whale score (red bars) and retail
score (green bars) explained above, over time, with dashed lines at the
momentum/rise/soar zone thresholds.

**Panel 4 — Volume:** how many shares traded each period, colored to match
the trend ribbon.

---

**😱 Fear & Greed Index** (separate page, left panel) — a market-wide
gauge, not specific to any one stock. It's a **contrarian** tool: historically,
extreme fear has been a better time to buy than extreme greed, roughly the
opposite of how it feels in the moment. It's built the same way as the whale
score — free data, six weighted indicators averaged into one 0-100 number
(50 = neutral) — modeled on CNN's well-known Fear & Greed Index. Use it as
background context for the signals above, not as a signal of its own: a 🟢
Buy during Extreme Fear carries more weight than the same 🟢 Buy during
Extreme Greed.

⚠️ *These are estimates built from free public data (FINRA off-exchange
trading volume, SEC 13F filings, price/volume patterns) — no public data
feed actually labels trades as coming from institutions. Signals update
weekly. This is not financial advice.*
"""
    )


def detail_page(cfg, ticker: str, timeframe: str):
    verdict_banner(ticker, cfg.thresholds_for(ticker))

    frame = load_ticker_frame(ticker, timeframe)
    if frame.empty:
        st.warning(
            "No data cached for this ticker yet. Click **🔄 Refresh data** in the left "
            "panel — if it still comes up empty, live FINRA/EDGAR/Yahoo fetches may "
            "be failing on this host; set the `WHALETRADING_DEMO=1` secret for a "
            "reliable demo instead."
        )
        return
    thr = cfg.thresholds_for(ticker)

    # Metrics always reflect the latest data regardless of the chart's
    # selected time range — same reasoning as the Close price fix below:
    # zooming the chart into a narrower window shouldn't make the current
    # whale/retail score look "stuck" on an older value.
    latest = frame.iloc[-1]
    zone = signals.zone_label(float(latest["whale_score"]), thr)

    # Bug fix: this used to read latest['close'] off `frame`, which is
    # resampled to whatever Bar size is selected above — so on Weekly/Monthly
    # it showed last Friday's/last month's close and looked "stuck" even
    # right after a refresh. Always show the true latest daily close instead,
    # independent of the chart's bar size, plus a best-effort live quote.
    latest_daily = load_latest_daily(ticker)
    quote = None if cfg.demo_mode else load_quote(ticker)
    company = None if cfg.demo_mode else load_company_name(ticker)
    st.markdown(f"#### {ticker}" + (f" — {company}" if company else ""))

    m1, m2, m3, m4 = st.columns(4)
    if quote is not None:
        day_close = float(latest_daily["close"]) if latest_daily is not None else None
        m1.metric(
            "Price (delayed quote)",
            f"{quote:,.2f}",
            f"{quote - day_close:+.2f} vs last close" if day_close is not None else None,
            help=(
                "Yahoo Finance quote, ~15 min delayed — can reflect today's "
                "price even before the daily bar finalizes."
                + (
                    f" Last COMPLETED daily close was {day_close:,.2f} on "
                    f"{latest_daily.name:%Y-%m-%d}."
                    if day_close is not None
                    else ""
                )
            ),
        )
    elif latest_daily is not None:
        m1.metric(
            f"Close ({latest_daily.name:%Y-%m-%d})",
            f"{float(latest_daily['close']):,.2f}",
            help=(
                "Last COMPLETED daily close — independent of the Bar size "
                "selected above, so this won't look stuck on an old weekly/"
                "monthly bar. A live delayed quote isn't available "
                + ("in demo mode." if cfg.demo_mode else "right now.")
            ),
        )
    else:
        m1.metric("Close", f"{latest['close']:,.2f}", help="Last close on the selected bar size.")
    m2.metric(
        "Whale score",
        f"{latest['whale_score']:.1f}",
        f"{latest['whale_delta']:+.1f}",
        help=(
            "0-100 estimate of how much big investors (\"whales\") are "
            "buying this stock, built from free public data (FINRA "
            "dark-pool volume, SEC 13F filings, price/volume patterns). "
            "50 = neutral. The small number below is how much it's changed "
            "over the last 3 bars — this is what the signal above reacts to."
        ),
    )
    m3.metric(
        "Retail score",
        f"{latest['retail_score']:.1f}",
        f"{latest['retail_delta']:+.1f}",
        delta_color="inverse",
        help=(
            "0-100 estimate of regular/individual-investor buying — the "
            "counterpart to the whale score. 50 = neutral. Colored in "
            "reverse: a rising retail score alongside a falling whale "
            "score is the 🔴 sell-signal pattern."
        ),
    )
    m4.metric(
        "Zone",
        zone,
        help="Whale-score threshold band: weak <35 · momentum 35-50 · rise 50-75 · soar >75 (per-ticker configurable in config/watchlist.yaml).",
    )

    # Right above the chart (and its legend), for quick adjustment while
    # looking at it, rather than scrolled away above the metrics.
    range_labels = list(RANGE_PRESETS) + ["Custom"]
    default_idx = range_labels.index(DEFAULT_RANGE[timeframe])
    range_label = st.radio(
        "Time range",
        range_labels,
        horizontal=True,
        index=default_idx,
        key=f"range_{ticker}",
        help="How far back the chart displays. This only changes the view — it doesn't affect the verdict or metrics above.",
    )
    chart_frame = frame
    if range_label == "Custom":
        c1, c2 = st.columns(2)
        start = c1.date_input("From", value=frame.index.min().date(), key=f"from_{ticker}")
        end = c2.date_input("To", value=frame.index.max().date(), key=f"to_{ticker}")
        chart_frame = frame.loc[str(start) : str(end)]
    else:
        days = RANGE_PRESETS[range_label]
        if days is not None:
            cutoff = frame.index.max() - pd.Timedelta(days=days)
            chart_frame = frame[frame.index >= cutoff]

    if chart_frame.empty:
        st.warning("No bars in the selected range — widen the time range.")
        return

    st.plotly_chart(
        four_panel_figure(chart_frame, ticker, thr),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    with st.expander("Score composition & data table"):
        conn = store.connect()
        metrics = store.load_metrics(conn, ticker)
        holdings = store.load_13f(conn, ticker)
        conn.close()
        if not metrics.empty:
            comp = json.loads(metrics["components"].iloc[-1] or "{}")
            st.write(
                "Latest composite inputs (0–100, 50 = neutral): "
                + ", ".join(f"**{k}** {v}" for k, v in comp.items() if v is not None)
            )
        if not holdings.empty:
            by_period = (
                holdings.groupby("report_period")["shares"].sum().rename("tracked 13F shares")
            )
            st.write("Tracked institutional (13F) shares by quarter:")
            st.dataframe(
                by_period.to_frame(),
                use_container_width=True,
                height=_table_height(len(by_period)),
            )
        st.caption(
            f"Full cached history for this bar size (up to {cfg.price_lookback_years} "
            "years) — independent of the Time range picked above, which only zooms the "
            "chart. Use the filters below to jump straight to every past buy/sell signal "
            "and check them against what actually happened afterward."
        )
        fcol1, fcol2 = st.columns(2)
        show_buy = fcol1.checkbox("🟢 Buy signal rows only", key="data_table_show_buy")
        show_sell = fcol2.checkbox("🔴 Sell signal rows only", key="data_table_show_sell")

        full = frame[
            ["close", "volume", "whale_score", "retail_score",
             "buy_signal", "buy_reason", "sell_signal", "sell_reason"]
        ]
        if show_buy and show_sell:
            full = full[full["buy_signal"] | full["sell_signal"]]
        elif show_buy:
            full = full[full["buy_signal"]]
        elif show_sell:
            full = full[full["sell_signal"]]

        if full.empty:
            st.info("No rows match the current filter.")
        else:
            # Capped, not sized to fit every row (_table_height's usual
            # no-inner-scrollbar approach) — up to ~1,260 daily bars for a
            # 5-year lookback would otherwise stretch the page absurdly
            # tall. st.dataframe's own scrollbar, sort, search, and CSV
            # export (toolbar, top-right) are exactly the right tools here.
            st.dataframe(
                full,
                use_container_width=True,
                height=min(_table_height(len(full)), 600),
            )


def go_to_ticker(ticker: str) -> None:
    """Jump the Overview table's 📈 button straight to that stock's chart.

    A widget's session_state key can't be reassigned after that widget has
    already been instantiated in the same run — so we stash the request and
    apply it at the very top of the next run, before the sidebar's nav
    buttons / ticker selectbox are created.
    """
    st.session_state["pending_nav"] = ("detail", ticker)
    st.rerun()


def main():
    cfg = get_config()
    working = get_working_watchlist(cfg)
    st.session_state.setdefault("current_page", "overview")
    st.session_state.setdefault("detail_ticker", working[0] if working else None)
    st.session_state.setdefault("bar_size_label", "Weekly")
    pending = st.session_state.pop("pending_nav", None)
    if pending:
        st.session_state["current_page"], st.session_state["detail_ticker"] = pending

    with st.spinner("First run — fetching data (FINRA / EDGAR / prices)…"):
        bootstrap_data(cfg)
    auto_refresh_if_stale(cfg)

    def _ticker_option(t: str) -> str:
        if cfg.demo_mode:
            return t
        name = load_company_name(t)
        return f"{t} — {name}" if name else t

    # Left panel: watchlist switcher, Refresh button, then link-style nav
    # buttons (no radio bullets), with the Ticker/Bar-size controls nested
    # directly under "Ticker detail" since they only matter for that page.
    # Collapsible and resizable by dragging its right edge — native
    # Streamlit sidebar behavior, nothing custom needed for that part.
    with st.sidebar:
        st.selectbox(
            "Watchlist", list(cfg.watchlists), key="active_watchlist",
            help=(
                "Which watchlist is active — sets what the Watchlist page "
                "shows, what the Ticker detail dropdown offers, and which "
                "list 🔄 Refresh data touches."
            ),
        )
        active_name = st.session_state["active_watchlist"]
        conn = store.connect()
        last = store.get_meta(conn, f"last_refresh:pipeline:{active_name}")
        conn.close()
        if st.button(
            "🔄 Refresh data",
            type="primary",
            use_container_width=True,
            help=(
                "Re-fetches FINRA (dark-pool volume), SEC EDGAR (13F holdings), "
                "and Yahoo Finance (prices) for every ticker in the *active* "
                "watchlist above, then recomputes whale/retail scores. Other "
                "watchlists are untouched — switch to one and refresh it "
                "separately. First run for a list can take a few minutes; "
                "later refreshes are incremental."
            ),
        ):
            with st.spinner(f"Fetching FINRA / EDGAR / prices for “{active_name}”…"):
                summary = refresh_all(cfg, tickers=get_working_watchlist(cfg))
                _mark_watchlist_refreshed(active_name)
            failed = summary.get("sources_failed", [])
            if failed:
                st.warning("Unavailable sources: " + ", ".join(failed))
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Last refresh of “{active_name}”: {_format_singapore(last)}")
        st.divider()

        # Collect a page switch and apply st.rerun() only after every widget
        # in this loop (including the Ticker selectbox below) has rendered
        # for this run. Streamlit drops a keyed widget's session_state entry
        # for any run that doesn't instantiate it — calling st.rerun() early
        # (e.g. right inside the button's own branch) would abort the script
        # before the selectbox below ever renders, silently resetting
        # detail_ticker back to the first ticker on the next run.
        pending_page = None
        for key, label in PAGES.items():
            active = st.session_state["current_page"] == key
            if st.button(
                label, key=f"nav_{key}", use_container_width=True,
                type="primary" if active else "secondary",
            ):
                pending_page = key
            if key == "detail" and working:
                if st.session_state.get("detail_ticker") not in working:
                    st.session_state["detail_ticker"] = working[0]
                with st.container(key="ticker_detail_nav"):
                    st.selectbox(
                        "Ticker", working, key="detail_ticker",
                        label_visibility="collapsed", format_func=_ticker_option,
                    )
                    st.radio(
                        "Bar size", list(TIMEFRAMES), horizontal=True,
                        key="bar_size_label", label_visibility="collapsed",
                        help="How much time each bar on the chart covers. The buy/sell verdict always uses Weekly, regardless of this setting.",
                    )
        if pending_page:
            st.session_state["current_page"] = pending_page
            st.rerun()

    st.markdown(f"### 🐋 WhaleTrading{_fg_chip_html()}", unsafe_allow_html=True)
    st.caption(
        "Institutional (whale) accumulation tracker on free data. See 📖 How "
        "to read this in the left panel for data freshness and a plain-"
        "language guide."
    )

    if cfg.demo_mode:
        st.info(
            "🎭 **Preview mode — every number on this page is a made-up example, not "
            "real market data.** To switch to real prices and real signals: open this "
            "app on **Streamlit Cloud → Settings → Secrets**, delete the "
            "`WHALETRADING_DEMO` line, then reboot the app."
        )

    if not working:
        ticker, timeframe = None, "W"
    else:
        ticker = st.session_state.get("detail_ticker")
        timeframe = TIMEFRAMES[st.session_state.get("bar_size_label", "Weekly")]

    st.divider()
    page = st.session_state["current_page"]
    if page == "overview":
        overview_page(cfg)
    elif page == "feargreed":
        fear_greed_page(cfg)
    elif page == "detail":
        if ticker:
            detail_page(cfg, ticker, timeframe)
        else:
            st.warning("Your watchlist is empty. Click ✏️ Edit on the Watchlist Overview page to add a ticker.")
    elif page == "guide":
        guide_page(cfg.demo_mode)


main()
