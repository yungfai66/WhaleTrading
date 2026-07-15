"""WhaleTrading dashboard.

Run:  streamlit run app.py
Demo: WHALETRADING_DEMO=1 streamlit run app.py   (synthetic data, no network)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from whaletrading import signals
from whaletrading.config import load_config
from whaletrading.data import prices as prices_mod
from whaletrading.data import store
from whaletrading.pipeline import refresh_all

# Streamlit Cloud secrets don't auto-populate os.environ — bridge the one
# setting we read that way. No-op locally (no secrets.toml) or off Cloud.
try:
    if "WHALETRADING_DEMO" in st.secrets:
        os.environ.setdefault("WHALETRADING_DEMO", str(st.secrets["WHALETRADING_DEMO"]))
except Exception:
    pass

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
}

# Date-range presets for the ticker detail chart: label -> calendar days back
# (None = show everything). Default preset differs by bar granularity so a
# fresh page load looks like a reasonable default window.
RANGE_PRESETS = {
    "1M": 30, "3M": 91, "6M": 182, "1Y": 365, "2Y": 730, "5Y": 1826, "All": None,
}
DEFAULT_RANGE = {"D": "1Y", "W": "2Y", "M": "All"}

NAV_OPTIONS = ["📊 Overview", "📈 Ticker detail"]


def get_working_watchlist(cfg) -> list[str]:
    """The ticker list actually shown, after this session's pins/adds/
    removes/reordering. Starts as a copy of cfg.watchlist. Session-scoped
    only (see manage_watchlist_panel) — resets on page reload, since the
    app's filesystem is ephemeral on Streamlit Cloud and silently rewriting
    config/watchlist.yaml would look like it worked locally but lose the
    change on every redeploy there."""
    st.session_state.setdefault("watchlist_order", list(cfg.watchlist))
    st.session_state.setdefault("pinned_tickers", set())
    return st.session_state["watchlist_order"]


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


st.set_page_config(page_title="WhaleTrading", page_icon="🐋", layout="wide")

# Narrow default sidebar width (Streamlit's default is ~336px). The sidebar
# only holds the refresh button + a config hint, so it doesn't need the
# space the main charts do. Users can still drag it wider if they want —
# this only changes the default.
#
# Also trims Streamlit's large default top padding and divider margins,
# which otherwise stack up into a noticeable gap between the title and the
# "Watchlist overview" section below the nav row.
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
    .stButton button { padding: 0.25rem 0.75rem !important; }
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
    empty dashboard. Runs at most once per container lifecycle (cache_resource)."""
    conn = store.connect()
    has_data = conn.execute("SELECT 1 FROM metrics LIMIT 1").fetchone() is not None
    conn.close()
    if has_data:
        return False
    refresh_all(_cfg)
    return True


# How stale the cache must be (minutes since last pipeline refresh) before a
# new page load/reload triggers an automatic refresh. Session-scoped (not
# st.cache_resource), so it fires once per browser session/reload rather than
# once per container lifetime — without this a re-opened tab could sit on a
# refresh from hours ago until someone remembers to click the button.
AUTO_REFRESH_STALE_MINUTES = 30


def auto_refresh_if_stale(cfg) -> None:
    if st.session_state.get("_auto_refresh_done"):
        return
    st.session_state["_auto_refresh_done"] = True
    conn = store.connect()
    last = store.get_meta(conn, "last_refresh:pipeline")
    conn.close()
    if last:
        age_min = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() / 60
        if age_min < AUTO_REFRESH_STALE_MINUTES:
            return
    with st.spinner("Refreshing data…"):
        refresh_all(cfg, tickers=get_working_watchlist(cfg))
    st.cache_data.clear()


@st.cache_data(ttl=60)
def load_quote(ticker: str) -> float | None:
    """Short-TTL cache (60s) so this stays meaningfully "current", separate
    from the 5-minute cache on the daily-bar data used for charting."""
    return prices_mod.fetch_quote(ticker)


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

        # act['detail'] already states "A buy/sell signal appeared this week: ..."
        # when the latest bar just fired one — no need to repeat it here.
        fresh_msg = ""
        if not weekly.empty:
            latest_bar = weekly.iloc[-1]
            if bool(latest_bar["sell_signal"]):
                fresh_msg = f"🔴 **{ticker}** — {act['detail']}"
            elif bool(latest_bar["buy_signal"]):
                fresh_msg = f"🟢 **{ticker}** — {act['detail']}"

        rows.append(
            {
                "Ticker": ticker,
                "Action": act["label"],
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
                "Zone": signals.zone_label(float(last["whale_score"]), thr),
                "Status": "ok",
                "_sort": signals.ACTION_ORDER.index(act["action"]),
                "_fresh_msg": fresh_msg,
                "_severity": act["severity"],
            }
        )
    conn.close()
    return pd.DataFrame(rows)


def four_panel_figure(frame: pd.DataFrame, ticker: str, thresholds: dict) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.45, 0.12, 0.25, 0.18],
        subplot_titles=(
            "Price · EMA ribbon · signals",
            "Volume (trend-colored)",
            "Whale (red) vs retail (green) accumulation",
            "MACD",
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
                name="Trend ribbon (5 EMAs, dark→light = short→long)",
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
            line=dict(color=C["violet"], width=1.6),
            name="Close price (line)",
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
            name="Candle (open/high/low/close)",
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

    # ── Panel 2: volume colored by ribbon trend ─────────────────────────
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
        row=2,
        col=1,
    )

    # ── Panel 3: whale vs retail bars + threshold guides ────────────────
    fig.add_trace(
        go.Bar(
            x=frame.index,
            y=frame["whale_score"],
            marker_color=C["whale"],
            name="Whale accumulation",
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

    # ── Panel 4: MACD ───────────────────────────────────────────────────
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
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index, y=frame["macd"], mode="lines",
            line=dict(color=C["blue"], width=2), name="MACD",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index, y=frame["macd_signal"], mode="lines",
            line=dict(color=C["orange"], width=2), name="Signal",
        ),
        row=4,
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
            hovertemplate="Golden cross %{x|%Y-%m-%d}<br>momentum turning up<extra></extra>",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=death.index, y=death["macd"], mode="markers",
            marker=dict(symbol="triangle-down", size=9, color=C["muted"],
                        line=dict(color=C["surface"], width=1)),
            name="Death cross",
            hovertemplate="Death cross %{x|%Y-%m-%d}<br>momentum turning down (not wired to a verdict)<extra></extra>",
        ),
        row=4,
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

    fig.update_layout(
        height=980,
        barmode="group",
        bargap=0.25,
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(color=C["ink2"], family='system-ui, -apple-system, "Segoe UI", sans-serif'),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0, font=dict(size=10)),
        margin=dict(l=40, r=20, t=90, b=30),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig


@st.cache_data(ttl=300)
def load_freshness() -> dict:
    conn = store.connect()
    fresh = store.source_freshness(conn)
    conn.close()
    return fresh


def freshness_panel(demo_mode: bool) -> None:
    """Data validity: latest data point + inherent delay + stale flag per source."""
    fresh = load_freshness()
    with st.expander("🕐 Data freshness & validity — how current is what you're seeing?"):
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
            "cycle — hit **Refresh data now** in the sidebar. \"Inherent delay\" "
            "is how old the information is even right after a refresh: prices "
            "and dark-pool volume describe **yesterday/today**, weekly ATS "
            "(Alternative Trading System) describes **2–4 weeks ago**, and 13F "
            "(SEC Form 13F, quarterly institutional-holdings filing) holdings "
            "describe **last quarter**. Signals weight the fast sources most, "
            "and 13F least. FINRA = Financial Industry Regulatory Authority "
            "(publishes the dark-pool data); SEC = Securities and Exchange "
            "Commission (publishes 13F)."
        )


def manage_watchlist_panel(cfg) -> None:
    """Pin/add/remove/reorder tickers for the current browser session."""
    working = get_working_watchlist(cfg)
    pinned = st.session_state["pinned_tickers"]

    # key= makes Streamlit persist open/closed state across reruns — without
    # it, every pin/add/move/delete click (each triggers a rerun) would snap
    # the expander shut again right after the user opened it.
    with st.expander(
        "⚙️ Manage watchlist — pin, add, remove, reorder", key="manage_watchlist_expander"
    ):
        # A widget's session_state value can't be reassigned after that
        # widget has already been instantiated in the same run — so clearing
        # the text input after "Add" needs the same stash-then-rerun pattern
        # used for cross-page navigation, applied *before* the widget below
        # is created.
        if st.session_state.pop("_clear_add_ticker_input", False):
            st.session_state["add_ticker_input"] = ""

        st.caption(
            "📌 Pin a stock to keep it at the top of the table. Changes here "
            "apply only to **your current browser session** and reset if you "
            "reload the page — for permanent changes, edit "
            "`config/watchlist.yaml` in the repo instead. After **adding** a "
            "ticker, click **Refresh data now** in the sidebar to fetch its "
            "data before it shows up with real numbers."
        )
        c1, c2 = st.columns([4, 1], gap="small")
        new_ticker = c1.text_input(
            "Add a ticker symbol",
            key="add_ticker_input",
            label_visibility="collapsed",
            placeholder="Add a ticker, e.g. GOOG",
        )
        if c2.button("➕ Add", use_container_width=True) and new_ticker.strip():
            sym = new_ticker.strip().upper()
            if sym in working:
                st.warning(f"{sym} is already in your watchlist.")
            else:
                working.append(sym)
                st.session_state["_clear_add_ticker_input"] = True
                st.rerun()

        if not working:
            st.warning("Your watchlist is empty — add a ticker above.")
            return

        ROW_COLS = [0.6, 2.2, 0.55, 0.55, 0.55, 0.55, 0.6]
        hdr = st.columns(ROW_COLS, gap="small")
        hdr[0].caption("Pin")
        hdr[1].caption("Ticker")
        hdr[2].caption("⤒")
        hdr[3].caption("↑")
        hdr[4].caption("↓")
        hdr[5].caption("⤓")
        hdr[6].caption("Del")
        for i, ticker in enumerate(working):
            row = st.columns(ROW_COLS, gap="small")
            is_pinned = row[0].checkbox(
                "Pin", value=ticker in pinned, key=f"pin_{ticker}", label_visibility="collapsed"
            )
            if is_pinned:
                pinned.add(ticker)
            else:
                pinned.discard(ticker)
            row[1].write(f"**{ticker}**")
            if row[2].button("⤒", key=f"top_{ticker}", disabled=(i == 0), help="Move to top"):
                working.insert(0, working.pop(i))
                st.rerun()
            if row[3].button("↑", key=f"up_{ticker}", disabled=(i == 0), help="Move up"):
                working[i - 1], working[i] = working[i], working[i - 1]
                st.rerun()
            if row[4].button("↓", key=f"down_{ticker}", disabled=(i == len(working) - 1), help="Move down"):
                working[i + 1], working[i] = working[i], working[i + 1]
                st.rerun()
            if row[5].button("⤓", key=f"bottom_{ticker}", disabled=(i == len(working) - 1), help="Move to bottom"):
                working.append(working.pop(i))
                st.rerun()
            if row[6].button("🗑️", key=f"del_{ticker}", help=f"Remove {ticker} from your watchlist"):
                working.remove(ticker)
                pinned.discard(ticker)
                if st.session_state.get("detail_ticker") == ticker:
                    st.session_state["detail_ticker"] = working[0] if working else None
                st.rerun()


def overview_page(cfg):
    st.subheader("Watchlist overview")
    freshness_panel(cfg.demo_mode)
    manage_watchlist_panel(cfg)

    working = get_working_watchlist(cfg)
    pinned = st.session_state["pinned_tickers"]
    if not working:
        return
    df = load_overview(tuple(working))
    ok = df[df["Status"] == "ok"].copy()
    ok["_pinned"] = ok["Ticker"].isin(pinned)
    missing = df[df["Status"] != "ok"]
    if not ok.empty:
        fresh = [m for m in ok["_fresh_msg"] if m]
        if fresh:
            st.markdown("##### 🔔 Buy/Sell Signals This Week")
            for msg in fresh:
                (st.warning if msg.startswith("🔴") else st.success)(msg)
        else:
            st.caption("🔔 No stock has an active buy or sell signal this week.")

        display = ok.sort_values(
            ["_pinned", "_sort", "Whale %"], ascending=[False, True, False]
        ).reset_index(drop=True)

        st.caption("💡 Click 📈 to open a stock's chart. Pinned stocks (📌) stay on top.")
        # A native Streamlit row list, not st.dataframe: avoids the built-in
        # row-selection checkbox (confusing — looked like a "pin" toggle) and
        # guarantees no nested scrollbar, since plain elements never scroll
        # internally the way a data-grid does.
        ROW_COLS = [0.35, 0.8, 1.1, 0.8, 1.6, 0.7, 0.8, 0.8, 0.45]
        HEADERS = [
            ("📌", "Pinned in 'Manage watchlist' above — stays sorted to the top."),
            ("Ticker", "The stock symbol."),
            ("Action", "🟢 Buy = buy signal · 🟠 Trim = sell signal · 🟡 Watch = building toward a buy · 🔵 Hold = trend positive, no signal · ⚪ Wait = nothing stands out"),
            ("Close", "Last COMPLETED daily close (not a live quote)."),
            ("Whale %", "0-100 estimate of big-investor buying (FINRA dark-pool volume + SEC 13F + price/volume patterns). 50 = neutral."),
            ("Δ 20d", "Change in Whale % over the last ~20 trading days. Positive = buying increasing."),
            ("Retail %", "0-100 estimate of regular/individual-investor buying. 50 = neutral."),
            ("Zone", "weak <35 · momentum 35-50 · rise 50-75 · soar >75."),
            ("📈", "Open this stock's full chart."),
        ]
        hdr = st.columns(ROW_COLS, gap="small")
        for c, (label, tip) in zip(hdr, HEADERS):
            c.markdown(f'<span title="{tip}" style="font-size:0.8rem;color:#898781;">{label}</span>', unsafe_allow_html=True)
        for _, r in display.iterrows():
            cells = st.columns(ROW_COLS, gap="small")
            cells[0].write("📌" if r["_pinned"] else "")
            cells[1].markdown(f"**{r['Ticker']}**")
            cells[2].write(r["Action"])
            cells[3].write(f"{r['Close']:,.2f}")
            wv = float(r["Whale %"])
            filled = int(round(wv / 10))
            bar = "▓" * filled + "░" * (10 - filled)
            cells[4].markdown(f"`{bar}` {wv:.1f}")
            d20 = r["Δ 20d"]
            cells[5].write("—" if d20 is None or pd.isna(d20) else f"{d20:+.1f}")
            cells[6].write(f"{float(r['Retail %']):.1f}")
            cells[7].write(r["Zone"])
            if cells[8].button("📈", key=f"chart_{r['Ticker']}", help=f"Open {r['Ticker']}'s chart"):
                go_to_ticker(r["Ticker"])
        st.caption(
            "**Action guide:** 🟢 Buy = this IS a buy signal, consider buying "
            "gradually (DCA, Dollar-Cost Averaging) rather than all at once · "
            "🟠 Trim = this IS a sell signal — big investors look like they're "
            "selling, consider taking some profit · "
            "🟡 Watch = no signal yet, but big-investor buying is rising and "
            "conditions may be building toward a buy signal · "
            "🔵 Hold = no signal, price trend still looks positive · "
            "⚪ Wait = no signal, nothing stands out right now. "
            "Zones: momentum >35, rise >50, soar >75 (per-ticker configurable). "
            "Hover any column header above for details, or click a row for "
            "the full chart and guide."
        )
    if not missing.empty:
        st.warning(
            "No cached data for: "
            + ", ".join(missing["Ticker"])
            + ". Run a data refresh (sidebar), or check the ticker symbol."
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
            "(see 🕐 Data freshness on the Overview tab for per-source delays)."
        )
    if caption_bits:
        st.caption("  \n".join(caption_bits))


def how_to_read_expander() -> None:
    with st.expander("📖 How to read this — plain-language guide (start here if you're new to investing)"):
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
- a "MACD golden cross" — a sign momentum is turning upward (see Panel 4 below)

**What triggers a 🔴 SELL SIGNAL (Trim warning):**
- the whale score falls while the retail score rises during an uptrend —
  a pattern often seen near a price peak
- a weak candle forms (price closes near its low) while the whale score is falling

**The whale-score zones** (thresholds configurable per stock): below 35 =
weak, above 35 = momentum, above 50 = rise, above 75 = soar.

---

**How to read the chart, panel by panel:**

**Panel 1 — Price & trend ribbon:** candles (defined above) with the trend
ribbon overlaid.
- **Blue** ribbon = downtrend. **Red** ribbon = uptrend.
- 🟢/🔴 triangles mark exactly where a buy/sell signal fired.

**Panel 2 — Volume:** how many shares traded each period, colored to match
the trend ribbon.

**Panel 3 — Whale vs retail score:** the whale score (red bars) and retail
score (green bars) explained above, over time, with dashed lines at the
momentum/rise/soar zone thresholds.

**Panel 4 — MACD (Moving Average Convergence Divergence):** a separate
momentum indicator (momentum = whether price is speeding up or slowing down).
- **MACD line** (blue) and **Signal line** (orange) — when the blue line
  crosses above the orange one, that's a **golden cross** (momentum turning
  up); crossing below is a **death cross** (momentum turning down).
- A golden cross only counts toward a 🟢 buy signal when the whale score is
  *also* rising at the same time — momentum alone is never enough on its
  own. The death cross is shown for reference but doesn't currently trigger
  a 🟠 sell signal by itself.

⚠️ *These are estimates built from free public data (FINRA off-exchange
trading volume, SEC 13F filings, price/volume patterns) — no public data
feed actually labels trades as coming from institutions. Signals update
weekly. This is not financial advice.*
"""
        )


def detail_page(cfg, ticker: str, timeframe: str):
    verdict_banner(ticker, cfg.thresholds_for(ticker))
    how_to_read_expander()

    frame = load_ticker_frame(ticker, timeframe)
    if frame.empty:
        st.warning(
            "No data cached for this ticker yet. Click **🔄 Refresh** at the top of "
            "the page — if it still comes up empty, live FINRA/EDGAR/Yahoo fetches may "
            "be failing on this host; set the `WHALETRADING_DEMO=1` secret for a "
            "reliable demo instead."
        )
        return

    range_labels = list(RANGE_PRESETS) + ["Custom"]
    default_idx = range_labels.index(DEFAULT_RANGE[timeframe])
    range_label = st.radio(
        "Time range",
        range_labels,
        horizontal=True,
        index=default_idx,
        key=f"range_{ticker}",
        help="How far back the chart displays. This only changes the view — it doesn't affect the verdict above.",
    )
    if range_label == "Custom":
        c1, c2 = st.columns(2)
        start = c1.date_input("From", value=frame.index.min().date(), key=f"from_{ticker}")
        end = c2.date_input("To", value=frame.index.max().date(), key=f"to_{ticker}")
        frame = frame.loc[str(start) : str(end)]
    else:
        days = RANGE_PRESETS[range_label]
        if days is not None:
            cutoff = frame.index.max() - pd.Timedelta(days=days)
            frame = frame[frame.index >= cutoff]

    if frame.empty:
        st.warning("No bars in the selected range — widen the time range.")
        return
    thr = cfg.thresholds_for(ticker)

    latest = frame.iloc[-1]
    zone = signals.zone_label(float(latest["whale_score"]), thr)

    # Bug fix: this used to read latest['close'] off `frame`, which is
    # resampled to whatever Bar size is selected above — so on Weekly/Monthly
    # it showed last Friday's/last month's close and looked "stuck" even
    # right after a refresh. Always show the true latest daily close instead,
    # independent of the chart's bar size, plus a best-effort live quote.
    latest_daily = load_latest_daily(ticker)
    quote = None if cfg.demo_mode else load_quote(ticker)

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

    st.plotly_chart(
        four_panel_figure(frame, ticker, thr),
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
        tail = frame[
            ["close", "volume", "whale_score", "retail_score",
             "buy_signal", "buy_reason", "sell_signal", "sell_reason"]
        ].tail(50)
        st.dataframe(
            tail,
            use_container_width=True,
            height=_table_height(len(tail)),
        )


def go_to_ticker(ticker: str) -> None:
    """Jump the Overview table's row-click straight to that stock's chart.

    st.tabs can't be switched from code, so navigation uses a top-level
    st.radio (still always visible, unlike the sidebar which auto-collapses
    on narrow embeds) bound to session_state — but a widget's session_state
    key can't be reassigned after that widget has already been instantiated
    in the same run. So we stash the request and apply it at the very top
    of the next run, before the radio/selectbox widgets are created.
    """
    st.session_state["pending_nav"] = (NAV_OPTIONS[1], ticker)
    st.rerun()


def main():
    cfg = get_config()
    working = get_working_watchlist(cfg)
    st.session_state.setdefault("nav_view", NAV_OPTIONS[0])
    st.session_state.setdefault("detail_ticker", working[0] if working else None)
    st.session_state.setdefault("bar_size_label", "Weekly")
    pending = st.session_state.pop("pending_nav", None)
    if pending:
        st.session_state["nav_view"], st.session_state["detail_ticker"] = pending

    with st.spinner("First run — fetching data (FINRA / EDGAR / prices)…"):
        bootstrap_data(cfg)
    auto_refresh_if_stale(cfg)

    title_col, refresh_col = st.columns([3, 1.3], gap="small")
    with title_col:
        st.markdown("### 🐋 WhaleTrading")
        st.caption(
            "Institutional (whale) accumulation tracker on free data. See 🕐 Data "
            "freshness and 📖 How to read this for definitions."
        )
    with refresh_col:
        conn = store.connect()
        last = store.get_meta(conn, "last_refresh:pipeline")
        conn.close()
        if st.button(
            "🔄 Refresh data",
            type="primary",
            use_container_width=True,
            help=(
                "Re-fetches FINRA (dark-pool volume), SEC EDGAR (13F holdings), "
                "and Yahoo Finance (prices) for every watchlist ticker, then "
                "recomputes whale/retail scores. First run can take a few "
                "minutes; later refreshes are incremental."
            ),
        ):
            with st.spinner("Fetching FINRA / EDGAR / prices…"):
                summary = refresh_all(cfg, tickers=get_working_watchlist(cfg))
            failed = summary.get("sources_failed", [])
            if failed:
                st.warning("Unavailable sources: " + ", ".join(failed))
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Last refresh: {_format_singapore(last)}")

    if cfg.demo_mode:
        st.info(
            "🎭 **Preview mode — every number on this page is a made-up example, not "
            "real market data.** To switch to real prices and real signals: open this "
            "app on **Streamlit Cloud → Settings → Secrets**, delete the "
            "`WHALETRADING_DEMO` line, then reboot the app."
        )

    if not working:
        st.warning("Your watchlist is empty. Switch to Overview → Manage watchlist to add a ticker.")
        view = NAV_OPTIONS[0]
        ticker, timeframe = None, "W"
    else:
        # Persistent top bar: view toggle, ticker, and bar size are always
        # visible together, on both pages — so you can line up the next
        # stock's chart while still looking at the Overview table, then flip
        # the toggle, instead of hunting for these controls inside the page.
        nav_col, tk_col, bs_col = st.columns([1.5, 1.2, 1.5], gap="small")
        view = nav_col.radio(
            "View", NAV_OPTIONS, horizontal=True, key="nav_view", label_visibility="collapsed"
        )
        if st.session_state.get("detail_ticker") not in working:
            st.session_state["detail_ticker"] = working[0]
        ticker = tk_col.selectbox("Ticker", working, key="detail_ticker", label_visibility="collapsed")
        tf_label = bs_col.radio(
            "Bar size",
            list(TIMEFRAMES),
            horizontal=True,
            key="bar_size_label",
            label_visibility="collapsed",
            help="How much time each bar on the chart covers. The buy/sell verdict always uses Weekly, regardless of this setting.",
        )
        timeframe = TIMEFRAMES[tf_label]

    st.divider()
    if view == NAV_OPTIONS[0]:
        overview_page(cfg)
    elif ticker:
        detail_page(cfg, ticker, timeframe)


main()
