"""WhaleTrading dashboard.

Run:  streamlit run app.py
Demo: WHALETRADING_DEMO=1 streamlit run app.py   (synthetic data, no network)
"""

from __future__ import annotations

import json
import os

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

st.set_page_config(page_title="WhaleTrading", page_icon="🐋", layout="wide")


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

        fresh_msg = ""
        if not weekly.empty:
            latest_bar = weekly.iloc[-1]
            if bool(latest_bar["sell_signal"]):
                fresh_msg = f"🟠 **{ticker}**: sell warning this week — {act['detail']}"
            elif bool(latest_bar["buy_signal"]):
                fresh_msg = f"🟢 **{ticker}**: buy setup this week — {act['detail']}"

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
        vertical_spacing=0.03,
        row_heights=[0.45, 0.12, 0.25, 0.18],
        subplot_titles=(
            "Price · EMA ribbon · signals",
            "Volume (trend-colored)",
            "Whale (red) vs retail (green) accumulation",
            "MACD",
        ),
    )

    # ── Panel 1: candles + EMA ribbon + signal markers ──────────────────
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Price",
            increasing_line_color=C["up"],
            increasing_fillcolor=C["up"],
            decreasing_line_color=C["down"],
            decreasing_fillcolor=C["down"],
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    ema_cols = [c for c in frame.columns if c.startswith("ema_")]
    for color, col in zip(RIBBON_STEPS, ema_cols):
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[col],
                mode="lines",
                line=dict(color=color, width=1),
                name=col.upper().replace("_", " "),
                hoverinfo="skip",
                showlegend=False,
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

    fig.update_layout(
        height=920,
        barmode="group",
        bargap=0.25,
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(color=C["ink2"], family='system-ui, -apple-system, "Segoe UI", sans-serif'),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=60, b=30),
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "⚠️ means the cache is older than the source's normal publication "
            "cycle — hit **Refresh data now** in the sidebar. \"Inherent delay\" "
            "is how old the information is even right after a refresh: prices "
            "and dark-pool volume describe **yesterday/today**, weekly ATS "
            "describes **2–4 weeks ago**, and 13F holdings describe **last "
            "quarter**. Signals weight the fast sources most, and 13F least."
        )


def overview_page(cfg):
    st.subheader("Watchlist overview")
    freshness_panel(cfg.demo_mode)
    df = load_overview(tuple(cfg.watchlist))
    ok = df[df["Status"] == "ok"]
    missing = df[df["Status"] != "ok"]
    if not ok.empty:
        fresh = [m for m in ok["_fresh_msg"] if m]
        if fresh:
            st.markdown("##### 🔔 Fresh signals this week")
            for msg in fresh:
                (st.warning if msg.startswith("🟠") else st.success)(msg)
        else:
            st.caption("🔔 No fresh buy/sell signals this week.")

        display = ok.sort_values(
            ["_sort", "Whale %"], ascending=[True, False]
        ).drop(columns=["Status", "_sort", "_fresh_msg", "_severity"])
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Whale %": st.column_config.ProgressColumn(
                    "Whale %", min_value=0, max_value=100, format="%.1f"
                ),
            },
        )
        st.caption(
            "**Action guide:** 🟢 Buy = entry setup fired, consider DCA · "
            "🟠 Trim = whales look like they're selling, consider taking profit · "
            "🟡 Watch = whales accumulating, wait for the entry candle · "
            "🔵 Hold = uptrend intact · ⚪ Wait = no edge. "
            "Zones: momentum >35, rise >50, soar >75 (per-ticker configurable). "
            "Open a stock's page (Ticker detail tab) for the full explanation."
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
    so it matches the overview table regardless of the selected bar size."""
    weekly = load_ticker_frame(ticker, "W")
    act = signals.current_action(weekly, thr)
    banner = BANNER_BY_SEVERITY.get(act["severity"], st.info)
    emoji = act["label"].split()[0]
    banner(f"**{emoji} {act['headline']}**\n\n{act['detail']}")
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
    with st.expander("📖 How to read this — plain-language guide"):
        st.markdown(
            """
**The one-line answer:** buy when whales are accumulating *and* the chart
confirms it with an entry candle; take profit when whales quietly hand their
shares to retail buyers near a top.

**The five actions** (shown on the overview and in the banner above):

| Action | Meaning | What to do |
|---|---|---|
| 🟢 Buy | An entry setup fired in the last 2 weeks | Consider dollar-cost averaging in — don't chase all at once |
| 🟠 Trim | Whales look like they're selling to retail | Consider taking some profit / tightening stops |
| 🟡 Watch | Whales accumulating, but no entry candle yet | Wait for confirmation before buying |
| 🔵 Hold | Uptrend intact, whales still positioned | Sit tight, don't over-trade |
| ⚪ Wait | No edge either way | Do nothing; re-check later |

**What makes a 🟢 Buy signal** (any one of these, always requiring whale
accumulation to be rising or retail falling):
- a strong bounce candle during a downtrend while the trend ribbon is compressing
- the trend ribbon flipping bullish
- a MACD golden cross

**What makes a 🟠 Trim warning:**
- whale accumulation falling while retail buying rises during an uptrend
  (the classic hand-off at a top)
- a weak candle with whales pulling back

**The whale score zones** (thresholds configurable per stock): below 35 =
weak, above 35 = momentum, above 50 = rise, above 75 = soar.

⚠️ *These are proxies built from free public data (FINRA off-exchange volume,
SEC 13F filings, volume patterns) — no feed truly labels trades as
institutional. Weekly signals; not financial advice.*
"""
        )


def detail_page(cfg):
    col1, col2 = st.columns([2, 1])
    ticker = col1.selectbox("Ticker", cfg.watchlist)
    tf_label = col2.radio("Bar size", list(TIMEFRAMES), horizontal=True, index=1)
    timeframe = TIMEFRAMES[tf_label]

    verdict_banner(ticker, cfg.thresholds_for(ticker))
    how_to_read_expander()

    frame = load_ticker_frame(ticker, timeframe)
    if frame.empty:
        st.warning(
            "No data cached for this ticker yet. Click **Refresh data now** in the "
            "sidebar — if it still comes up empty, live FINRA/EDGAR/Yahoo fetches may "
            "be failing on this host; set the `WHALETRADING_DEMO=1` secret for a "
            "reliable demo instead."
        )
        return

    range_labels = list(RANGE_PRESETS) + ["Custom"]
    default_idx = range_labels.index(DEFAULT_RANGE[timeframe])
    range_label = st.radio(
        "Time range", range_labels, horizontal=True, index=default_idx, key=f"range_{ticker}"
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
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Close", f"{latest['close']:,.2f}")
    m2.metric("Whale score", f"{latest['whale_score']:.1f}", f"{latest['whale_delta']:+.1f}")
    m3.metric("Retail score", f"{latest['retail_score']:.1f}", f"{latest['retail_delta']:+.1f}",
              delta_color="inverse")
    m4.metric("Zone", zone)

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
            st.dataframe(by_period.to_frame(), use_container_width=True)
        st.dataframe(
            frame[
                ["close", "volume", "whale_score", "retail_score",
                 "buy_signal", "buy_reason", "sell_signal", "sell_reason"]
            ].tail(50),
            use_container_width=True,
        )


def main():
    cfg = get_config()
    with st.spinner("First run — fetching data (FINRA / EDGAR / prices)…"):
        bootstrap_data(cfg)

    st.title("🐋 WhaleTrading")
    st.caption(
        "Institutional (whale) accumulation tracker built entirely on free data: "
        "FINRA off-exchange volume, SEC 13F filings, and volume-classified OHLCV. "
        "Scores are proxies — no public feed labels trades as institutional."
    )
    if cfg.demo_mode:
        st.info("Demo mode: synthetic data (WHALETRADING_DEMO=1). Unset it for live data.")

    with st.sidebar:
        st.header("Data")
        conn = store.connect()
        last = store.get_meta(conn, "last_refresh:pipeline")
        conn.close()
        st.write(f"Last refresh: {last[:16].replace('T', ' ') if last else 'never'} UTC")
        if st.button("Refresh data now", type="primary"):
            with st.spinner("Fetching FINRA / EDGAR / prices…"):
                summary = refresh_all(cfg)
            failed = summary.get("sources_failed", [])
            if failed:
                st.warning("Unavailable sources: " + ", ".join(failed))
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption(
            "Edit `config/watchlist.yaml` to change tickers, thresholds, "
            "weights, and tracked 13F managers."
        )

    # Top-level tabs, not sidebar nav: the sidebar auto-collapses on narrow
    # embeds (e.g. Streamlit Community Cloud), which was hiding "Ticker
    # detail" entirely. Tabs stay visible regardless of sidebar state.
    tab_overview, tab_detail = st.tabs(["📊 Overview", "📈 Ticker detail"])
    with tab_overview:
        overview_page(cfg)
    with tab_detail:
        detail_page(cfg)


main()
