"""WhaleTrading dashboard.

Run:  streamlit run app.py
Demo: WHALETRADING_DEMO=1 streamlit run app.py   (synthetic data, no network)
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from whaletrading import signals
from whaletrading.config import load_config
from whaletrading.data import prices as prices_mod
from whaletrading.data import store
from whaletrading.pipeline import refresh_all

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
MAX_BARS = {"D": 504, "W": 260, "M": 120}

st.set_page_config(page_title="WhaleTrading", page_icon="🐋", layout="wide")


@st.cache_resource
def get_config():
    return load_config()


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
        recent = weekly.tail(4) if not weekly.empty else pd.DataFrame()
        rows.append(
            {
                "Ticker": ticker,
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
                "Buy (4w)": bool(recent["buy_signal"].any()) if not recent.empty else False,
                "Sell (4w)": bool(recent["sell_signal"].any()) if not recent.empty else False,
                "Status": "ok",
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


def overview_page(cfg):
    st.subheader("Watchlist overview")
    df = load_overview(tuple(cfg.watchlist))
    ok = df[df["Status"] == "ok"].drop(columns=["Status"])
    missing = df[df["Status"] != "ok"]
    if not ok.empty:
        st.dataframe(
            ok.sort_values("Whale %", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Whale %": st.column_config.ProgressColumn(
                    "Whale %", min_value=0, max_value=100, format="%.1f"
                ),
            },
        )
        st.caption(
            "Zone thresholds (per-ticker configurable): momentum >35, rise >50, "
            "soar >75. Buy/Sell flags = any weekly signal in the last 4 weeks."
        )
    if not missing.empty:
        st.warning(
            "No cached data for: "
            + ", ".join(missing["Ticker"])
            + ". Run a data refresh (sidebar), or check the ticker symbol."
        )


def detail_page(cfg):
    col1, col2 = st.columns([2, 1])
    ticker = col1.selectbox("Ticker", cfg.watchlist)
    tf_label = col2.radio("Timeframe", list(TIMEFRAMES), horizontal=True, index=1)
    timeframe = TIMEFRAMES[tf_label]

    frame = load_ticker_frame(ticker, timeframe)
    if frame.empty:
        st.warning("No data cached for this ticker yet — run a refresh from the sidebar.")
        return
    frame = frame.tail(MAX_BARS[timeframe])
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
        page = st.radio("Page", ["Overview", "Ticker detail"])
        st.divider()
        st.caption(
            "Edit `config/watchlist.yaml` to change tickers, thresholds, "
            "weights, and tracked 13F managers."
        )

    if page == "Overview":
        overview_page(cfg)
    else:
        detail_page(cfg)


main()
