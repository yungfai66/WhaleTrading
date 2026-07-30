# 🐋 WhaleTrading

Track institutional ("whale") buying vs selling pressure for NYSE/NASDAQ stocks
using **only free, no-subscription data sources**, visualized in a 4-panel
Streamlit dashboard:

1. **Price** — candlesticks + EMA ribbon (8/13/21/34/55) + buy/sell markers
2. **Volume** — colored by ribbon trend (red = bullish ribbon, blue = bearish)
3. **Whale (red) vs retail (green) accumulation** — composite 0–100 scores with
   the strategy's 35 / 50 / 75 threshold guides
4. **MACD** — 12/26/9 with golden/death crosses

There's also a market-wide **😱 Fear & Greed Index** page — a contrarian
sentiment gauge modeled on CNN's, computed from free data. See
[Market sentiment](#market-sentiment--the-fear--greed-index) below.

## The honest disclaimer, first

**No public feed labels a trade "institutional."** Brokers (including IBKR) and
exchanges do not expose who is behind a print; real-time dark-pool buy/sell
attribution is exactly what paid products (Unusual Whales, FlowAlgo, TradeAlgo…)
charge for. What *is* free are strong proxies, and this app blends them into a
composite **whale score** per stock:

| Source | What it gives | Freshness | Cost |
|---|---|---|---|
| [FINRA Reg SHO daily files](https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files) | Per-ticker daily off-exchange (dark pool / internalizer) short + total volume | Same evening | Free, no key |
| [FINRA ATS weekly data](https://www.finra.org/finra-data/browse-catalog/otc-transparency-data) | Per-ticker weekly dark-pool (ATS) share volume | ~2–4 week delay | Free |
| [SEC EDGAR 13F filings](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=13F-HR) | Exact holdings of large managers (Vanguard, BlackRock, …) | Quarterly, 45-day lag | Free |
| Yahoo Finance (`yfinance`) | Daily OHLCV for candles, ribbons, MACD, volume classification | ~Live | Free |

### The whale score (0–100, 50 = neutral)

Weighted blend, renormalized when a source is unavailable (weights in config):

- **big_money_volume (0.45)** — days whose volume z-score exceeds a threshold
  are treated as big-money days, classified accumulation vs distribution by
  where the close lands in the bar's range (Chaikin close-location), netted
  over a rolling 20-day window.
- **dark_pool (0.35)** — FINRA daily off-exchange short-volume ratio vs its own
  60-day baseline; persistent above-baseline readings are the popular proxy for
  hidden accumulation (market makers shorting to fill large buy orders).
- **inst_13f (0.20)** — quarter-over-quarter change in shares held by the
  tracked 13F managers. Slow, but it is *actual* institutional positioning.

The **retail score** mirrors the first component on *low*-volume days — up-moves
on thin volume read as retail chasing.

### Signals (translated from the strategy notes)

Buy — any of:
- **dip reversal**: red (bullish reversal) candle + bearish ribbon tightening + whale rising / retail falling
- **ribbon turn**: red candle as the ribbon turns bullish + same confirmation
- **MACD cross**: golden cross + whale rising

Sell / trim warning:
- whale falling while retail rises during a bullish ribbon (the top pattern)
- yellow (bearish) candle + falling whale score

Zone badges per ticker: `<35` weak · `35–50` momentum · `50–75` rise · `>75` soar
— thresholds are per-ticker configurable, since *"different stocks require
different percentages of whale accumulation."*

### How to act on the signals

Every stock gets one plain-language verdict (overview table + banner on its
detail page), decided from the weekly chart in priority order:

| Action | When | What to do |
|---|---|---|
| 🟢 **Buy** | An entry setup fired in the last 2 weeks | Consider DCA-ing in — don't chase all at once |
| 🟠 **Trim** | Whales appear to be selling to retail (top pattern) | Consider taking profit / tightening stops |
| 🟡 **Watch** | Whales accumulating but no entry candle yet | Wait for confirmation |
| 🔵 **Hold** | Uptrend intact, whales still positioned | Sit tight |
| ⚪ **Wait** | No edge either way | Do nothing, re-check later |

Each verdict comes with the reason in plain English and an explicit
*"this changes if…"* invalidation condition.

## Quick start

```bash
pip install -r requirements.txt

# 1. Pull data (FINRA + EDGAR + Yahoo — needs open internet)
python -m whaletrading.pipeline

# 2. Launch the dashboard
streamlit run app.py
```

First refresh backfills ~1 year of FINRA daily files (one HTTP request per
trading day), so it takes a few minutes; later refreshes are incremental.
You can also refresh from the sidebar button in the app.

### Demo mode (no network needed)

```bash
WHALETRADING_DEMO=1 python -m whaletrading.pipeline
WHALETRADING_DEMO=1 streamlit run app.py
```

Generates deterministic synthetic data so you can explore the UI offline.
Delete `data/whaletrading.db` before switching between demo and live data.

The app also **auto-bootstraps on first load**: if the SQLite cache is empty
(e.g. a fresh container), it runs the pipeline once automatically before
rendering, so you never land on a blank dashboard.

## Free online preview — Streamlit Community Cloud

Streamlit apps don't run on Netlify (it only hosts static sites / serverless
functions, not a persistent Python/WebSocket process). The free host built
for this is [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub (public repo — free tier requirement).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   pick the repo/branch → main file path `app.py`.
3. Under **Advanced settings → Secrets**, paste the contents of
   `.streamlit/secrets.toml.example` (i.e. `WHALETRADING_DEMO = "1"`) so the
   public preview always works, even if FINRA/EDGAR are briefly unreachable
   from Streamlit Cloud's IPs. Omit it if you'd rather serve live data (first
   load will be slower while it backfills FINRA history).
4. Deploy — you get a `https://<your-app>.streamlit.app` URL to share.

Community Cloud containers sleep after inactivity and reset their filesystem
on redeploy/wake, which is exactly what the auto-bootstrap step above exists
to handle.

## Sync your watchlist across devices (optional, free)

The **Manage watchlist** panel (pin/add/remove/reorder tickers) is
session-only by default: it resets on a page reload, and each browser/device
keeps its own separate list, because Community Cloud's filesystem is
ephemeral — silently rewriting `config/watchlist.yaml` on disk would look
like it worked locally but lose the change on every redeploy.

To make it persist and sync across devices instead, at zero cost, the app
can optionally store it in a private GitHub Gist:

1. Create a **private** Gist at [gist.github.com](https://gist.github.com)
   with one file named `watchlist_state.json` containing `{}`. Copy its ID
   from the URL (`https://gist.github.com/<user>/<this-part>`).
2. Create a GitHub **Personal Access Token** scoped to **`gist` only** — not
   repo access, to keep the blast radius small if it ever leaks. (Settings →
   Developer settings → Personal access tokens.)
3. In Streamlit Cloud → your app → Settings → Secrets, add:
   ```
   GITHUB_GIST_TOKEN = "<your token>"
   GITHUB_GIST_ID = "<your gist id>"
   ```
   (See `.streamlit/secrets.toml.example`.)
4. Reboot the app. The Manage watchlist panel's caption will confirm sync is
   active — pins/order/added tickers now save to the Gist on every change
   and load from it on every new session, on any device.

If sync fails for any reason (bad token, network hiccup, rate limit), the
app falls back to session-only behavior for that change and shows a small
warning — it never blocks the pin/add/remove/reorder action itself. Omit
both secrets to keep today's session-only behavior.

## Customization — `config/watchlist.yaml`

- **watchlist** — add/remove any NYSE/NASDAQ tickers
- **thresholds.overrides** — per-ticker momentum/rise/soar levels
- **settings.whale_weights** — re-weight the composite components
- **managers_13f** — which large managers to track on EDGAR (CIK numbers)
- **issuer_aliases** — 13F filings name issuers (e.g. `NVIDIA CORP`), not
  tickers; add an alias when you add a ticker so the 13F component matches it
- **settings.sec_user_agent** — SEC fair-access rules require a contact address

## Data validity — how old is what you're looking at?

The app shows this live in the **🕐 Data freshness** panel (Overview tab), but
the inherent delays are worth internalizing:

| Source | Describes | Available | Effective information age |
|---|---|---|---|
| Yahoo prices/volume | Today's trading | Intraday quotes ~15 min delayed; daily bar final after close | Hours |
| FINRA daily short volume | Today's off-exchange trading | Same evening (~6pm ET) | Hours–1 day |
| FINRA ATS weekly | One week of dark-pool volume | 2 weeks later (4 for smaller stocks) | 2–5 weeks |
| SEC 13F holdings | Quarter-end positions | Up to 45 days after quarter end | 45–135 days |
| Fear & Greed inputs (Yahoo) | Today's market-wide sentiment | Same cadence as prices | Hours |

So: the whale score's fast components (volume classification, daily dark-pool
pressure) are **at most a day old**, while the 13F layer is **a quarter old by
design** — that's why it carries the smallest weight. Treat any signal as
end-of-day/weekly information, not intraday timing.

## Market sentiment — the Fear & Greed Index

A market-wide, contrarian sentiment gauge on its own page (**😱 Fear & Greed**
in the left panel), modeled on
[CNN's Fear & Greed Index](https://edition.cnn.com/markets/fear-and-greed) —
not specific to any one stock, and not scraped from CNN. It's computed the
same way the whale score is: free Yahoo Finance data, six 0–100 sub-scores
(50 = neutral) averaged into one composite. CNN uses seven indicators; this
app matches six with a documented proxy and openly omits the seventh:

| Indicator | This app's proxy | Symbols | vs. CNN |
|---|---|---|---|
| Market Momentum | S&P 500 vs. its 125-day average | `^GSPC` | Same idea |
| Market Volatility | VIX vs. its 50-day average (inverted — a calm VIX = greed) | `^VIX` | Same idea |
| Safe Haven Demand | SPY 20-day return minus TLT 20-day return | `SPY`, `TLT` | Same idea |
| Junk Bond Demand | HYG 20-day return minus LQD 20-day return | `HYG`, `LQD` | ETF total-return divergence stands in for the real high-yield/investment-grade spread |
| Stock Price Strength | Share of a fixed ~40-stock large-cap basket near its 52-week high minus the share near its 52-week low | large-cap basket | Basket proxy, not full NYSE new-highs/lows |
| Stock Price Breadth | McClellan-style summation of net advancing volume across the same basket | large-cap basket | Basket proxy, not full NYSE advance/decline volume |
| Put/Call Options | **Omitted** | — | No free daily put/call history exists |

Each raw signal is z-scored against its own trailing ~1-year history, then
squashed to 0–100 with the same tanh normalization the whale score uses. The
composite is the mean of whatever indicators are available that day — if a
symbol fails to fetch, that indicator drops out and the rest renormalize,
same fail-open behavior as the whale score. The page shows the current
gauge, the label band (Extreme Fear · Fear · Neutral · Greed · Extreme
Greed), prior readings (previous close, 1 week/month/year ago), a 1-year
history chart, and a card per indicator — plus a chip next to the app title
on every page so the reading is always visible. It refreshes with the same
🔄 Refresh data button as everything else (see `whaletrading/indicators/fear_greed.py`).

This is a directional, contrarian tool, not a buy/sell signal for any
individual stock — treat it as context alongside the whale-score verdicts,
not a replacement for them.

## Getting fresher data — free and minimum-cost upgrades

Everything above is $0. If you want *faster* data, roughly in order of
cost-effectiveness:

| Option | Cost | What you gain | Catch |
|---|---|---|---|
| **Alpaca Basic** (alpaca.markets) | Free | Real-time streaming quotes/trades via API from the IEX exchange | Covers only ~2% of consolidated volume — fine for prices, thin for block detection |
| **Finnhub / Tiingo free tiers** | Free | Near-real-time US quotes, generous daily bars | Rate limits; terms restrict redistribution |
| **IBKR account, no data subs** | Free (account) | TWS API: 15-min delayed quotes + solid historical daily bars; free real-time Cboe One/IEX quotes *on the TWS platform itself* | API real-time needs paid subscriptions; must run TWS/IB Gateway |
| **IBKR + non-pro data bundles** | ~US$1.50–15/mo (often waived with commission activity) | Consolidated real-time via API → true tick/block-print detection, the best low-cost upgrade for this app | Requires funded account; non-professional status |
| **Alpaca Algo Trader Plus** | ~US$99/mo | Full consolidated (SIP) real-time feed via clean REST/websocket API | Priciest of the lot |

What none of these change: **no feed labels trades institutional**, and true
real-time dark-pool attribution (Unusual Whales-style) remains a paid-product
category. The cheapest genuine upgrade path for this app is an IBKR account
with one non-pro bundle feeding a block-print detector as an extra whale-score
component.

## Architecture

```
config/watchlist.yaml        tickers, thresholds, weights, 13F managers
whaletrading/
  data/prices.py             yfinance OHLCV
  data/finra_short_volume.py FINRA Reg SHO daily files
  data/finra_ats.py          FINRA ATS weekly (Query API)
  data/sec_13f.py            EDGAR 13F holdings + QoQ deltas
  data/demo.py               synthetic fixtures for demo mode
  data/store.py              SQLite cache (data/whaletrading.db)
  indicators/                ribbon, MACD, candles, whale_score composite
  indicators/fear_greed.py   market-wide Fear & Greed composite
  signals.py                 buy/sell rules
  pipeline.py                fetch → compute → persist (graceful degradation)
app.py                       Streamlit dashboard
```

Every data source degrades gracefully: if one is unreachable, its component is
dropped and the composite renormalizes over the rest.

*Not financial advice. Proxies are approximations — validate against your own
charts before trading on any signal.*
