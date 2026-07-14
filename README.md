# 🐋 WhaleTrading

Track institutional ("whale") buying vs selling pressure for NYSE/NASDAQ stocks
using **only free, no-subscription data sources**, visualized in a 4-panel
Streamlit dashboard:

1. **Price** — candlesticks + EMA ribbon (8/13/21/34/55) + buy/sell markers
2. **Volume** — colored by ribbon trend (red = bullish ribbon, blue = bearish)
3. **Whale (red) vs retail (green) accumulation** — composite 0–100 scores with
   the strategy's 35 / 50 / 75 threshold guides
4. **MACD** — 12/26/9 with golden/death crosses

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

## Customization — `config/watchlist.yaml`

- **watchlist** — add/remove any NYSE/NASDAQ tickers
- **thresholds.overrides** — per-ticker momentum/rise/soar levels
- **settings.whale_weights** — re-weight the composite components
- **managers_13f** — which large managers to track on EDGAR (CIK numbers)
- **issuer_aliases** — 13F filings name issuers (e.g. `NVIDIA CORP`), not
  tickers; add an alias when you add a ticker so the 13F component matches it
- **settings.sec_user_agent** — SEC fair-access rules require a contact address

## What about IBKR?

The IBKR TWS API needs an account, and without paid market-data subscriptions
its data is 15-minute delayed — and it still never tags trades as institutional.
If you later get an account, a natural extension is a block-print detector on
IBKR time-and-sales feeding an extra component into the whale score.

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
  signals.py                 buy/sell rules
  pipeline.py                fetch → compute → persist (graceful degradation)
app.py                       Streamlit dashboard
```

Every data source degrades gracefully: if one is unreachable, its component is
dropped and the composite renormalizes over the rest.

*Not financial advice. Proxies are approximations — validate against your own
charts before trading on any signal.*
