# Backtester design: `whaletrading/backtest.py`

Date: 2026-07-31
Status: approved, pending implementation plan

## Purpose

Answer one question with numbers instead of intuition: do the buy/sell signals
in `whaletrading/signals.py` (dip reversal, ribbon turn, MACD cross, whale→
retail shift, yellow candle) carry real forward-return edge, and which of the
five reasons — if any — is doing the work?

This is a validation harness first, a Streamlit page second. The engine is a
pure function with no UI dependency; a page can call it later without any
logic living in `app.py`.

## Why not a conventional strategy backtest

A round-trip-trade backtest (enter on buy, exit on sell, plot an equity curve,
compute Sharpe/drawdown) was considered and rejected for v1:

- `sell_signal` in `signals.py` is documented as a *trim warning*, not a
  symmetric exit — some tickers may never fire one, leaving trades open
  indefinitely and distorting any equity-curve metric.
- An exit rule, position sizing, and stop-loss are all free parameters the
  strategy notes never specified. Every parameter added is a knob that can
  turn "does this work" into "I tuned it until it looked like it works."

Instead this is an **event study**: measure the forward return after each
signal fires, compare to the ticker's own baseline, and stop there. Nothing to
overfit. A trade-based equity-curve layer can be added later on top of the
same event data if the Streamlit page wants one, but it is out of scope here.

## Critical bug this design works around: non-causal `inst_13f`

`whale_score.py::composite_whale_score` stamps a single scalar
(`inst_13f_score`, computed from the two most recent 13F report periods) onto
**every row** of the score frame:

```python
score_13f, pct_13f = inst_13f_score(holdings_13f)
if score_13f is not None:
    components["inst_13f"] = score_13f
```

That means a bar from 2023 carries 13F data filed in 2026 — look-ahead bias.
Harmless for the live "what do I do now" verdict (it's a current snapshot by
construction), but fatal for a backtest: roughly 20% of every historical whale
score would be borrowed from the future.

**Fix for this design:** the backtest reconstructs the whale score itself from
the stored per-component JSON (`metrics.components`, written at
`pipeline.py::_recompute_metrics`), dropping `inst_13f` entirely and
renormalizing over whichever of `big_money_volume` / `dark_pool` remain — the
same renormalization `composite_whale_score` already does when a source is
missing. `big_money_volume` and `dark_pool` are both built from
backward-looking rolling/EWM windows, so a bar's value never depends on a
future bar once `inst_13f` is excluded. No schema change, no migration —
this is confined to `backtest.py`.

Rebuilding `inst_13f` as a genuinely point-in-time series (tracking `filed_date`,
not just `report_period`, and backfilling deeper EDGAR history) is a separate
project and explicitly out of scope here.

## Data depth (accepted constraint, not fixed by this design)

- `config/watchlist.yaml` sets `finra_short_volume_days: 365`, so
  `dark_pool` only exists for roughly the trailing year. Before that, the
  reconstructed score is `big_money_volume`-only.
- The backtest runs on whatever history exists today (5 years of prices by
  default) rather than restricting to the ~1-year window where both
  components are present. Every event record carries a `components` column
  (see below) naming which components were available at that bar, so results
  can be filtered to the 2-component era separately from the
  1-component-only era. This is a deliberate choice to preserve sample size;
  it means early and late events are not strictly apples-to-apples, and the
  summary output must make that visible rather than hide it.
- Deepening `finra_short_volume_days` (e.g. to ~1250) to backfill more
  dark-pool history is a future config change, not part of this spec.

## Event definition and entry convention

- Signals are produced by the existing `signals.evaluate()` — the backtest
  does not reimplement any rule, only supplies it a causally-reconstructed
  score frame.
- Timeframe: weekly (`W-FRI`), matching what `current_action` already uses
  for the live verdict.
- Entry price: bar `t+1` **open**, where `t` is the signal bar. Never bar `t`
  close — that price is only fully known once bar `t` has completed, and using
  it would itself be a smaller look-ahead leak.
- Forward return over horizon `h` (weeks): `close[t+1+h] / open[t+1] - 1`.
- Horizons: 4, 8, 13 weeks.
- Events without enough remaining bars to fill a horizon are dropped for that
  horizon, not padded or truncated.

## Baseline and significance

- **Baseline:** each ticker is compared to *itself* — mean forward return
  over the same horizon and entry convention across all bars for that ticker
  (not just signal bars). This controls for the ticker's own drift so a
  rising stock doesn't read as signal edge. Reported metric is **excess
  return** = signal-conditional mean − unconditional mean.
- **Secondary baseline:** SPY-relative excess return, since SPY is already
  fetched into the shared `prices` table via the Fear & Greed basket
  (`fear_greed.REQUIRED_SYMBOLS`) — effectively free to add.
- **Significance:** block permutation test. Redraw the same number of signal
  dates at random (1000 draws), recompute excess return each time, report the
  fraction of random draws that beat the observed value as an empirical
  p-value. Block size equals the longest horizon (13 weeks) so overlapping
  windows aren't treated as independent — naive iid resampling would badly
  overstate confidence given how much the 13-week windows overlap on weekly
  bars.
- Below 20 events for a given reason/horizon/ticker-group cut, the p-value is
  suppressed in output and only `n` is shown, to avoid presenting a number
  that looks precise but isn't.

## Attribution (the actual point of this tool)

Every metric above is computed per `buy_reason` / `sell_reason` value
(`dip reversal`, `ribbon turn`, `MACD cross`, `whale→retail shift`,
`yellow candle`), both pooled across the whole watchlist and broken out
per-ticker. Sell reasons are measured with the identical machinery; a working
sell signal should show forward excess return *below* zero (and below the
buy-reason baseline), not a different metric.

## Public interface

```python
def run_event_study(
    tickers: list[str],
    timeframe: str = "W",
    horizons: tuple[int, ...] = (4, 8, 13),
    db_path=None,
) -> BacktestResult:
    ...
```

`BacktestResult` (a small dataclass, no behavior) holds three DataFrames:

- `events` — one row per fired signal: ticker, date, reason, entry price,
  forward return per horizon, components-present at that bar.
- `summary` — one row per (reason, horizon), each with two variants: one over
  all events, one restricted to events where `dark_pool` was present in
  `components` (the coverage-era split described above — not an arbitrary
  grouping). Columns: n, mean excess return, SPY-relative excess return,
  empirical p-value (or blank below the n=20 floor).
- `coverage` — one row per ticker: total bars, event count, date range,
  which components were available and over what fraction of the range.

CLI entry point `python -m whaletrading.backtest` runs the study over
`cfg.all_tickers` and prints `summary` and `coverage` (mirrors the existing
`python -m whaletrading.pipeline` CLI shape). A later Streamlit page calls
`run_event_study` directly and plots the same three frames — no duplicated
logic in `app.py`.

## Explicitly out of scope for v1

- Position sizing, stop-losses, transaction costs, slippage.
- An equity curve or Sharpe/Calmar/drawdown (those require the round-trip
  framing rejected above).
- Parameter sweeps over `DELTA_EPS` / `DELTA_BARS` or the whale-score weights
  — tuning against the same data used to judge edge is how you manufacture a
  false positive.
- Rebuilding `inst_13f` as a point-in-time series.
- Backfilling deeper FINRA short-volume history.

## Expected outcome, stated up front

With weekly bars and signals that require a candle pattern + ribbon state +
score-delta to align simultaneously, individual tickers may produce only a
handful of events each. Pooled across the whole watchlist this may still be a
small sample. "Not enough events to distinguish from noise" is an anticipated
and legitimate output — not a failure of the tool — and the `n`-floor
suppression above exists specifically so that outcome is reported honestly
instead of dressed up with a spurious-looking p-value.

## Testing

- **Causality guard:** shuffle all price/score bars strictly after some cutoff
  date `D`; assert no event on or before `D` changes. This is the regression
  test that would have caught the `inst_13f` look-ahead bug had one existed
  for the composite score.
- **Null calibration:** run the study with a synthetic "signal" equal to every
  bar (unconditional). Excess return must be ~0 and empirical p ~0.5 — this
  validates the baseline arithmetic independent of any real signal logic.
- **Known-edge synthetic:** using the existing `whaletrading/data/demo.py`
  generators, construct a price series with a planted up-move immediately
  following a signal date; assert the measured excess return is positive and
  the permutation p-value is low.
- **Entry-timing check:** assert entry price always equals `open[t+1]` for a
  signal at `t`, and is never derived from `close[t]` or any earlier bar.
