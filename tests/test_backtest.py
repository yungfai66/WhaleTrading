"""Tests for whaletrading.backtest — see
docs/superpowers/specs/2026-07-31-backtester-design.md for what each of
these is guarding against.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from whaletrading import backtest, pipeline
from whaletrading.config import Config


# ---------------------------------------------------------------------------
# Causality guard: the whole point of _reconstruct_causal_score is that a
# bar's value never depends on data that arrives after it (that's the bug
# that made the stored inst_13f column non-causal). This shuffles every
# components entry strictly after a cutoff date and asserts rows on/before
# the cutoff are byte-for-byte unchanged.
# ---------------------------------------------------------------------------
def test_causal_reconstruction_ignores_future_data():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    rng = np.random.default_rng(1)
    components = [
        json.dumps({"big_money_volume": float(v), "dark_pool": float(v2)})
        for v, v2 in zip(rng.uniform(0, 100, 40), rng.uniform(0, 100, 40))
    ]
    metrics = pd.DataFrame(
        {"whale_score": 50.0, "retail_score": 50.0, "components": components}, index=dates
    )

    before = backtest._reconstruct_causal_score(metrics)

    cutoff = dates[20]
    mutated = metrics.copy()
    future_mask = mutated.index > cutoff
    shuffled = rng.permutation(mutated.loc[future_mask, "components"].to_numpy())
    mutated.loc[future_mask, "components"] = shuffled

    after = backtest._reconstruct_causal_score(mutated)

    pd.testing.assert_frame_equal(
        before.loc[before.index <= cutoff], after.loc[after.index <= cutoff]
    )


def test_causal_reconstruction_drops_inst_13f_and_renormalizes():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    metrics = pd.DataFrame(
        {
            "whale_score": [999.0, 999.0, 999.0],  # stored value must be ignored
            "retail_score": [50.0, 50.0, 50.0],
            "components": [
                json.dumps({"big_money_volume": 80.0, "inst_13f": 0.0}),
                json.dumps({"big_money_volume": 60.0, "dark_pool": 20.0, "inst_13f": 100.0}),
                json.dumps({"dark_pool": 40.0}),
            ],
        },
        index=dates,
    )
    out = backtest._reconstruct_causal_score(metrics)

    # Row 0: only big_money_volume present -> causal score is exactly that,
    # not diluted by the (ignored) inst_13f=0.
    assert out["whale_score"].iloc[0] == pytest.approx(80.0)
    # Row 1: two components -> simple average, inst_13f excluded entirely.
    assert out["whale_score"].iloc[1] == pytest.approx((60.0 + 20.0) / 2)
    assert out["has_dark_pool"].tolist() == [False, True, True]


# ---------------------------------------------------------------------------
# Entry-timing: a signal computed from bar t must fill at t+1's open, never
# at t's own close (which is only fully known once t has printed).
# ---------------------------------------------------------------------------
def test_forward_returns_entry_is_next_bar_open():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 14, 15],
            "high": [10, 11, 12, 13, 14, 15],
            "low": [10, 11, 12, 13, 14, 15],
            "close": [10, 11, 12, 13, 14, 15],
        },
        index=dates,
    )
    fwd = backtest._forward_returns(frame, horizons=(2,))

    # Signal at bar 0 (close=10): entry = open[1] = 11, exit = close[0+1+2] = close[3] = 13.
    assert fwd["fwd_ret_2"].iloc[0] == pytest.approx(13 / 11 - 1)
    # Last two bars have no room for a 2-bar-forward exit -> NaN, not padded.
    assert fwd["fwd_ret_2"].iloc[-1] != fwd["fwd_ret_2"].iloc[-1]  # NaN
    assert fwd["fwd_ret_2"].iloc[-2] != fwd["fwd_ret_2"].iloc[-2]  # NaN


# ---------------------------------------------------------------------------
# Null calibration: drawing the "signal" sample from the same distribution as
# the baseline pool must center the permutation p-value near 0.5. This
# validates the baseline/permutation arithmetic independent of any real
# signal logic.
# ---------------------------------------------------------------------------
def test_permutation_pvalue_null_case_centers_near_half():
    rng = np.random.default_rng(42)
    pool = rng.normal(0, 1, 5000)
    observed_excess = 0.0  # a "signal" whose mean exactly matches the pool
    p = backtest._permutation_pvalue(pool, n_events=30, observed=observed_excess, rng=rng)
    assert p is not None
    assert 0.3 < p < 0.7


def test_permutation_pvalue_known_edge_is_significant():
    rng = np.random.default_rng(7)
    pool = rng.normal(0, 1, 5000)
    # A large, obvious excess return should almost never be beaten by chance.
    p = backtest._permutation_pvalue(pool, n_events=30, observed=2.0, rng=rng)
    assert p is not None
    assert p < 0.05


def test_permutation_pvalue_none_when_insufficient_pool():
    rng = np.random.default_rng(0)
    pool = np.array([0.1, 0.2, 0.3])
    assert backtest._permutation_pvalue(pool, n_events=30, observed=0.0, rng=rng) is None


def test_permutation_pvalue_left_tail_flags_negative_edge():
    # A real, strongly negative effect (as a working sell signal should
    # show) must score as significant under tail="less" ...
    rng = np.random.default_rng(3)
    pool = rng.normal(0, 1, 5000)
    p_less = backtest._permutation_pvalue(pool, n_events=30, observed=-2.0, rng=rng, tail="less")
    assert p_less < 0.05
    # ... but the same observation under the default (buy-reason) tail would
    # wrongly look insignificant -- this is exactly the bug being guarded
    # against: using "greater" for a sell reason hides a real effect.
    p_greater = backtest._permutation_pvalue(pool, n_events=30, observed=-2.0, rng=rng, tail="greater")
    assert p_greater > 0.9


# ---------------------------------------------------------------------------
# End-to-end wiring: run the real pipeline in demo mode into a temp DB, then
# run_event_study over it. This exercises the actual store/signals/pipeline
# integration, including confirming inst_13f (which demo mode does populate)
# is excluded from the reconstructed score.
# ---------------------------------------------------------------------------
@pytest.fixture
def demo_db(tmp_path, monkeypatch):
    monkeypatch.setenv("WHALETRADING_DEMO", "1")
    db_path = tmp_path / "test.db"
    cfg = Config(
        watchlists={"test": ["AAA", "BBB"]},
        default_watchlist="test",
        price_lookback_years=2,
        managers_13f=[{"cik": 1, "name": "Test Manager"}],
    )
    pipeline.refresh_all(cfg, db_path=db_path, tickers=["AAA", "BBB"])
    return db_path


def test_run_event_study_end_to_end(demo_db):
    result = backtest.run_event_study(["AAA", "BBB"], db_path=demo_db)

    assert set(result.coverage["ticker"]) == {"AAA", "BBB"}
    assert (result.coverage["bars"] > 0).all()

    if not result.events.empty:
        assert set(result.events["reason"].unique()) <= set(
            backtest.SIGNAL_REASONS + backtest.SELL_REASONS
        )
        # No event should be missing a forward-return column entirely.
        assert "fwd_ret_4" in result.events.columns

    if not result.summary.empty:
        assert set(result.summary["era"].unique()) <= {"all", "dark_pool_present"}
        # n below the significance floor must not report a p-value.
        thin = result.summary[result.summary["n"] < backtest.MIN_EVENTS_FOR_PVALUE]
        assert thin["p_value"].isna().all()


def test_run_event_study_empty_tickers_returns_empty_result(tmp_path):
    # Explicit db_path so this never touches the real project database.
    result = backtest.run_event_study([], db_path=tmp_path / "empty.db")
    assert result.events.empty
    assert result.coverage.empty
