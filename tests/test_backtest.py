"""Tests for the strictly-causal backtest / evidence harness.

The cardinal risk in this module is **lookahead bias**: a signal computed at
bar ``t`` must depend only on data up to and including ``t``. The headline test
here (``test_causal_score_is_invariant_to_future_bars``) locks that contract by
asserting that appending future bars never changes the signal at ``t``.
"""

import math

import numpy as np
import pandas as pd
import pytest

from volume_price_analysis import backtest
from volume_price_analysis.backtest import (
    causal_score_at,
    compute_observations,
    evaluate_observations,
    format_report,
    forward_returns,
    main,
    pool_observations,
    run_evidence,
)


def _synthetic_data(n: int, seed: int = 7) -> pd.DataFrame:
    """Deterministic OHLCV data with enough variation to move indicators."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2022-01-01", periods=n, freq="B")
    # Random-walk close with mild drift so trend/volume indicators vary.
    steps = rng.normal(0.3, 2.0, size=n)
    close = 100 + np.cumsum(steps)
    close = np.maximum(close, 1.0)  # keep prices positive
    high = close + rng.uniform(0.2, 2.0, size=n)
    low = close - rng.uniform(0.2, 2.0, size=n)
    open_ = close - rng.normal(0.0, 1.0, size=n)
    volume = rng.integers(800_000, 2_000_000, size=n).astype(float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": np.maximum.reduce([high, open_, close]),
            "Low": np.minimum.reduce([low, open_, close]),
            "Close": close,
            "Volume": volume,
        }
    )


# --------------------------------------------------------------------------- #
# forward_returns
# --------------------------------------------------------------------------- #


def test_forward_returns_matches_definition():
    data = pd.DataFrame({"Close": [100.0, 110.0, 121.0, 100.0, 50.0]})
    fr = forward_returns(data, horizon=2)
    # bar 0: Close[2]/Close[0]-1 = 121/100 - 1 = 0.21
    assert fr.iloc[0] == pytest.approx(0.21)
    # bar 1: Close[3]/Close[1]-1 = 100/110 - 1
    assert fr.iloc[1] == pytest.approx(100 / 110 - 1)
    # bar 2: Close[4]/Close[2]-1 = 50/121 - 1
    assert fr.iloc[2] == pytest.approx(50 / 121 - 1)


def test_forward_returns_tail_is_nan():
    """The last ``horizon`` bars have no forward bar and must be NaN."""
    data = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]})
    fr = forward_returns(data, horizon=2)
    assert math.isnan(fr.iloc[-1])
    assert math.isnan(fr.iloc[-2])
    assert not math.isnan(fr.iloc[-3])


def test_forward_returns_rejects_bad_horizon():
    data = pd.DataFrame({"Close": [100.0, 101.0]})
    with pytest.raises(ValueError):
        forward_returns(data, horizon=0)


# --------------------------------------------------------------------------- #
# causal_score_at — the no-lookahead contract
# --------------------------------------------------------------------------- #


def test_causal_score_is_invariant_to_future_bars():
    """Signal at bar ``t`` must not change when future bars are added/removed.

    This is the definitive no-lookahead test. We compute the signal at ``t``
    three ways: on data truncated exactly at ``t``, on data with a handful of
    future bars, and on the full series. All three must be identical.
    """
    data = _synthetic_data(200)
    t = 120

    snap_minimal = causal_score_at(data.iloc[: t + 1], t, holding_period=14)
    snap_some_future = causal_score_at(data.iloc[: t + 30], t, holding_period=14)
    snap_full = causal_score_at(data, t, holding_period=14)

    assert snap_minimal["composite_score"] == snap_full["composite_score"]
    assert snap_some_future["composite_score"] == snap_full["composite_score"]
    assert snap_minimal["adx"] == snap_full["adx"]
    assert snap_minimal["iv_percentile"] == snap_full["iv_percentile"]


def test_causal_score_unaffected_by_garbage_future():
    """Corrupting bars after ``t`` must leave the signal at ``t`` unchanged."""
    data = _synthetic_data(160)
    t = 100
    baseline = causal_score_at(data, t, holding_period=14)

    corrupted = data.copy()
    corrupted.loc[corrupted.index[t + 1 :], "Close"] *= 5.0
    corrupted.loc[corrupted.index[t + 1 :], "Volume"] *= 100.0
    after = causal_score_at(corrupted, t, holding_period=14)

    assert after["composite_score"] == baseline["composite_score"]
    assert after["adx"] == baseline["adx"]


def test_causal_score_out_of_range_raises():
    data = _synthetic_data(40)
    with pytest.raises(IndexError):
        causal_score_at(data, 40, holding_period=14)


def _strong_uptrend(n: int) -> pd.DataFrame:
    """Monotonic strong uptrend: cumulative and short-rolling VWAP anchors diverge."""
    dates = pd.date_range(start="2022-01-01", periods=n, freq="B")
    close = np.linspace(100.0, 400.0, n)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 1.0,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        }
    )


# --------------------------------------------------------------------------- #
# VWAP anchoring (HOM-84): the harness threads vwap_window into the scorer
# --------------------------------------------------------------------------- #


def test_causal_score_threads_vwap_window_into_scorer():
    """causal_score_at forwards its vwap_window straight to the composite scorer."""
    from unittest.mock import patch

    data = _synthetic_data(80)
    with patch(
        "volume_price_analysis.backtest.calculate_composite_score",
        wraps=backtest.calculate_composite_score,
    ) as spy:
        causal_score_at(data, 79, holding_period=14, vwap_window=15)

    assert spy.call_count == 1
    call = spy.call_args
    passed = call.kwargs.get("vwap_window", call.args[2] if len(call.args) > 2 else None)
    assert passed == 15


def test_compute_observations_threads_vwap_window_into_scorer():
    """Every per-bar score in an observation set uses the requested anchor."""
    from unittest.mock import patch

    data = _synthetic_data(120)
    with patch(
        "volume_price_analysis.backtest.calculate_composite_score",
        wraps=backtest.calculate_composite_score,
    ) as spy:
        compute_observations(
            data, horizon=10, holding_period=14, min_history=60, step=10, vwap_window=20
        )

    # bars 59..99 step 10 -> 5 evaluable bars, each scored exactly once
    assert spy.call_count >= 5
    windows = [
        c.kwargs.get("vwap_window", c.args[2] if len(c.args) > 2 else None)
        for c in spy.call_args_list
    ]
    assert all(w == 20 for w in windows)


def test_rolling_anchor_is_invariant_to_future_bars():
    """The no-lookahead contract holds for the rolling VWAP anchor too.

    The cumulative arm is locked by test_causal_score_is_invariant_to_future_bars;
    this proves a non-None vwap_window introduces no future leak either.
    """
    data = _synthetic_data(200)
    t = 120

    minimal = causal_score_at(data.iloc[: t + 1], t, holding_period=14, vwap_window=10)
    full = causal_score_at(data, t, holding_period=14, vwap_window=10)

    corrupted = data.copy()
    corrupted.loc[corrupted.index[t + 1 :], ["Close", "High", "Low"]] *= 5.0
    corrupted.loc[corrupted.index[t + 1 :], "Volume"] *= 100.0
    after = causal_score_at(corrupted, t, holding_period=14, vwap_window=10)

    assert minimal["composite_score"] == full["composite_score"]
    assert full["composite_score"] == after["composite_score"]


def test_windowed_anchor_changes_some_scores_on_trend():
    """A short rolling anchor moves at least one composite score vs the cumulative anchor."""
    data = _strong_uptrend(160)
    cumulative = compute_observations(
        data, horizon=10, holding_period=14, min_history=50, step=5, vwap_window=None
    )
    rolling = compute_observations(
        data, horizon=10, holding_period=14, min_history=50, step=5, vwap_window=5
    )
    # Same bars are evaluated in both arms; the anchor change must perturb scores.
    assert len(cumulative) == len(rolling) and len(cumulative) > 0
    assert (cumulative["composite_score"].to_numpy() != rolling["composite_score"].to_numpy()).any()


# --------------------------------------------------------------------------- #
# compute_observations
# --------------------------------------------------------------------------- #


def test_compute_observations_no_nan_forward_returns():
    """Every observation must have a realized forward return (no leakage tail)."""
    data = _synthetic_data(200)
    obs = compute_observations(data, horizon=10, holding_period=14, min_history=50)
    assert len(obs) > 0
    assert obs["forward_return"].notna().all()
    assert {"date", "composite_score", "adx", "iv_percentile", "forward_return"} <= set(obs.columns)


def test_compute_observations_respects_boundaries():
    """No bar before ``min_history`` and none within ``horizon`` of the end."""
    data = _synthetic_data(200)
    horizon, min_history = 14, 60
    obs = compute_observations(data, horizon=horizon, holding_period=14, min_history=min_history)
    assert obs["bar"].min() >= min_history - 1
    assert obs["bar"].max() <= len(data) - 1 - horizon


def test_compute_observations_forward_return_matches_close():
    data = _synthetic_data(120)
    horizon = 7
    obs = compute_observations(data, horizon=horizon, holding_period=14, min_history=50)
    close = data["Close"].to_numpy()
    for _, row in obs.iterrows():
        t = int(row["bar"])
        expected = close[t + horizon] / close[t] - 1
        assert row["forward_return"] == pytest.approx(expected)


def test_compute_observations_score_invariant_to_truncation():
    """Observations shared between full and truncated runs must match exactly.

    This is the no-lookahead contract at the dataset level: truncating future
    bars must not change any already-evaluable observation's score.
    """
    data = _synthetic_data(220)
    obs_full = compute_observations(data, horizon=5, holding_period=14, min_history=50)
    obs_trunc = compute_observations(data.iloc[:160], horizon=5, holding_period=14, min_history=50)
    merged = obs_full.merge(obs_trunc, on="bar", suffixes=("_full", "_trunc"))
    assert len(merged) > 0
    pd.testing.assert_series_equal(
        merged["composite_score_full"],
        merged["composite_score_trunc"],
        check_names=False,
    )


def test_compute_observations_step_subsamples():
    data = _synthetic_data(200)
    obs_all = compute_observations(data, horizon=5, holding_period=14, min_history=50, step=1)
    obs_step = compute_observations(data, horizon=5, holding_period=14, min_history=50, step=3)
    assert len(obs_step) < len(obs_all)
    # stepped bars are a subset spaced by `step`
    bars = obs_step["bar"].to_numpy()
    assert np.all(np.diff(bars) == 3)


def test_compute_observations_insufficient_history_returns_empty():
    data = _synthetic_data(20)
    obs = compute_observations(data, horizon=5, holding_period=14, min_history=50)
    assert len(obs) == 0


def test_compute_observations_rejects_bad_horizon_and_step():
    data = _synthetic_data(60)
    with pytest.raises(ValueError):
        compute_observations(data, horizon=0)
    with pytest.raises(ValueError):
        compute_observations(data, horizon=5, step=0)


# --------------------------------------------------------------------------- #
# evaluate_observations
# --------------------------------------------------------------------------- #


def _obs(scores, fwd, adx=None, iv=None) -> pd.DataFrame:
    n = len(scores)
    return pd.DataFrame(
        {
            "bar": range(n),
            "date": pd.date_range("2023-01-01", periods=n, freq="B"),
            "composite_score": scores,
            "adx": adx if adx is not None else [30.0] * n,
            "iv_percentile": iv if iv is not None else [40.0] * n,
            "forward_return": fwd,
        }
    )


def test_evaluate_hit_rate_perfect():
    # score sign always matches forward-return sign -> hit rate 1.0
    obs = _obs([5, -5, 3, -3], [0.02, -0.01, 0.05, -0.04])
    res = evaluate_observations(obs)
    assert res["hit_rate_directional"] == pytest.approx(1.0)
    assert res["n"] == 4


def test_evaluate_hit_rate_inverted():
    # score sign always opposes forward-return sign -> hit rate 0.0
    obs = _obs([5, -5, 3, -3], [-0.02, 0.01, -0.05, 0.04])
    res = evaluate_observations(obs)
    assert res["hit_rate_directional"] == pytest.approx(0.0)


def test_evaluate_directional_return_is_signed_mean():
    scores = [4, -4, 2]
    fwd = [0.03, -0.02, 0.01]
    obs = _obs(scores, fwd)
    res = evaluate_observations(obs)
    expected = np.mean([np.sign(s) * f for s, f in zip(scores, fwd, strict=True)])
    assert res["mean_directional_return"] == pytest.approx(expected)


def test_evaluate_ic_monotonic_positive():
    scores = list(range(-5, 6))
    fwd = [s * 0.01 for s in scores]  # perfectly rank-correlated
    obs = _obs(scores, fwd)
    res = evaluate_observations(obs)
    assert res["spearman_ic"] == pytest.approx(1.0)
    assert res["pearson_ic"] == pytest.approx(1.0)


def test_evaluate_ic_monotonic_negative():
    scores = list(range(-5, 6))
    fwd = [-s * 0.01 for s in scores]
    obs = _obs(scores, fwd)
    res = evaluate_observations(obs)
    assert res["spearman_ic"] == pytest.approx(-1.0)


def test_evaluate_degenerate_inputs():
    # zero-variance scores -> IC undefined (None), no crash
    obs = _obs([0, 0, 0, 0], [0.01, -0.02, 0.03, 0.0])
    res = evaluate_observations(obs)
    assert res["spearman_ic"] is None
    assert res["pearson_ic"] is None
    # empty -> graceful zeros/None
    empty = _obs([], [])
    res_empty = evaluate_observations(empty)
    assert res_empty["n"] == 0
    assert res_empty["hit_rate_directional"] is None


def test_evaluate_score_buckets_partition():
    scores = [-6, -3, 0, 3, 6, 1]
    fwd = [-0.01, -0.02, 0.0, 0.03, 0.05, 0.01]
    obs = _obs(scores, fwd)
    res = evaluate_observations(obs)
    counts = {b["bucket"]: b["n"] for b in res["by_score_bucket"]}
    assert counts["strong_bearish"] == 1
    assert counts["bearish"] == 1
    assert counts["neutral"] == 2  # scores 0 and 1
    assert counts["bullish"] == 1
    assert counts["strong_bullish"] == 1
    assert sum(b["n"] for b in res["by_score_bucket"]) == 6


def test_evaluate_gate_thresholds_monotone():
    scores = [1, 2, 3, 4, 5, -2, -4]
    fwd = [0.0, 0.01, 0.02, 0.03, 0.04, -0.01, -0.03]
    obs = _obs(scores, fwd)
    res = evaluate_observations(obs, gate_thresholds=(2, 3, 4))
    gates = {g["min_abs_score"]: g["n"] for g in res["by_gate_threshold"]}
    # stricter gate -> fewer or equal observations pass
    assert gates[2] >= gates[3] >= gates[4]
    assert gates[2] == 6  # |score| >= 2: 2, 3, 4, 5, -2, -4
    assert gates[4] == 3  # |score| >= 4: 4, 5, -4


def test_evaluate_high_conviction_gate():
    # Only the first row clears |score|>=4 AND adx>=28 AND iv<=50
    obs = _obs(
        scores=[5, 5, 1],
        fwd=[0.04, 0.03, 0.0],
        adx=[30.0, 20.0, 30.0],
        iv=[40.0, 40.0, 40.0],
    )
    res = evaluate_observations(obs)
    hc = res["high_conviction_gate"]
    assert hc["n"] == 1
    assert hc["mean_directional_return"] == pytest.approx(0.04)


# --------------------------------------------------------------------------- #
# pooling + reporting
# --------------------------------------------------------------------------- #


def test_pool_observations_concatenates():
    a = _obs([1, 2], [0.01, 0.02])
    a.insert(0, "symbol", "AAA")
    b = _obs([3], [0.03])
    b.insert(0, "symbol", "BBB")
    pooled = pool_observations([a, b])
    assert len(pooled) == 3
    assert set(pooled["symbol"]) == {"AAA", "BBB"}


def test_format_report_contains_key_metrics():
    obs = _obs([5, -5, 3, -3], [0.02, -0.01, 0.05, -0.04])
    res = evaluate_observations(obs)
    report = format_report(res, meta={"symbols": "TEST", "horizon": 10, "holding_period": 14})
    assert "Hit rate" in report
    assert "IC" in report
    assert "TEST" in report


def test_pool_observations_empty_returns_empty():
    assert pool_observations([]).empty


# --------------------------------------------------------------------------- #
# Correctness fixes from review (causality-adjacent quant bugs)
# --------------------------------------------------------------------------- #


def test_mean_directional_return_excludes_neutral():
    """Neutral (score==0) bars take no position and must not dilute dir return."""
    obs = _obs([4, 0], [0.02, 0.10])
    res = evaluate_observations(obs)
    assert res["mean_directional_return"] == pytest.approx(0.02)  # only the score=4 bar
    assert res["mean_forward_return"] == pytest.approx(0.06)  # raw mean over all bars
    assert res["hit_rate_directional"] == pytest.approx(1.0)
    assert res["n"] == 2
    assert res["n_directional"] == 1


def test_score_buckets_exact_partition_at_boundaries():
    """Boundary scores (+/-2, +/-5) must each land in exactly one bucket.

    Each score gets a unique forward return (score/100) so we can assert the
    *right* score is in each bucket — not just that the counts happen to total
    correctly (a dropped 2.0 and a double-counted 5.0 cancel in the total).
    """
    scores = [-5, -2, 0, 2, 5]
    obs = _obs(scores, [s / 100 for s in scores])
    res = evaluate_observations(obs)
    by = {b["bucket"]: b for b in res["by_score_bucket"]}
    assert by["strong_bearish"]["n"] == 1 and by["strong_bearish"][
        "mean_forward_return"
    ] == pytest.approx(-0.05)
    assert by["bearish"]["n"] == 1 and by["bearish"]["mean_forward_return"] == pytest.approx(-0.02)
    assert by["neutral"]["n"] == 1 and by["neutral"]["mean_forward_return"] == pytest.approx(0.0)
    assert by["bullish"]["n"] == 1 and by["bullish"]["mean_forward_return"] == pytest.approx(0.02)
    assert by["strong_bullish"]["n"] == 1 and by["strong_bullish"][
        "mean_forward_return"
    ] == pytest.approx(0.05)
    assert sum(b["n"] for b in res["by_score_bucket"]) == 5  # exact partition


def test_causal_score_adx_matches_production_period_14():
    """High-conviction gate mirrors the scan, which gates on period-14 ADX."""
    from volume_price_analysis.indicators import calculate_adx

    data = _synthetic_data(120)
    t = 100
    snap = causal_score_at(data, t, holding_period=14)
    expected = calculate_adx(data.iloc[: t + 1], 14)["adx"]
    assert snap["adx"] == pytest.approx(expected)


def test_compute_observations_drops_dirty_bars():
    """A NaN Close (data gap) must never leak a NaN into the observation set."""
    data = _synthetic_data(160)
    data.loc[data.index[100], "Close"] = np.nan
    obs = compute_observations(data, horizon=5, holding_period=14, min_history=50)
    assert len(obs) > 0
    assert obs["forward_return"].notna().all()
    assert obs["composite_score"].notna().all()


def test_forward_returns_zero_close_is_nan_not_inf():
    data = pd.DataFrame({"Close": [0.0, 100.0, 110.0]})
    fr = forward_returns(data, horizon=1)
    assert math.isnan(fr.iloc[0])  # 100/0 must be NaN, not inf


# --------------------------------------------------------------------------- #
# run_evidence + main (orchestration; fetch is monkeypatched, no network)
# --------------------------------------------------------------------------- #


def _fake_fetch_factory(monkeypatch):
    """Patch fetch_stock_data: known good symbols return data, others raise."""

    def fake_fetch(symbol, start, end, period):
        if symbol == "BAD":
            raise ValueError("no data found for BAD")
        if symbol == "SHORT":
            return _synthetic_data(20)  # too little history -> empty observations
        return _synthetic_data(200, seed=hash(symbol) % 1000)

    monkeypatch.setattr(backtest, "fetch_stock_data", fake_fetch)


def test_run_evidence_isolates_per_symbol_failures(monkeypatch):
    """One bad symbol must not sink the run; good symbols still pool."""
    _fake_fetch_factory(monkeypatch)
    result = run_evidence(["GOOD", "BAD"], period="2y", horizon=10, min_history=50)
    assert result["evaluation"]["n"] > 0
    assert any("BAD" in e for e in result["errors"])
    # pooled observations carry the contributing symbol
    assert "GOOD" in set(result["observations"]["symbol"])
    assert "BAD" not in set(result["observations"]["symbol"])


def test_run_evidence_records_insufficient_history(monkeypatch):
    _fake_fetch_factory(monkeypatch)
    result = run_evidence(["SHORT"], period="2y", horizon=10, min_history=50)
    assert result["evaluation"]["n"] == 0
    assert any("insufficient history" in e for e in result["errors"])


def test_run_evidence_pools_multiple_symbols(monkeypatch):
    _fake_fetch_factory(monkeypatch)
    one = run_evidence(["GOOD"], period="2y", horizon=10, min_history=50)
    two = run_evidence(["GOOD", "ALSO"], period="2y", horizon=10, min_history=50)
    assert two["evaluation"]["n"] > one["evaluation"]["n"]


def test_main_prints_report_and_returns_zero(monkeypatch, capsys):
    _fake_fetch_factory(monkeypatch)
    rc = main(["GOOD", "BAD", "--period", "2y", "--horizon", "10", "--min-history", "50"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "VPA EVIDENCE HARNESS" in out
    assert "Errors:" in out  # BAD failure surfaced
    assert "BAD" in out
    assert "VWAP anchor: cumulative" in out  # default anchor labelled


def test_main_rolling_vwap_window_labels_anchor(monkeypatch, capsys):
    """--vwap-window N runs the rolling arm and labels it in the report."""
    _fake_fetch_factory(monkeypatch)
    rc = main(["GOOD", "--horizon", "10", "--min-history", "50", "--vwap-window", "20"])
    assert rc == 0
    assert "VWAP anchor: rolling-20" in capsys.readouterr().out


def test_main_vwap_window_cumulative_keyword(monkeypatch, capsys):
    """The literal 'cumulative' keyword selects the cumulative anchor."""
    _fake_fetch_factory(monkeypatch)
    rc = main(["GOOD", "--horizon", "10", "--min-history", "50", "--vwap-window", "cumulative"])
    assert rc == 0
    assert "VWAP anchor: cumulative" in capsys.readouterr().out


def test_main_vwap_window_rejects_invalid():
    """A non-integer, non-keyword --vwap-window is rejected at parse time."""
    with pytest.raises(SystemExit):
        main(["GOOD", "--vwap-window", "abc"])


def test_main_vwap_window_rejects_nonpositive():
    """--vwap-window must be a positive window when numeric."""
    with pytest.raises(SystemExit):
        main(["GOOD", "--vwap-window", "0"])
