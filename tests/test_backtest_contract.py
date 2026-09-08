"""Gate-parity contract between the evidence harness and the production scan.

The harness only measures something useful if it classifies a bar as
high-conviction exactly when ``run_scan`` would. These tests pin that: the same
predicate (``analysis.passes_high_conviction_gate``) and the same ADX lookback
(``indicators.composite_adx_period``) on both sides, including bars where
ADX(10) and ADX(14) fall on opposite sides of the 28 gate.
"""

import numpy as np
import pandas as pd
import pytest

from volume_price_analysis.analysis import passes_high_conviction_gate, score_symbol
from volume_price_analysis.backtest import (
    causal_score_at,
    high_conviction_mask,
    observation_to_candidate,
)
from volume_price_analysis.indicators import calculate_adx, composite_adx_period

HOLDING_PERIOD = 14


def _frame(n: int, seed: int, drift: float, vol: float) -> pd.DataFrame:
    """Deterministic OHLCV data; the parameters select the ADX regime."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2022-01-01", periods=n, freq="B")
    close = np.maximum(100 + np.cumsum(rng.normal(drift, vol, size=n)), 1.0)
    # Draw order matters: it is what makes the seeds below reproducible.
    high = close + rng.uniform(0.2, 2.0, size=n)
    low = close - rng.uniform(0.2, 2.0, size=n)
    open_ = close - rng.normal(0.0, 1.0, size=n)
    volume = rng.integers(800_000, 2_000_000, size=n).astype(float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


# Two frames whose last bar straddles the ADX>=28 leg of the gate: ADX(10) and
# ADX(14) disagree, so the harness's old fixed ADX(14) classified them the
# opposite way from the scan, which reads the adaptive ADX(10) for a 14d hold.
#   id                     n    seed  drift  vol   ADX(10)  ADX(14)  scan verdict
ADX_DISAGREEMENT_FRAMES = [
    pytest.param(120, 115, 0.3, 2.0, id="adx10_passes_adx14_fails"),  # 33.98 / 26.89 -> high
    pytest.param(200, 22, 0.5, 1.5, id="adx10_fails_adx14_passes"),  # 27.00 / 30.91 -> not high
]


def _harness_verdict(data: pd.DataFrame) -> bool:
    """How the evidence harness classifies the final bar of ``data``."""
    snap = causal_score_at(data, len(data) - 1, holding_period=HOLDING_PERIOD)
    obs = pd.DataFrame([{**snap, "forward_return": 0.0}])
    return bool(high_conviction_mask(obs)[0])


def _scan_verdict(data: pd.DataFrame) -> bool:
    """How ``run_scan``'s candidate pipeline classifies the final bar."""
    candidate = score_symbol(
        data,
        "TEST",
        holding_period=HOLDING_PERIOD,
        min_score=0,
        min_adx=0,
        max_iv=100,
        direction="both",
        min_avg_volume=0,
    )
    assert candidate is not None, "fixture must clear the permissive scan filters"
    return passes_high_conviction_gate(candidate)


@pytest.mark.parametrize(("n", "seed", "drift", "vol"), ADX_DISAGREEMENT_FRAMES)
def test_harness_and_scan_agree_when_adx10_and_adx14_disagree(n, seed, drift, vol):
    """The contract, on the bars that used to expose the divergence."""
    data = _frame(n, seed, drift, vol)

    adx10 = float(calculate_adx(data, 10)["adx"])
    adx14 = float(calculate_adx(data, 14)["adx"])
    # Guard the fixture: if this stops holding, the test no longer covers the case.
    assert (adx10 >= 28.0) != (adx14 >= 28.0), f"ADX(10)={adx10} ADX(14)={adx14} agree"

    scan = _scan_verdict(data)
    assert _harness_verdict(data) == scan
    # And the shared verdict follows the adaptive period, not the old fixed one.
    assert scan is (adx10 >= 28.0)


@pytest.mark.parametrize(("n", "seed", "drift", "vol"), ADX_DISAGREEMENT_FRAMES)
def test_harness_adx_equals_scan_adx(n, seed, drift, vol):
    """Both sides read the same ADX number at the same period."""
    data = _frame(n, seed, drift, vol)
    snap = causal_score_at(data, len(data) - 1, holding_period=HOLDING_PERIOD)
    candidate = score_symbol(data, "TEST", HOLDING_PERIOD, 0, 0, 100, "both", 0)
    assert candidate is not None
    assert snap["adx_period"] == composite_adx_period(HOLDING_PERIOD) == candidate["adx_period"]
    assert round(snap["adx"], 1) == candidate["adx"]


def test_observation_candidate_rounds_like_the_scan():
    """A borderline ADX must round the scan's way before the gate reads it."""
    candidate = observation_to_candidate(
        {"composite_score": 4.004, "adx": 27.96, "iv_percentile": 49.96}
    )
    assert candidate == {"composite_score": 4.0, "adx": 28.0, "hv_percentile": 50.0}
    assert passes_high_conviction_gate(candidate) is True


def test_high_conviction_mask_is_the_scan_predicate():
    """The mask holds no thresholds of its own — it is the scan's gate, row-wise."""
    obs = pd.DataFrame(
        [
            {"composite_score": 5.0, "adx": 30.0, "iv_percentile": 40.0},  # passes
            {"composite_score": 5.0, "adx": 20.0, "iv_percentile": 40.0},  # ADX fails
            {"composite_score": 1.0, "adx": 30.0, "iv_percentile": 40.0},  # score fails
            {"composite_score": 5.0, "adx": 30.0, "iv_percentile": 80.0},  # HV fails
            {"composite_score": 5.0, "adx": np.nan, "iv_percentile": 40.0},  # NaN fails
        ]
    )
    expected = [
        passes_high_conviction_gate(observation_to_candidate(r)) for r in obs.to_dict("records")
    ]
    assert list(high_conviction_mask(obs)) == expected == [True, False, False, False, False]


def test_high_conviction_mask_on_empty_frame():
    assert high_conviction_mask(pd.DataFrame()).tolist() == []
