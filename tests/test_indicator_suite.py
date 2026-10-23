"""Indicator suite contract: adaptive tuning and undefined latest values."""

import pytest

from volume_price_analysis.indicator_suite import build_snapshot


@pytest.mark.parametrize(
    ("holding_period", "expected"),
    [
        (14, (7, 10, 7, 10, 10)),
        (15, (10, 14, 10, 14, 14)),
        (21, (10, 14, 10, 14, 14)),
        (22, (14, 20, 14, 14, 20)),
    ],
)
def test_snapshot_periods(sample_stock_data, holding_period, expected):
    snapshot = build_snapshot(sample_stock_data, holding_period)
    assert tuple(snapshot.periods.values()) == expected


def test_short_history_defaults(sample_stock_data):
    snapshot = build_snapshot(sample_stock_data.iloc[:1], 14)
    assert snapshot.latest_mfi == 50.0
    assert snapshot.latest_cmf == 0.0
    assert snapshot.rsi_data["rsi"] == 50.0
    assert snapshot.adx_data["adx"] == 0.0
    assert snapshot.breakout["is_breakout"] is False


def test_nan_tail_defaults(sample_stock_data):
    sample_stock_data.loc[sample_stock_data.index[-1], ["High", "Low", "Close", "Volume"]] = float(
        "nan"
    )
    snapshot = build_snapshot(sample_stock_data, 14)
    assert snapshot.latest_mfi == pytest.approx(83.6788110352297)
    assert snapshot.latest_cmf == 0.0
    assert 0.0 <= snapshot.rsi_data["rsi"] <= 100.0
    assert 0.0 <= snapshot.adx_data["adx"] <= 100.0
    assert snapshot.breakout["is_breakout"] is False


@pytest.mark.parametrize("holding_period", [7, 14, 15, 21, 22, 30])
def test_options_computes_shared_indicators_once(sample_stock_data, mocker, holding_period):
    from volume_price_analysis import indicators
    from volume_price_analysis.analysis import run_options_analysis

    names = (
        "calculate_obv",
        "calculate_accumulation_distribution",
        "calculate_vwap",
        "calculate_vwma",
        "calculate_mfi",
        "calculate_chaikin_money_flow",
        "calculate_rsi_with_divergence",
        "calculate_adx",
        "detect_volume_breakout",
    )
    calls = [mocker.spy(indicators, name) for name in names]
    run_options_analysis("TEST", sample_stock_data, holding_period)
    assert [call.call_count for call in calls] == [1] * len(names)


def test_snapshot_preserves_and_reuses_distinct_tuning(sample_stock_data):
    snapshot = build_snapshot(sample_stock_data, 14)
    assert snapshot.rsi(7, 10) is snapshot.rsi_data
    assert snapshot.rsi(14, 10) is snapshot.rsi(14, 10)
    assert snapshot.rsi(14, 10)["rsi"] != snapshot.rsi_data["rsi"]
