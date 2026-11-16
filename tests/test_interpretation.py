"""Boundary contracts for shared indicator labels."""

import pytest

from volume_price_analysis import interpretation as labels


@pytest.mark.parametrize(
    ("value", "condition", "hint"),
    [
        (19.99, "Oversold", "consider_calls"),
        (20, "Neutral", "neutral"),
        (20.01, "Neutral", "neutral"),
        (79.99, "Neutral", "neutral"),
        (80, "Neutral", "neutral"),
        (80.01, "Overbought", "consider_puts"),
    ],
)
def test_mfi_boundaries(value, condition, hint):
    assert labels.threshold_verdict(value, labels.MFI_CONDITION_BANDS, "Neutral") == condition
    assert labels.threshold_verdict(value, labels.MFI_OPTIONS_SIGNAL_BANDS, "neutral") == hint


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (19.99, "weak"),
        (20, "weak"),
        (20.01, "moderate"),
        (24.99, "moderate"),
        (25, "moderate"),
        (25.01, "strong"),
        (49.99, "strong"),
        (50, "strong"),
        (50.01, "very_strong"),
    ],
)
def test_adx_boundaries(value, expected):
    assert labels.adx_strength(value) == expected


@pytest.mark.parametrize("offset, expected", [(-0.001, True), (0, False), (0.001, False)])
def test_squeeze_boundary(offset, expected):
    assert labels.squeeze_on(0.7 + offset, 1.0) is expected


@pytest.mark.parametrize(
    ("value", "condition", "hint"),
    [
        (-0.251, "Strong selling", "strong_selling"),
        (-0.25, "Neutral", "neutral"),
        (-0.249, "Neutral", "neutral"),
        (0.249, "Neutral", "neutral"),
        (0.25, "Neutral", "neutral"),
        (0.251, "Strong buying", "strong_buying"),
    ],
)
def test_cmf_strong_boundaries(value, condition, hint):
    assert labels.threshold_verdict(value, labels.CMF_CONDITION_BANDS, "Neutral") == condition
    assert labels.threshold_verdict(value, labels.CMF_SIGNAL_BANDS, "neutral") == hint


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.201, "depressed"),
        (-0.2, "neutral"),
        (-0.199, "neutral"),
        (0.199, "neutral"),
        (0.2, "neutral"),
        (0.201, "elevated"),
    ],
)
def test_cmf_extreme_boundaries(value, expected):
    assert labels.threshold_verdict(value, labels.CMF_EXTREME_BANDS, "neutral") == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-0.001, "Selling Pressure (<0)"), (0, "Neutral (0)"), (0.001, "Buying Pressure (>0)")],
)
def test_cmf_pressure_boundary(value, expected):
    assert labels.threshold_verdict(value, labels.CMF_PRESSURE_BANDS, "Neutral (0)") == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (19.99, "none"),
        (20, "none"),
        (20.01, "weak"),
        (24.99, "weak"),
        (25, "weak"),
        (25.01, "moderate"),
        (29.99, "moderate"),
        (30, "moderate"),
        (30.01, "strong"),
    ],
)
def test_adx_options_boundaries(value, expected):
    assert labels.adx_options_tier(value) == expected


@pytest.mark.parametrize("value", [20, 25, 75, 80, None, float("nan")])
def test_mfi_options_hint_stays_neutral_inside_conventional_bounds(value):
    assert labels.threshold_verdict(value, labels.MFI_OPTIONS_SIGNAL_BANDS, "neutral") == "neutral"


def test_options_payload_uses_conventional_mfi_bounds(sample_stock_data):
    from volume_price_analysis.analysis import run_options_analysis

    # Seven final money flows total 78 positive / 22 negative: MFI is exactly 78,
    # which formerly produced a puts hint despite a Neutral condition.
    data = sample_stock_data.copy()
    data["Close"] = [100.0 + i % 2 for i in range(len(data))]
    data["Open"] = data["Close"]
    data["High"] = data["Close"] + 1
    data["Low"] = data["Close"] - 1
    data["Volume"] = data["Volume"].astype(float)
    positive = data["Close"].diff().iloc[-7:] > 0
    for index, rising in positive.items():
        flow = 78 / positive.sum() if rising else 22 / (~positive).sum()
        data.loc[index, "Volume"] = flow / data.loc[index, "Close"] * 100_000
    result = run_options_analysis("TEST", data, 14)
    mfi = result["volume_indicators"]["mfi"]
    assert mfi["value"] == pytest.approx(78)
    assert mfi["condition"] == "Neutral"
    assert mfi["options_signal"] == "neutral"
