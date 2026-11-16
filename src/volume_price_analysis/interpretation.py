"""Pure labeling rules; indicator calculation and scan scoring stay separate."""

import math
from collections.abc import Sequence

import pandas as pd

MFI_OVERBOUGHT = 80.0
MFI_OVERSOLD = 20.0
CMF_STRONG = 0.25
CMF_EXTREME = 0.2
CMF_NEUTRAL = 0.0
ADX_TREND = 20.0
ADX_STRONG = 25.0
ADX_DIRECTIONAL = 30.0
ADX_VERY_STRONG = 50.0

#: A threshold band: ``(operator, bound, label)``. See :func:`threshold_verdict`.
ThresholdBand = tuple[str, float, str]

#: MFI overbought/oversold ladder (default label: ``"Neutral"``).
MFI_CONDITION_BANDS: tuple[ThresholdBand, ...] = (
    (">", MFI_OVERBOUGHT, "Overbought"),
    ("<", MFI_OVERSOLD, "Oversold"),
)

#: MFI ladder with the bounds spelled out in the label (default: ``"Neutral (20-80)"``).
MFI_TOOL_CONDITION_BANDS: tuple[ThresholdBand, ...] = (
    (">", MFI_OVERBOUGHT, "Overbought (>80)"),
    ("<", MFI_OVERSOLD, "Oversold (<20)"),
)

#: MFI ladder driving the options directional hint (default: ``"neutral"``).
MFI_OPTIONS_SIGNAL_BANDS: tuple[ThresholdBand, ...] = (
    (">", MFI_OVERBOUGHT, "consider_puts"),
    ("<", MFI_OVERSOLD, "consider_calls"),
)

#: CMF strong-flow ladder (default label: ``"Neutral"``).
CMF_CONDITION_BANDS: tuple[ThresholdBand, ...] = (
    (">", CMF_STRONG, "Strong buying"),
    ("<", -CMF_STRONG, "Strong selling"),
)

#: CMF strong-flow ladder in snake_case (default label: ``"neutral"``).
CMF_SIGNAL_BANDS: tuple[ThresholdBand, ...] = (
    (">", CMF_STRONG, "strong_buying"),
    ("<", -CMF_STRONG, "strong_selling"),
)

#: CMF sign ladder used by the CMF tool (default label: ``"Neutral (0)"``).
CMF_PRESSURE_BANDS: tuple[ThresholdBand, ...] = (
    (">", CMF_NEUTRAL, "Buying Pressure (>0)"),
    ("<", CMF_NEUTRAL, "Selling Pressure (<0)"),
)

#: CMF ladder for the "extreme" MFI+CMF insight (default label: ``"neutral"``).
CMF_EXTREME_BANDS: tuple[ThresholdBand, ...] = (
    (">", CMF_EXTREME, "elevated"),
    ("<", -CMF_EXTREME, "depressed"),
)


def threshold_verdict(
    value: float | None,
    bands: Sequence[ThresholdBand],
    default: str,
    *,
    missing: str | None = None,
) -> str:
    """Return the label of the first band whose comparison holds for ``value``.

    Ladders such as MFI's 80/20 overbought/oversold split are data rather than
    code: each band is ``(operator, bound, label)`` where ``operator`` is one of
    ``">"``, ``">="``, ``"<"`` or ``"<="``. Bands are evaluated in order, so the
    first match wins.

    Args:
        value: The indicator value, or ``None`` when it could not be computed.
        bands: Ordered bands to test.
        default: Label returned when no band matches.
        missing: Label returned when ``value`` is ``None`` or NaN. Defaults to
            ``default``.

    Returns:
        The matching band's label, ``missing`` for an absent value, or ``default``.

    Raises:
        ValueError: If a band carries an unknown operator.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return default if missing is None else missing

    numeric = float(value)
    for operator, bound, label in bands:
        if operator == ">":
            matched = numeric > bound
        elif operator == ">=":
            matched = numeric >= bound
        elif operator == "<":
            matched = numeric < bound
        elif operator == "<=":
            matched = numeric <= bound
        else:
            raise ValueError(f"unknown threshold operator: {operator!r}")
        if matched:
            return label

    return default


# Bollinger squeeze canonical configuration. Deliberately not parameterized:
# every analysis path must report the same squeeze verdict for the same data.
SQUEEZE_WINDOW = 20  # band period and trailing-baseline window
SQUEEZE_THRESHOLD = 0.7  # squeeze = latest bandwidth below this fraction of the baseline
MIN_SQUEEZE_DETECTION_PERIODS = 5  # fewer valid bandwidth points than this -> never a squeeze


def adx_strength(value: float) -> str:
    """Standard ADX strength, with strict boundaries matching existing payloads."""
    return threshold_verdict(
        value,
        (
            (">", ADX_VERY_STRONG, "very_strong"),
            (">", ADX_STRONG, "strong"),
            (">", ADX_TREND, "moderate"),
        ),
        "weak",
    )


def adx_options_tier(value: float) -> str:
    """Directional-options suitability; intentionally finer than standard ADX."""
    return threshold_verdict(
        value,
        ((">", ADX_DIRECTIONAL, "strong"), (">", ADX_STRONG, "moderate"), (">", ADX_TREND, "weak")),
        "none",
    )


def squeeze_on(latest: float, baseline: float) -> bool:
    """Whether bandwidth is strictly below the canonical baseline fraction."""
    return bool(latest < baseline * SQUEEZE_THRESHOLD)
