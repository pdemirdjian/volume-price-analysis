"""Hand-written JSON Schemas for tool output, advertised as each tool's MCP ``outputSchema``.

The schemas pin only what downstream consumers read (the briefing agent,
agentic-trading, scripts): those keys are ``required`` and loosely typed, and
every object stays open (``additionalProperties: true``) so adding a field never
breaks a client. Numbers are nullable because NaN/Infinity serialise as null.
The TypedDicts in :mod:`.payloads` describe the full shape for type checking;
these schemas are the narrower runtime contract.
"""

from typing import Any

_NUMBER: dict[str, Any] = {"type": ["number", "null"]}
_STRING: dict[str, Any] = {"type": "string"}
_INTEGER: dict[str, Any] = {"type": "integer"}
_OBJECT: dict[str, Any] = {"type": "object", "additionalProperties": True}


def _object(required: list[str], **properties: dict[str, Any]) -> dict[str, Any]:
    """An open object schema requiring ``required`` and typing ``properties``."""
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": True,
    }


_HEADLINE = _object(
    ["recommendation", "composite_score", "signal_quality", "rationale"],
    recommendation=_STRING,
    composite_score=_NUMBER,
    signal_quality=_STRING,
    rationale=_STRING,
)

_CANDIDATE = _object(
    [
        "symbol",
        "composite_score",
        "adx",
        "iv_percentile",
        "hv_percentile",
        "latest_price",
        "atr",
        "signal_quality",
        "key_levels",
    ],
    symbol=_STRING,
    composite_score=_NUMBER,
    adx=_NUMBER,
    iv_percentile=_NUMBER,
    hv_percentile=_NUMBER,
    latest_price=_NUMBER,
    atr=_NUMBER,
    signal_quality=_STRING,
    key_levels=_OBJECT,
    # PDE-248: GICS sector for sp500 members, null otherwise. Optional output field.
    sector={"type": ["string", "null"]},
)

_CANDIDATES: dict[str, Any] = {"type": "array", "items": _CANDIDATE}

SCAN_RESULT_SCHEMA = _object(
    [
        "scan_parameters",
        "summary",
        "high_conviction_setups",
        "top_bullish",
        "top_bearish",
        "errors",
    ],
    scan_parameters=_object(["symbols_scanned"], symbols_scanned=_INTEGER),
    summary=_object(
        ["total_candidates", "bullish_setups", "bearish_setups", "high_conviction", "breadth"],
        total_candidates=_INTEGER,
        bullish_setups=_INTEGER,
        bearish_setups=_INTEGER,
        high_conviction=_INTEGER,
        # PDE-210: run_scan always emits every key; the briefing header reads them.
        breadth=_object(
            [
                "universe",
                "symbols_counted",
                "pct_above_sma20",
                "pct_above_sma50",
                "up_volume",
                "down_volume",
                "up_down_volume_ratio",
                "new_highs",
                "new_lows",
                "net_new_highs",
            ],
            universe=_STRING,
            symbols_counted=_INTEGER,
            pct_above_sma20=_NUMBER,
            pct_above_sma50=_NUMBER,
            up_volume=_NUMBER,
            down_volume=_NUMBER,
            up_down_volume_ratio=_NUMBER,
            new_highs=_INTEGER,
            new_lows=_INTEGER,
            net_new_highs=_INTEGER,
        ),
        # PDE-248: sector ETFs ranked by 20-session return minus SPY's; optional.
        sector_strength={
            "type": "array",
            "items": _object(
                ["sector", "etf", "rs_20d", "rank"],
                sector=_STRING,
                etf=_STRING,
                return_5d=_NUMBER,
                return_20d=_NUMBER,
                rs_5d=_NUMBER,
                rs_20d=_NUMBER,
                rank=_INTEGER,
            ),
        },
    ),
    high_conviction_setups=_CANDIDATES,
    top_bullish=_CANDIDATES,
    top_bearish=_CANDIDATES,
    errors={"type": "array", "items": _object(["symbol", "error"], symbol=_STRING, error=_STRING)},
    market_regime=_OBJECT,
)

_BOOLEAN: dict[str, Any] = {"type": "boolean"}

# Nested sections the briefing agent projects by name (agent/ai_client.py,
# ``_project_deep_analysis``). Every key below is emitted on every successful result.
_VOLUME_PROFILE_LEVELS = _object(
    ["point_of_control", "value_area_high", "value_area_low", "current_position"],
    point_of_control=_NUMBER,
    value_area_high=_NUMBER,
    value_area_low=_NUMBER,
    current_position=_STRING,
)
_OBV = _object(["trend"], trend=_STRING)
_MFI = _object(["condition"], condition=_STRING)
_RELATIVE_VOLUME = _object(["current_rvol"], current_rvol=_NUMBER)
_VOLUME_BREAKOUT = _object(["is_breakout"], is_breakout=_BOOLEAN)

_OPTIONS_TREND = _object(
    ["adx", "rsi"],
    adx=_object(
        ["value", "trend_strength", "trend_direction"],
        value=_NUMBER,
        trend_strength=_STRING,
        trend_direction=_STRING,
    ),
    rsi=_object(
        ["value", "condition", "divergence_type"],
        value=_NUMBER,
        condition=_STRING,
        divergence_type=_STRING,
    ),
)

_OPTIONS_VOLATILITY = _object(
    ["iv_percentile_proxy", "expected_move", "atr", "bollinger_bands"],
    iv_percentile_proxy=_object(
        ["percentile", "hv_percentile", "options_implication"],
        percentile=_NUMBER,
        hv_percentile=_NUMBER,
        options_implication=_STRING,
    ),
    expected_move=_object(
        ["percent", "upper_target", "lower_target"],
        percent=_NUMBER,
        upper_target=_NUMBER,
        lower_target=_NUMBER,
    ),
    atr=_object(
        ["daily_range", "stop_loss_suggestion"],
        daily_range=_STRING,
        stop_loss_suggestion=_STRING,
    ),
    bollinger_bands=_object(
        ["position", "squeeze_detected"], position=_STRING, squeeze_detected=_BOOLEAN
    ),
)

_OPTIONS_VOLUME = _object(
    [
        "obv",
        "accumulation_distribution",
        "mfi",
        "cmf",
        "relative_volume",
        "volume_breakout",
    ],
    obv=_OBV,
    accumulation_distribution=_object(["signal"], signal=_STRING),
    mfi=_MFI,
    cmf=_object(["signal"], signal=_STRING),
    relative_volume=_RELATIVE_VOLUME,
    volume_breakout=_VOLUME_BREAKOUT,
)

# comprehensive_analysis has no in-repo reader; its sections that share a producer
# and key path with options_analysis pin the same keys.
_COMPREHENSIVE_VOLUME = _object(
    ["obv", "mfi", "relative_volume", "volume_breakout"],
    obv=_OBV,
    mfi=_MFI,
    relative_volume=_RELATIVE_VOLUME,
    volume_breakout=_VOLUME_BREAKOUT,
)

OPTIONS_ANALYSIS_SCHEMA = _object(
    [
        "symbol",
        "latest_price",
        "headline",
        "composite_signal",
        "trend_analysis",
        "volatility_analysis",
        "volume_indicators",
        "volume_profile",
        "options_insights",
    ],
    symbol=_STRING,
    latest_price=_NUMBER,
    headline=_HEADLINE,
    composite_signal=_object(
        ["score", "recommendation", "action", "signal_quality"],
        score=_NUMBER,
        recommendation=_STRING,
        action=_STRING,
        signal_quality=_STRING,
    ),
    trend_analysis=_OPTIONS_TREND,
    volatility_analysis=_OPTIONS_VOLATILITY,
    volume_indicators=_OPTIONS_VOLUME,
    volume_profile=_VOLUME_PROFILE_LEVELS,
    options_insights={"type": "array", "items": _STRING},
)

COMPREHENSIVE_ANALYSIS_SCHEMA = _object(
    [
        "symbol",
        "latest_price",
        "headline",
        "volume_indicators",
        "price_indicators",
        "volatility_indicators",
        "volume_profile",
        "summary",
    ],
    symbol=_STRING,
    latest_price=_NUMBER,
    headline=_HEADLINE,
    volume_indicators=_COMPREHENSIVE_VOLUME,
    price_indicators=_OBJECT,
    volatility_indicators=_OBJECT,
    volume_profile=_VOLUME_PROFILE_LEVELS,
    summary={"type": "array", "items": _STRING},
)
