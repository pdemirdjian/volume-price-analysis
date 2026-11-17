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
        ["total_candidates", "bullish_setups", "bearish_setups", "high_conviction"],
        total_candidates=_INTEGER,
        bullish_setups=_INTEGER,
        bearish_setups=_INTEGER,
        high_conviction=_INTEGER,
    ),
    high_conviction_setups=_CANDIDATES,
    top_bullish=_CANDIDATES,
    top_bearish=_CANDIDATES,
    errors={"type": "array", "items": _object(["symbol", "error"], symbol=_STRING, error=_STRING)},
    market_regime=_OBJECT,
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
    trend_analysis=_OBJECT,
    volatility_analysis=_OBJECT,
    volume_indicators=_OBJECT,
    volume_profile=_OBJECT,
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
    volume_indicators=_OBJECT,
    price_indicators=_OBJECT,
    volatility_indicators=_OBJECT,
    volume_profile=_OBJECT,
    summary={"type": "array", "items": _STRING},
)
