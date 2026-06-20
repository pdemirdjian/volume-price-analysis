"""AI integration for generating morning briefings.

Supports multiple providers via the AI_PROVIDER environment variable:
- "gemini" (default): Google Gemini API via google-genai SDK
- "anthropic": Anthropic Claude API via anthropic SDK
"""

import json
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a professional options trading analyst writing a morning briefing email.

Your audience is an experienced options trader who wants actionable intelligence,
not generic advice. Be specific about symbols, scores, levels, and strategies.

IMPORTANT: All analysis is optimized for a **14-day holding period**. Indicator
periods, expected moves, and strategy suggestions are calibrated for this
timeframe. Make this clear in your strategy suggestions (e.g., "14-day calls"
not just "calls"). Readers doing shorter-duration plays (e.g., 0-5 DTE) should
note that signals and levels may not apply to their timeframe.

VOLATILITY HONESTY: The volatility percentile in the data (fields named
"iv_percentile", "hv_percentile", or "iv_percentile_proxy", basis
"historical_volatility") is a **Historical Volatility (HV) proxy** computed from
realized price moves — it is NOT options-market implied volatility (IV). Always
label it as an HV percentile (e.g., "HV percentile 78% (proxy)") and never
present it as implied volatility. Frame cheap/expensive options conclusions as
inferences from HV, not measured IV.

Format the briefing in markdown with clear sections:
1. **Executive Summary** - 2-3 sentence overview of today's market setup
2. **Top Picks** - For each high-conviction candidate, include:
   - Symbol, price, composite score, and direction (bullish/bearish)
   - Key levels (support/resistance from volume profile)
   - Suggested strategy (calls, puts, spreads, etc.) with 14-day DTE framing
   - Risk factors to watch
3. **Market Context** - Overall scan statistics, sector themes
4. **Risk Warnings** - Any divergences, extreme readings, or caution flags

Keep it concise but thorough. Use bullet points. No fluff."""

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-5-20250929",
}

_TRUNCATION_WARNING = (
    "\n\n---\n"
    "**WARNING: This briefing was cut short due to output length limits. "
    "Some candidates or sections may be missing.**"
)


def generate_briefing(
    scan_results: dict,
    deep_analyses: list[dict],
    provider: str = "gemini",
    model: str = "",
    api_key: str = "",
) -> str:
    """
    Generate a natural-language briefing from scan results and deep analyses.

    Args:
        scan_results: Output from run_scan().
        deep_analyses: List of outputs from run_options_analysis() for top candidates.
        provider: AI provider ("gemini" or "anthropic").
        model: Model name. If empty, uses the default for the provider.
        api_key: API key for the selected provider.

    Returns:
        Markdown-formatted briefing text.
    """
    if not model:
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["gemini"])

    user_content = _build_user_message(scan_results, deep_analyses)

    if provider == "anthropic":
        return _generate_anthropic(user_content, model, api_key)
    elif provider == "gemini":
        return _generate_gemini(user_content, model, api_key)
    else:
        msg = f"Unknown AI provider: {provider!r}. Use 'gemini' or 'anthropic'."
        raise ValueError(msg)


# Curated scan-result keys passed to the model. The candidate setups are already
# compact dicts from analyze_single_symbol; the raw per-symbol `errors` list is
# intentionally excluded (the count is preserved in `summary`).
_SCAN_PROJECTION_KEYS = (
    "scan_parameters",
    "summary",
    "high_conviction_setups",
    "top_bullish",
    "top_bearish",
)


def _project_scan_results(scan_results: dict) -> dict:
    """Project scan results down to the high-signal fields a briefing needs.

    Keeps scan parameters, summary stats, and the already-compact candidate
    setups; drops the verbose raw per-symbol error list (its count lives in
    ``summary``). Missing keys are simply omitted so sparse inputs are safe.
    """
    return {key: scan_results[key] for key in _SCAN_PROJECTION_KEYS if key in scan_results}


def _as_dict(value: object) -> dict:
    """Return ``value`` if it is a dict, else an empty dict.

    Guards the nested ``.get()`` chains below against malformed inputs where a
    section is present but holds a non-dict value (so the projection degrades
    gracefully instead of raising AttributeError).
    """
    return value if isinstance(value, dict) else {}


def _project_deep_analysis(analysis: dict) -> dict:
    """Project a full options-analysis dict to its briefing-relevant essentials.

    ``run_options_analysis`` returns a deeply nested object (score breakdowns,
    raw indicator magnitudes, tuning parameters). Dumping several of these blows
    past the model's output budget and buries the signal. This keeps the headline
    call, key trend/volatility readings, support/resistance levels, interpreted
    volume signals, and the human-readable insights — dropping raw magnitudes and
    internal tuning. Defensive against sparse or malformed inputs (e.g. fallback/
    test dicts where a section is missing or holds a non-dict value).
    """
    projected: dict = {"symbol": analysis.get("symbol", "Unknown")}
    if "latest_price" in analysis:
        projected["latest_price"] = analysis["latest_price"]

    headline = analysis.get("headline")
    if headline:
        projected["headline"] = headline

    composite = analysis.get("composite_signal")
    if isinstance(composite, dict):
        projected["composite"] = {
            "score": composite.get("score"),
            "recommendation": composite.get("recommendation"),
            "signal_quality": composite.get("signal_quality"),
            "action": composite.get("action"),
        }

    trend = analysis.get("trend_analysis")
    if isinstance(trend, dict):
        adx = _as_dict(trend.get("adx"))
        rsi = _as_dict(trend.get("rsi"))
        projected["trend"] = {
            "adx": adx.get("value"),
            "trend_strength": adx.get("trend_strength"),
            "trend_direction": adx.get("trend_direction"),
            "rsi": rsi.get("value"),
            "rsi_condition": rsi.get("condition"),
            "rsi_divergence": rsi.get("divergence_type"),
        }

    volatility = analysis.get("volatility_analysis")
    if isinstance(volatility, dict):
        proxy = _as_dict(volatility.get("iv_percentile_proxy"))
        move = _as_dict(volatility.get("expected_move"))
        atr = _as_dict(volatility.get("atr"))
        bbands = _as_dict(volatility.get("bollinger_bands"))
        projected["volatility"] = {
            "hv_percentile": proxy.get("hv_percentile", proxy.get("percentile")),
            "hv_implication": proxy.get("options_implication"),
            "expected_move_pct": move.get("percent"),
            "upper_target": move.get("upper_target"),
            "lower_target": move.get("lower_target"),
            "atr_daily_range": atr.get("daily_range"),
            "stop_loss": atr.get("stop_loss_suggestion"),
            "bollinger_position": bbands.get("position"),
            "squeeze": bbands.get("squeeze_detected"),
        }

    profile = analysis.get("volume_profile")
    if isinstance(profile, dict):
        projected["key_levels"] = {
            "point_of_control": profile.get("point_of_control"),
            "value_area_high": profile.get("value_area_high"),
            "value_area_low": profile.get("value_area_low"),
            "current_position": profile.get("current_position"),
        }

    volume = analysis.get("volume_indicators")
    if isinstance(volume, dict):
        obv = _as_dict(volume.get("obv"))
        ad = _as_dict(volume.get("accumulation_distribution"))
        mfi = _as_dict(volume.get("mfi"))
        cmf = _as_dict(volume.get("cmf"))
        rvol = _as_dict(volume.get("relative_volume"))
        breakout = _as_dict(volume.get("volume_breakout"))
        projected["volume_signals"] = {
            "obv_trend": obv.get("trend"),
            "ad_signal": ad.get("signal"),
            "mfi_condition": mfi.get("condition"),
            "cmf_signal": cmf.get("signal"),
            "rvol": rvol.get("current_rvol"),
            "volume_breakout": breakout.get("is_breakout"),
        }

    insights = analysis.get("options_insights")
    if insights:
        projected["insights"] = insights

    return projected


def _build_user_message(scan_results: dict, deep_analyses: list[dict]) -> str:
    """Build the user message with a curated, high-signal projection of the data.

    Rather than dumping the full raw scan/analysis JSON (noisy and prone to
    truncation against the model's output cap), this projects each section to the
    fields a briefing needs. Every emitted ticker/level still comes straight from
    the scan/analysis data, so the model stays grounded in real inputs.
    """
    projected_scan = _project_scan_results(scan_results)

    user_content = "Generate a morning options trading briefing from this data.\n"
    user_content += "Base the briefing only on the curated data below.\n\n"
    user_content += "## Scan Results\n"
    user_content += f"```json\n{json.dumps(projected_scan, indent=2, default=str)}\n```\n\n"

    if deep_analyses:
        user_content += "## Deep Analysis (Top Candidates)\n"
        for analysis in deep_analyses:
            projected = _project_deep_analysis(analysis)
            symbol = projected.get("symbol", "Unknown")
            user_content += f"### {symbol}\n"
            user_content += f"```json\n{json.dumps(projected, indent=2, default=str)}\n```\n\n"

    return user_content


def _generate_anthropic(user_content: str, model: str, api_key: str) -> str:
    """Generate briefing using Anthropic Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    logger.info("Sending briefing request to Anthropic (%s)", model)

    message = client.messages.create(
        model=model,
        max_tokens=16384,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    briefing = message.content[0].text  # type: ignore[union-attr]
    logger.info(
        "Briefing generated: %d chars, %d input tokens, %d output tokens",
        len(briefing),
        message.usage.input_tokens,
        message.usage.output_tokens,
    )
    if message.stop_reason == "max_tokens":
        logger.warning("Briefing was TRUNCATED — output hit max_tokens limit")
        briefing += _TRUNCATION_WARNING

    return briefing


def _generate_gemini(user_content: str, model: str, api_key: str) -> str:
    """Generate briefing using Google Gemini API."""
    from google import genai

    client = genai.Client(api_key=api_key)

    logger.info("Sending briefing request to Gemini (%s)", model)

    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "max_output_tokens": 16384,
        },
    )

    briefing = response.text or ""
    logger.info("Briefing generated: %d chars", len(briefing))

    candidates = response.candidates
    finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    finish_reason_value = getattr(finish_reason, "name", finish_reason)
    if finish_reason_value and str(finish_reason_value) == "MAX_TOKENS":
        logger.warning("Briefing was TRUNCATED — output hit max_output_tokens limit")
        briefing += _TRUNCATION_WARNING

    return briefing
