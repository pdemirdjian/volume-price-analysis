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


def _build_user_message(scan_results: dict, deep_analyses: list[dict]) -> str:
    """Build the user message with structured data for the AI."""
    user_content = "Generate a morning options trading briefing from this data:\n\n"
    user_content += "## Scan Results\n"
    user_content += f"```json\n{json.dumps(scan_results, indent=2, default=str)}\n```\n\n"

    if deep_analyses:
        user_content += "## Deep Analysis (Top Candidates)\n"
        for analysis in deep_analyses:
            symbol = analysis.get("symbol", "Unknown")
            user_content += f"### {symbol}\n"
            user_content += f"```json\n{json.dumps(analysis, indent=2, default=str)}\n```\n\n"

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
