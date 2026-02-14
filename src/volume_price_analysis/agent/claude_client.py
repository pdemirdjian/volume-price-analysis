"""Claude API integration for generating morning briefings."""

import json
import logging

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a professional options trading analyst writing a morning briefing email.

Your audience is an experienced options trader who wants actionable intelligence,
not generic advice. Be specific about symbols, scores, levels, and strategies.

Format the briefing in markdown with clear sections:
1. **Executive Summary** - 2-3 sentence overview of today's market setup
2. **Top Picks** - For each high-conviction candidate, include:
   - Symbol, price, composite score, and direction (bullish/bearish)
   - Key levels (support/resistance from volume profile)
   - Suggested strategy (calls, puts, spreads, etc.)
   - Risk factors to watch
3. **Market Context** - Overall scan statistics, sector themes
4. **Risk Warnings** - Any divergences, extreme readings, or caution flags

Keep it concise but thorough. Use bullet points. No fluff."""


def generate_briefing(
    scan_results: dict,
    deep_analyses: list[dict],
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str = "",
) -> str:
    """
    Send scan results and deep analyses to Claude for a natural-language briefing.

    Args:
        scan_results: Output from run_scan().
        deep_analyses: List of outputs from run_options_analysis() for top candidates.
        model: Claude model to use.
        api_key: Anthropic API key.

    Returns:
        Markdown-formatted briefing text.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Build the user message with structured data
    user_content = "Generate a morning options trading briefing from this data:\n\n"
    user_content += "## Scan Results\n"
    user_content += f"```json\n{json.dumps(scan_results, indent=2, default=str)}\n```\n\n"

    if deep_analyses:
        user_content += "## Deep Analysis (Top Candidates)\n"
        for analysis in deep_analyses:
            symbol = analysis.get("symbol", "Unknown")
            user_content += f"### {symbol}\n"
            user_content += f"```json\n{json.dumps(analysis, indent=2, default=str)}\n```\n\n"

    logger.info("Sending briefing request to Claude (%s)", model)

    message = client.messages.create(
        model=model,
        max_tokens=4096,
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

    return briefing
