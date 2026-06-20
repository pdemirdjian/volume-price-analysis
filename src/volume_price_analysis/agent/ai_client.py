"""AI integration for generating morning briefings.

Supports multiple providers via the AI_PROVIDER environment variable:
- "gemini" (default): Google Gemini API via google-genai SDK
- "anthropic": Anthropic Claude API via anthropic SDK
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a professional options trading analyst writing a morning briefing email.

Your audience is an experienced options trader who wants actionable intelligence,
not generic advice. Be specific about symbols, scores, levels, and strategies.

GROUNDING (CRITICAL): Use ONLY the tickers, prices, composite scores, key levels,
and indicator readings that appear in the data provided below. Every ticker symbol
you name MUST come from the scan results or deep-analysis data. Never invent,
guess, infer, or substitute a symbol, price, or level that is not present in the
input. If the data does not contain something, say so plainly instead of filling
the gap. Do not add tickers from memory or general market knowledge.

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

# Uppercase tokens that look like tickers but are domain acronyms or common
# words. Used to keep the anti-hallucination check high-precision (few false
# positives) so its warnings stay meaningful. This is a best-effort denylist for
# a logging guard, not a security boundary — a token here is simply never flagged
# as a hallucinated ticker, even if some entries (e.g. "DD", "IPO") happen to
# collide with real symbols.
_NON_TICKER_TOKENS: frozenset[str] = frozenset(
    {
        # Indicator / options acronyms emitted throughout the analysis output.
        "ADX",
        "RSI",
        "VWAP",
        "VWMA",
        "OBV",
        "MFI",
        "CMF",
        "ATR",
        "POC",
        "VAH",
        "VAL",
        "IV",
        "HV",
        "DTE",
        "ROC",
        "VPT",
        "DI",
        "EMA",
        "SMA",
        "MACD",
        "ATM",
        "OTM",
        "ITM",
        "PCR",
        "BB",
        "STD",
        "EM",
        "ADL",
        "OI",
        # Direction / strategy / emphasis words the model may capitalize.
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
        "STRONG",
        "WEAK",
        "HIGH",
        "LOW",
        "NO",
        "YES",
        "TREND",
        "EXTREME",
        "VOLUME",
        "BREAKOUT",
        "SQUEEZE",
        "EXPECTED",
        "MOVE",
        "WARNING",
        "URGENT",
        "BUY",
        "SELL",
        "HOLD",
        "LONG",
        "SHORT",
        "CALL",
        "CALLS",
        "PUT",
        "PUTS",
        "RISK",
        "ALERT",
        "WATCH",
        "NOTE",
        "KEY",
        "TARGET",
        "ENTRY",
        "EXIT",
        "STOP",
        "GAIN",
        "LOSS",
        # Generic English / market abbreviations.
        "A",
        "I",
        "AN",
        "THE",
        "AND",
        "OR",
        "IF",
        "IS",
        "IT",
        "TO",
        "IN",
        "ON",
        "AT",
        "BY",
        "OF",
        "AS",
        "BE",
        "US",
        "USD",
        "ETF",
        "ETFS",
        "AI",
        "PM",
        "AM",
        "EST",
        "EDT",
        "PST",
        "PDT",
        "UTC",
        "GMT",
        "EOD",
        "EPS",
        "PE",
        "YOY",
        "QOQ",
        "FY",
        "Q",
        "S",
        "P",
        "E",
        "R",
        "U",
        "N",
        "OK",
        "NA",
        "TBD",
        "FAQ",
        "CEO",
        "CFO",
        "FOMC",
        "FED",
        "GDP",
        "CPI",
        "DAY",
        "DAYS",
        "WEEK",
        "VS",
        "VIA",
        "PER",
        "MAX",
        "MIN",
        "AVG",
        "ETC",
        "ROI",
        "SEC",
        "IPO",
        "ATH",
        "YTD",
        "EV",
        "ER",
        "PT",
        "SL",
        "TP",
        "DD",
    }
)

# Matches a ticker-shaped token: an optional "$" cashtag, 1-5 uppercase letters,
# and an optional class-share suffix (e.g. BRK.B / BRK-B). Lookarounds prevent
# matching letters embedded in mixed-case words (e.g. the "P" in "iPhone").
_TICKER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.])(\$?)([A-Z]{1,5})(?:[.\-]([A-Z]{1,2}))?(?![A-Za-z0-9])"
)


def _normalize_symbol(symbol: str) -> str:
    """Uppercase and strip class-share separators for comparison (BRK.B -> BRKB)."""
    return re.sub(r"[^A-Z]", "", symbol.upper())


def _collect_input_symbols(scan_results: dict, deep_analyses: list[dict]) -> set[str]:
    """Collect every ticker symbol present anywhere in the scan/analysis input.

    Includes scan candidates (high-conviction / bullish / bearish) and symbols
    the scan attempted but errored on, plus every deep-analysis symbol. A symbol
    the scan reported on is *grounded*: naming it is not a hallucination.
    """
    symbols: set[str] = set()
    for key in ("high_conviction_setups", "top_bullish", "top_bearish", "errors"):
        for entry in scan_results.get(key, []) or []:
            if isinstance(entry, dict) and entry.get("symbol"):
                symbols.add(_normalize_symbol(str(entry["symbol"])))
    for analysis in deep_analyses or []:
        if isinstance(analysis, dict) and analysis.get("symbol"):
            symbols.add(_normalize_symbol(str(analysis["symbol"])))
    symbols.discard("")
    return symbols


def find_ungrounded_tickers(
    briefing: str, scan_results: dict, deep_analyses: list[dict]
) -> list[str]:
    """Return ticker-like tokens in the briefing that are absent from the input.

    Best-effort, high-precision heuristic backing the briefing anti-hallucination
    guardrail: it extracts cashtags and uppercase ticker-shaped tokens, drops
    known indicator/acronym/common words, and flags anything left that does not
    appear in the scan or deep-analysis symbols. Bare single-letter tokens are
    ignored (too noisy, e.g. "Plan B") unless written as an explicit cashtag.

    The result is a sorted, de-duplicated list of the offending tokens, suitable
    for logging. It is intentionally conservative: it favors missing a borderline
    case over emitting a false alarm.
    """
    allowed = _collect_input_symbols(scan_results, deep_analyses)
    ungrounded: set[str] = set()
    for match in _TICKER_PATTERN.finditer(briefing or ""):
        cashtag, core, suffix = match.group(1), match.group(2), match.group(3)
        token = core + (suffix or "")
        # Bare single letters ("A", "B", "F") are too noisy to treat as tickers.
        if not cashtag and len(token) == 1:
            continue
        # Skip known non-ticker acronyms/words unless explicitly cashtagged.
        if not cashtag and core in _NON_TICKER_TOKENS:
            continue
        normalized = _normalize_symbol(token)
        if normalized and normalized not in allowed:
            ungrounded.add(token)
    return sorted(ungrounded)


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
        briefing = _generate_anthropic(user_content, model, api_key)
    elif provider == "gemini":
        briefing = _generate_gemini(user_content, model, api_key)
    else:
        msg = f"Unknown AI provider: {provider!r}. Use 'gemini' or 'anthropic'."
        raise ValueError(msg)

    # Anti-hallucination guardrail: every ticker named in the briefing should
    # exist in the scan/analysis data we passed to the model. Log (don't block)
    # any that don't so grounding regressions are visible in the run logs.
    ungrounded = find_ungrounded_tickers(briefing, scan_results, deep_analyses)
    if ungrounded:
        logger.warning(
            "Briefing references %d ticker(s) absent from scan/analysis data "
            "(possible hallucination): %s",
            len(ungrounded),
            ", ".join(ungrounded),
        )

    return briefing


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
