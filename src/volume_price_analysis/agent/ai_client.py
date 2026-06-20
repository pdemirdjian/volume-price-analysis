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

GROUNDING (CRITICAL): Use ONLY the tickers, prices, composite scores, and price
levels present in the structured data provided in the user message. Never invent,
recall, or guess a symbol, number, or level from memory. Every ticker you mention
MUST appear in the provided scan results or deep analysis. If a value is not in
the data, do not state it — say the data is unavailable instead. Do not introduce
symbols that were not scanned.

IMPORTANT: All analysis is optimized for a **14-day holding period**. Indicator
periods, expected moves, and strategy suggestions are calibrated for this
timeframe. Make this clear in your strategy suggestions (e.g., "14-day calls"
not just "calls"). Readers doing shorter-duration plays (e.g., 0-5 DTE) should
note that signals and levels may not apply to their timeframe.

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

# Matches a stock-ticker-like token: 1-5 uppercase letters, optionally a leading
# cashtag ($AAPL) and an optional share-class suffix (BRK.B / BRK-B). Group 1 is
# the cashtag marker; group 2 is the symbol itself.
_TICKER_RE = re.compile(r"(\$)?\b([A-Z]{1,5}(?:[.\-][A-Z]{1,2})?)\b")

# Uppercase tokens that legitimately appear in briefings but are NOT tickers.
# Used to suppress false positives in the grounding check. Covers the indicator
# vocabulary emitted by indicators.py, common options/trading shorthand, and
# short English/markdown words that may appear capitalized in headers or prose.
_NON_TICKER_TOKENS = frozenset(
    {
        # Technical indicators / studies
        "OBV",
        "VWAP",
        "VPT",
        "MFI",
        "ATR",
        "HV",
        "IV",
        "CMF",
        "VWMA",
        "ROC",
        "DI",
        "ADX",
        "ADXR",
        "RSI",
        "BB",
        "AD",
        "RVOL",
        "MACD",
        "EMA",
        "SMA",
        "POC",
        "VA",
        "VAH",
        "VAL",
        "OHLC",
        "ATL",
        "ATH",
        # Options / trading shorthand
        "DTE",
        "OTM",
        "ITM",
        "ATM",
        "OI",
        "PUT",
        "PUTS",
        "CALL",
        "CALLS",
        "PE",
        "EPS",
        "ETF",
        "ETFS",
        "YOY",
        "QOQ",
        "YTD",
        "BUY",
        "SELL",
        "HOLD",
        "LONG",
        "PT",
        "SL",
        "TP",
        "RR",
        # General / units / time
        "AI",
        "CEO",
        "CFO",
        "USD",
        "ET",
        "EST",
        "EDT",
        "UTC",
        "AM",
        "PM",
        "FAQ",
        "TLDR",
        "FYI",
        "Q",
        # Macro / index / market-structure acronyms (context, not equity picks)
        "CPI",
        "PPI",
        "PCE",
        "GDP",
        "PMI",
        "ISM",
        "NFP",
        "FOMC",
        "FED",
        "ECB",
        "BOJ",
        "BOE",
        "FX",
        "VIX",
        "SPX",
        "NDX",
        "DJI",
        "DJIA",
        "RUT",
        "ER",
        "TTM",
        "FCF",
        "ROE",
        "ROA",
        "GAAP",
        # Short English / markdown words that may appear capitalized
        "AN",
        "AND",
        "OR",
        "THE",
        "FOR",
        "TO",
        "OF",
        "ON",
        "AT",
        "IN",
        "IS",
        "BE",
        "BY",
        "AS",
        "IF",
        "IT",
        "NO",
        "SO",
        "UP",
        "WE",
        "US",
        "TOP",
        "KEY",
        "RISK",
        "LOW",
        "HIGH",
        "BULL",
        "BEAR",
    }
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
        briefing = _generate_anthropic(user_content, model, api_key)
    elif provider == "gemini":
        briefing = _generate_gemini(user_content, model, api_key)
    else:
        msg = f"Unknown AI provider: {provider!r}. Use 'gemini' or 'anthropic'."
        raise ValueError(msg)

    # Anti-hallucination guardrail: flag any ticker in the briefing that was not
    # in the input data. Logs only — the briefing text is never modified.
    _check_briefing_grounding(briefing, scan_results, deep_analyses)
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


def _collect_known_symbols(scan_results: dict, deep_analyses: list[dict]) -> set[str]:
    """Gather every ticker symbol present in the scan/analysis input data.

    Includes candidate setups, errored symbols, and deep-analysis symbols — i.e.
    every symbol the model was actually shown, normalized to uppercase.
    """
    known: set[str] = set()
    for key in ("high_conviction_setups", "top_bullish", "top_bearish", "errors"):
        for item in scan_results.get(key) or []:
            sym = item.get("symbol") if isinstance(item, dict) else None
            if sym:
                known.add(str(sym).upper())
    for item in deep_analyses or []:
        sym = item.get("symbol") if isinstance(item, dict) else None
        if sym:
            known.add(str(sym).upper())
    return known


def _check_briefing_grounding(
    briefing: str, scan_results: dict, deep_analyses: list[dict]
) -> list[str]:
    """Flag ticker-like tokens in the briefing that are absent from the input data.

    A guardrail against LLM hallucination: the briefing must only reference symbols
    that were actually scanned or analyzed. Any unrecognized symbols are logged at
    WARNING level and returned (sorted, de-duplicated). The briefing text is never
    modified — this is a detection/alerting check, not a filter.
    """
    # Compare on de-punctuated symbols so share classes (BRK.B / BRK-B) and
    # dotted abbreviations ("U.S." -> "US") normalize consistently.
    known = {
        sym.replace(".", "").replace("-", "")
        for sym in _collect_known_symbols(scan_results, deep_analyses)
    }
    unknown: set[str] = set()
    for match in _TICKER_RE.finditer(briefing):
        is_cashtag = bool(match.group(1))
        token = match.group(2).upper()
        norm = token.replace(".", "").replace("-", "")
        if norm in known:
            continue
        # Bare single letters in prose ("I", "A", "U.S.") are too noisy to treat
        # as tickers; only honor them when explicitly cashtagged.
        if not is_cashtag and len(norm) == 1:
            continue
        if not is_cashtag and norm in _NON_TICKER_TOKENS:
            continue
        unknown.add(token)

    result = sorted(unknown)
    if result:
        logger.warning(
            "Briefing grounding check: %d unrecognized ticker(s) not in scan data: %s",
            len(result),
            ", ".join(result),
        )
    return result


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
