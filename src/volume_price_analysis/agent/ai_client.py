"""AI integration for generating morning briefings.

Supports multiple providers via the AI_PROVIDER environment variable:
- "gemini" (default): Google Gemini API via google-genai SDK
- "anthropic": Anthropic Claude API via anthropic SDK
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Maximum output tokens requested from any provider. Shared so both adapters
# stay in lockstep; override per call with the ``max_tokens`` keyword.
MAX_OUTPUT_TOKENS = 16384

# A briefing provider turns (user_content, model, api_key) into briefing text.
# The two production adapters below satisfy it; tests inject plain callables.
BriefingProvider = Callable[[str, str, str], str]


@dataclass(frozen=True)
class BriefingResult:
    """A generated briefing plus the grounding-check verdict for it.

    ``ungrounded_tickers`` lists ticker-like tokens the model named that are
    absent from the scan/analysis input (see :func:`find_ungrounded_tickers`).
    It is advisory: the briefing is returned regardless, and callers decide
    whether to act on it.
    """

    text: str
    ungrounded_tickers: list[str] = field(default_factory=list)


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
inferences from HV, not measured IV. Note the percentile ranks the symbol
against its OWN recent history: a low HV percentile means volatility is low
for that symbol, not that the expected move is small in absolute terms.

MARKET REGIME: The scan data may include a "market_regime" verdict (SPY's
prior-session close vs its 20-day SMA), and individual picks may carry a
"regime_conflict" note meaning their direction fights that regime. State the
regime verdict in the Executive Summary. Keep flagged picks ranked exactly as
scanned, but note the conflict explicitly wherever such a pick is presented
and repeat it under Risk Warnings. Treat the regime as context for the
reader, not a proven edge filter — never drop or reorder picks because of it.

CONSISTENCY: Cite exactly ONE value per metric per symbol. If a symbol appears
in both the scan results and the deep analysis, use the deep-analysis values
for prices, targets, and levels — never quote a second, conflicting number for
the same metric elsewhere in the briefing. The "upper_target" / "lower_target"
fields are a ±1 standard deviation expected move over the holding period;
label them as such (e.g., "14-day ±1σ upper target"), not as predictions or
price objectives.

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
    "gemini": "gemini-2.5-pro",
    "anthropic": "claude-sonnet-4-6",
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


def resolve_model(provider_name: str, model: str = "") -> str:
    """Return ``model`` if set, else the default model for ``provider_name``."""
    return model or DEFAULT_MODELS.get(provider_name, DEFAULT_MODELS["gemini"])


def generate_briefing(
    scan_results: dict,
    deep_analyses: list[dict],
    provider: BriefingProvider,
    model: str,
    api_key: str,
    earnings_preamble: str = "",
) -> BriefingResult:
    """
    Generate a natural-language briefing from scan results and deep analyses.

    Args:
        scan_results: Output from run_scan().
        deep_analyses: List of outputs from run_options_analysis() for top candidates.
        provider: Callable taking (user_content, model, api_key) and returning briefing
            text. Use ``PROVIDERS[name]`` to pick a production adapter.
        model: Model name passed through to the provider.
        api_key: API key passed through to the provider.
        earnings_preamble: Optional earnings-event-risk warning block prepended to user message.

    Returns:
        A BriefingResult carrying the markdown briefing and any ungrounded tickers.
    """
    user_content = build_briefing_prompt(scan_results, deep_analyses, earnings_preamble)

    briefing = provider(user_content, model, api_key)

    # Anti-hallucination guardrail: every ticker named in the briefing should
    # exist in the scan/analysis data we passed to the model. Log (don't block)
    # any that don't so grounding regressions are visible in the run logs, and
    # surface them on the result so callers can act on them too.
    ungrounded = find_ungrounded_tickers(briefing, scan_results, deep_analyses)
    if ungrounded:
        logger.warning(
            "Briefing references %d ticker(s) absent from scan/analysis data "
            "(possible hallucination): %s",
            len(ungrounded),
            ", ".join(ungrounded),
        )

    return BriefingResult(text=briefing, ungrounded_tickers=ungrounded)


# Curated scan-result keys passed to the model. The candidate setups are already
# compact dicts from analyze_single_symbol; the raw per-symbol `errors` list is
# intentionally excluded (the count is preserved in `summary`).
_SCAN_PROJECTION_KEYS = (
    "scan_parameters",
    "summary",
    "market_regime",
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


# Scan-candidate fields superseded by the deep analysis for the same symbol.
# The scan and the deep analysis compute expected-move targets from different
# data windows, so passing both invites the model to cite two conflicting
# targets for one symbol (e.g. a "$40.32 scan target" next to a "$39.75
# 14-day lower target").
_SCAN_FIELDS_SUPERSEDED_BY_DEEP = ("key_levels", "expected_move_pct")


def _drop_superseded_scan_fields(projected_scan: dict, deep_analyses: list[dict]) -> dict:
    """Strip scan-level targets from candidates that also have a deep analysis.

    Returns a copy; candidate dicts are shared with the caller's scan results
    and must not be mutated.
    """
    deep_symbols = {
        a.get("symbol") for a in deep_analyses or [] if isinstance(a, dict) and a.get("symbol")
    }
    if not deep_symbols:
        return projected_scan

    result = dict(projected_scan)
    for key in ("high_conviction_setups", "top_bullish", "top_bearish"):
        candidates = result.get(key)
        if not isinstance(candidates, list):
            continue
        result[key] = [
            {k: v for k, v in c.items() if k not in _SCAN_FIELDS_SUPERSEDED_BY_DEEP}
            if isinstance(c, dict) and c.get("symbol") in deep_symbols
            else c
            for c in candidates
        ]
    return result


def build_briefing_prompt(
    scan_results: dict, deep_analyses: list[dict], earnings_preamble: str = ""
) -> str:
    """Build the user message sent to a briefing provider.

    Runs the projection chain (curate scan results, drop scan fields superseded
    by a deep analysis, project each deep analysis) and renders the result as
    JSON blocks under markdown headings, optionally preceded by an
    earnings-event-risk preamble. Pairs with :data:`SYSTEM_PROMPT`, which the
    provider adapters supply as the system instruction.
    """
    projected_scan = _project_scan_results(scan_results)
    projected_scan = _drop_superseded_scan_fields(projected_scan, deep_analyses)
    user_content = "Generate a morning options trading briefing from this data:\n\n"
    if earnings_preamble:
        user_content += earnings_preamble + "\n"
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


def generate_anthropic(
    user_content: str, model: str, api_key: str, *, max_tokens: int = MAX_OUTPUT_TOKENS
) -> str:
    """Generate briefing using Anthropic Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    logger.info("Sending briefing request to Anthropic (%s)", model)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
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


def generate_gemini(
    user_content: str, model: str, api_key: str, *, max_tokens: int = MAX_OUTPUT_TOKENS
) -> str:
    """Generate briefing using Google Gemini API."""
    from google import genai

    client = genai.Client(api_key=api_key)

    logger.info("Sending briefing request to Gemini (%s)", model)

    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "max_output_tokens": max_tokens,
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


# Production provider adapters keyed by AI_PROVIDER name. AgentConfig.validate()
# is the single validator of provider names; a KeyError here means config was
# bypassed.
PROVIDERS: dict[str, BriefingProvider] = {
    "anthropic": generate_anthropic,
    "gemini": generate_gemini,
}
