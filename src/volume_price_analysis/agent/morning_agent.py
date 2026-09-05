"""Morning briefing agent - main orchestrator.

Usage:
    python -m volume_price_analysis.agent.morning_agent [--dry-run] [--no-ai]

Flags:
    --dry-run   Print briefing to stdout instead of sending email
    --no-ai     Skip AI briefing generation, email raw data instead
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from ..analysis import run_options_analysis, run_scan
from ..data_fetcher import fetch_stock_data
from .ai_client import generate_briefing
from .config import AgentConfig
from .email_sender import send_briefing_email, send_error_email, send_raw_data_email
from .regime import (
    REGIME_SMA_PERIOD,
    annotate_regime_conflicts,
    compute_market_regime,
    format_regime_header,
)

# Configure logging to stdout (Docker best practice)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class BriefingRunResult:
    """Outcome of a single morning-briefing run.

    ``degraded`` says the briefing was delivered but not at full quality;
    ``reason`` says why, so callers can log something more useful than a bare
    boolean.
    """

    degraded: bool
    reason: str | None
    regime: dict
    symbols_analyzed: list[str]
    email_sent: bool


async def run_morning_briefing(
    config: AgentConfig, dry_run: bool = False, no_ai: bool = False
) -> BriefingRunResult:
    """
    Execute the full morning briefing pipeline.

    1. Run market scan
    2. Run deep options analysis on top candidates
    3. Generate AI briefing (unless --no-ai)
    4. Send email (unless --dry-run)

    Returns:
        A BriefingRunResult describing how the run went.
    """
    start_time = time.monotonic()
    now = datetime.now(UTC)
    date_str = now.strftime("%Y-%m-%d")

    logger.info("Starting morning briefing for %s", date_str)

    # Step 1: Run market scan
    logger.info("Step 1: Scanning universe '%s'...", config.scan_universe)
    scan_results = await run_scan(
        universe=config.scan_universe,
        period="3mo",
        holding_period=14,
        min_score=2.0,
        min_adx=20,
        max_iv_percentile=70,
        min_avg_daily_volume=500_000,
        direction="any",
        max_results=15,
    )

    total_candidates = scan_results["summary"]["total_candidates"]
    logger.info(
        "Scan complete: %d candidates (%d bullish, %d bearish, %d high conviction)",
        total_candidates,
        scan_results["summary"]["bullish_setups"],
        scan_results["summary"]["bearish_setups"],
        scan_results["summary"]["high_conviction"],
    )

    # Step 1b: Market-regime check (PDE-66) — context only: counter-regime
    # picks are flagged but keep their high-conviction billing and priority.
    # The briefing must never die on this path, so any failure degrades to an
    # unknown verdict with the scan results left unannotated.
    logger.info("Step 1b: Market regime check (SPY close vs %d-day SMA)...", REGIME_SMA_PERIOD)
    try:
        regime = _fetch_market_regime()
        scan_results = annotate_regime_conflicts(scan_results, regime)
        conflict_count = sum(
            1 for c in scan_results.get("high_conviction_setups", []) if c.get("regime_conflict")
        )
        regime_header = format_regime_header(regime, conflict_count)
    except Exception:
        logger.exception("Regime annotation failed; continuing without it")
        regime = {"regime": "unknown", "reason": "regime check failed"}
        regime_header = format_regime_header(regime)
    logger.info("Regime verdict: %s", regime.get("regime", "unknown"))

    # Step 2: Deep analysis on top N candidates
    top_symbols = _get_top_symbols(scan_results, config.max_deep_analysis)
    logger.info("Step 2: Deep analysis on %d symbols: %s", len(top_symbols), top_symbols)

    deep_analyses = []
    for symbol in top_symbols:
        try:
            data = fetch_stock_data(symbol, None, None, "3mo")
            analysis = run_options_analysis(symbol, data, holding_period=14)
            deep_analyses.append(analysis)
            logger.info("  %s: score=%.1f", symbol, analysis["composite_signal"]["score"])
        except Exception:
            logger.exception("  Failed to analyze %s", symbol)

    elapsed_analysis = time.monotonic() - start_time
    logger.info("Analysis complete in %.1fs", elapsed_analysis)

    # Step 2b: Earnings guard — batch-fetch for all analysed symbols
    analysed_symbols = [a["symbol"] for a in deep_analyses if "symbol" in a]
    earnings_warnings = _fetch_earnings_warnings(analysed_symbols, now)
    if earnings_warnings:
        logger.info("Earnings warnings: %s", earnings_warnings)
        for analysis in deep_analyses:
            sym = analysis.get("symbol")
            if sym and sym in earnings_warnings:
                analysis["earnings_warning"] = earnings_warnings[sym]

    # Step 3: Generate briefing
    degraded_reason: str | None = None
    if no_ai:
        logger.info("Step 3: Skipping AI (--no-ai mode)")
        briefing = None
    else:
        logger.info(
            "Step 3: Generating AI briefing via %s (%s)...",
            config.ai_provider,
            config.ai_model or "default model",
        )
        earnings_preamble = build_earnings_preamble(earnings_warnings)
        try:
            briefing = generate_briefing(
                scan_results=scan_results,
                deep_analyses=deep_analyses,
                provider=config.ai_provider,
                model=config.ai_model,
                api_key=config.ai_provider_api_key,
                earnings_preamble=earnings_preamble,
            )
        except Exception:
            logger.exception("AI briefing generation failed")
            briefing = _fallback_briefing(scan_results, deep_analyses)
            degraded_reason = (
                f"AI briefing generation failed via {config.ai_provider}; used fallback briefing"
            )
            logger.warning("Using fallback briefing — AI provider was unavailable")

    # The regime verdict heads every rendered briefing (AI or fallback); the
    # no-AI raw email carries it inside the scan_results JSON instead.
    if briefing is not None:
        briefing = regime_header + "\n\n" + briefing

    # Step 4: Deliver
    elapsed_total = time.monotonic() - start_time
    stats_line = build_stats_line(
        elapsed_s=elapsed_total,
        symbols_scanned=scan_results.get("scan_parameters", {}).get("symbols_scanned"),
        total_candidates=total_candidates,
        deep_count=len(deep_analyses),
    )

    if dry_run:
        logger.info("Step 4: Dry run - printing to stdout")
        if briefing:
            print(briefing + stats_line)
        else:
            print(regime_header)
            print(json.dumps(scan_results, indent=2, default=str))
            for a in deep_analyses:
                print(json.dumps(a, indent=2, default=str))
    elif no_ai:
        logger.info("Step 4: Sending raw data email")
        send_raw_data_email(
            scan_results=scan_results,
            deep_analyses=deep_analyses,
            from_addr=config.email_from,
            password=config.email_password,
            to_addr=config.email_to,
            smtp_host=config.email_smtp_host,
            smtp_port=config.email_smtp_port,
            date_str=date_str,
            preamble=regime_header,
        )
    else:
        logger.info("Step 4: Sending briefing email")
        assert briefing is not None  # Always set when not no_ai
        subject = f"Morning Market Briefing - {date_str}"
        send_briefing_email(
            subject=subject,
            body_markdown=briefing + stats_line,
            from_addr=config.email_from,
            password=config.email_password,
            to_addr=config.email_to,
            smtp_host=config.email_smtp_host,
            smtp_port=config.email_smtp_port,
            ticker_symbols=_candidate_symbols(scan_results, deep_analyses),
        )

    logger.info("Morning briefing complete in %.1fs", elapsed_total)
    return BriefingRunResult(
        degraded=degraded_reason is not None,
        reason=degraded_reason,
        regime=regime,
        symbols_analyzed=analysed_symbols,
        email_sent=not dry_run,
    )


_EARNINGS_WARN_DAYS = 14

# Cap concurrent earnings lookups regardless of how many symbols were analysed
_EARNINGS_MAX_WORKERS = 8


def _check_earnings(symbol: str, now: datetime) -> str | None:
    """Return a warning string if the symbol has earnings within 14 days, else None."""
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info
        raw = info.get("earningsDate") or info.get("earningsTimestamp")
        if raw is None:
            return None

        # yfinance may return a list (range) or a single value
        if isinstance(raw, list):
            raw = raw[0]

        # Normalise to an aware datetime
        if isinstance(raw, (int, float)):
            earnings_dt = datetime.fromtimestamp(raw, tz=UTC)
        elif isinstance(raw, datetime):
            earnings_dt = raw if raw.tzinfo else raw.replace(tzinfo=UTC)
        else:
            return None

        delta = earnings_dt - now
        if timedelta(0) <= delta <= timedelta(days=_EARNINGS_WARN_DAYS):
            days_out = delta.days
            return f"EARNINGS in {days_out} day(s) ({earnings_dt.strftime('%Y-%m-%d')})"
        return None
    except Exception:
        logger.debug("Earnings lookup failed for %s", symbol, exc_info=True)
        return None


def _fetch_earnings_warnings(symbols: list[str], now: datetime) -> dict[str, str]:
    """Fetch earnings dates for all symbols concurrently. Returns symbol -> warning string."""
    if not symbols:
        return {}
    with ThreadPoolExecutor(max_workers=min(len(symbols), _EARNINGS_MAX_WORKERS)) as pool:
        futures = {sym: pool.submit(_check_earnings, sym, now) for sym in symbols}
        result: dict[str, str] = {}
        for sym, fut in futures.items():
            warning = fut.result()
            if warning is not None:
                result[sym] = warning
        return result


def _fetch_market_regime() -> dict:
    """Fetch SPY history and compute the market regime; failures degrade to unknown."""
    try:
        spy_data = fetch_stock_data("SPY", None, None, "3mo")
        # Exclude any in-progress session so a manual intraday run stays strictly
        # causal — the check must always read the prior session's close.
        today_eastern = datetime.now(ZoneInfo("America/New_York")).date()
        return compute_market_regime(spy_data, today=today_eastern)
    except Exception:
        logger.exception("Market regime check failed")
        return compute_market_regime(None)


def _get_top_symbols(scan_results: dict, max_count: int) -> list[str]:
    """Extract top candidate symbols from scan results for deep analysis."""
    symbols = []
    seen = set()

    # Prioritize high conviction setups
    for candidate in scan_results.get("high_conviction_setups", []):
        sym = candidate["symbol"]
        if sym not in seen:
            symbols.append(sym)
            seen.add(sym)

    # Then top bullish
    for candidate in scan_results.get("top_bullish", []):
        sym = candidate["symbol"]
        if sym not in seen:
            symbols.append(sym)
            seen.add(sym)

    # Then top bearish
    for candidate in scan_results.get("top_bearish", []):
        sym = candidate["symbol"]
        if sym not in seen:
            symbols.append(sym)
            seen.add(sym)

    return symbols[:max_count]


def _candidate_symbols(scan_results: dict, deep_analyses: list[dict]) -> set[str]:
    """Collect every symbol the briefing may mention, for ticker linkification."""
    symbols = {
        candidate["symbol"]
        for key in ("high_conviction_setups", "top_bullish", "top_bearish")
        for candidate in scan_results.get(key, [])
        if "symbol" in candidate
    }
    symbols.update(a["symbol"] for a in deep_analyses if "symbol" in a)
    return symbols


def _fallback_briefing(scan_results: dict, deep_analyses: list[dict]) -> str:
    """Generate a basic text briefing when Claude API fails."""
    lines = ["# Morning Market Briefing (Fallback - AI unavailable)\n"]

    summary = scan_results.get("summary", {})
    lines.append(f"**Candidates found:** {summary.get('total_candidates', 0)}")
    lines.append(f"**Bullish setups:** {summary.get('bullish_setups', 0)}")
    lines.append(f"**Bearish setups:** {summary.get('bearish_setups', 0)}")
    lines.append(f"**High conviction:** {summary.get('high_conviction', 0)}\n")

    conflict_symbols = {
        c.get("symbol")
        for c in scan_results.get("high_conviction_setups", [])
        if isinstance(c, dict) and c.get("regime_conflict")
    }
    if deep_analyses:
        lines.append("## Top Candidates\n")
        for a in deep_analyses:
            sym = a.get("symbol", "?")
            score = a.get("composite_signal", {}).get("score", 0)
            rec = a.get("composite_signal", {}).get("recommendation", "?")
            price = a.get("latest_price", 0)
            line = f"- **{sym}** @ ${price:.2f} | Score: {score:.1f} | {rec}"
            if sym in conflict_symbols:
                line += " | ⚠️ counter-regime setup"
            lines.append(line)

    return "\n".join(lines)


def build_earnings_preamble(warnings: dict[str, str]) -> str:
    """Render the earnings-risk preamble prepended to the AI prompt.

    Returns an empty string when no analysed symbol has upcoming earnings.
    """
    if not warnings:
        return ""
    lines = [f"  - {sym}: {warn}" for sym, warn in sorted(warnings.items())]
    return (
        "\n\n**EARNINGS EVENT RISK** — the following candidates have earnings "
        f"within {_EARNINGS_WARN_DAYS} days. Factor event risk into sizing and strategy:\n"
        + "\n".join(lines)
        + "\n"
    )


def build_stats_line(
    elapsed_s: float,
    symbols_scanned: int | None,
    total_candidates: int,
    deep_count: int,
) -> str:
    """Render the footer appended to every delivered briefing."""
    scanned_part = f"{symbols_scanned} symbols scanned | " if symbols_scanned else ""
    return (
        f"\n\n---\n"
        f"**14-day holding period** - Indicators, expected moves, and strategies "
        f"are calibrated for approx. 14 DTE options. Shorter-duration plays (0-5 DTE) "
        f"may need different setups.\n\n"
        f"*Generated in {elapsed_s:.1f}s | "
        f"{scanned_part}"
        f"{total_candidates} candidates found | "
        f"{deep_count} deep analyses*"
    )


def _config_errors(config: AgentConfig, dry_run: bool, no_ai: bool) -> list[str]:
    """Select the config errors that actually block this run mode.

    A dry run never emails, and --no-ai never calls a provider, so each mode
    ignores the half of ``AgentConfig.validate()`` it cannot trip over.
    """
    errors = config.validate()
    if dry_run and no_ai:
        return []
    if dry_run:
        return [e for e in errors if "API_KEY" in e or "AI_PROVIDER" in e]
    if no_ai:
        return [e for e in errors if "API_KEY" not in e and "AI_PROVIDER" not in e]
    return errors


def main():
    """Entry point for the morning briefing agent."""
    parser = argparse.ArgumentParser(description="Morning market briefing agent")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout, don't email")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI generation, send raw data")
    args = parser.parse_args()

    config = AgentConfig.from_env()

    errors = _config_errors(config, dry_run=args.dry_run, no_ai=args.no_ai)
    if errors:
        for error in errors:
            logger.error("Config error: %s", error)
        sys.exit(1)

    try:
        result = asyncio.run(run_morning_briefing(config, dry_run=args.dry_run, no_ai=args.no_ai))
        if result.degraded:
            sys.exit(2)
    except Exception as e:
        logger.exception("Morning briefing failed critically")
        # Try to send error notification
        if not args.dry_run and config.email_from and config.email_password and config.email_to:
            send_error_email(
                error_message=str(e),
                from_addr=config.email_from,
                password=config.email_password,
                to_addr=config.email_to,
                smtp_host=config.email_smtp_host,
                smtp_port=config.email_smtp_port,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
