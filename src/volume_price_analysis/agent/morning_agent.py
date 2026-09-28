"""Morning briefing agent - main orchestrator.

Usage:
    python -m volume_price_analysis.agent.morning_agent [--dry-run] [--no-ai]

Flags:
    --dry-run   Print briefing to stdout instead of sending email
    --no-ai     Skip AI briefing generation, email raw data instead
"""

import argparse
import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from ..analysis import run_options_analysis, run_scan
from ..data_fetcher import DataSource, get_default_data_source
from .ai_client import PROVIDERS, generate_briefing, resolve_model
from .briefing import BriefingInputs, render, render_raw
from .config import AgentConfig
from .email_sender import SmtpCreds, build_briefing_message, build_error_message, send_email
from .picks import build_picks
from .regime import REGIME_SMA_PERIOD, annotate_regime_conflicts, compute_market_regime

# Configure logging to stdout (Docker best practice)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Briefings are dated in market time: the scheduler fires at 08:30 ET, and a
# UTC date would roll over at 20:00 ET the evening before.
MARKET_TZ = ZoneInfo("America/New_York")


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
    config: AgentConfig,
    dry_run: bool = False,
    no_ai: bool = False,
    data_source: DataSource | None = None,
    now: datetime | None = None,
) -> BriefingRunResult:
    """
    Execute the full morning briefing pipeline.

    1. Run market scan
    2. Run deep options analysis on top candidates
    3. Generate AI briefing (unless --no-ai)
    4. Send email (unless --dry-run)

    Args:
        data_source: Market-data seam used for the scan, the SPY regime fetch,
            per-symbol fetches, and earnings checks. None uses production.
        now: The run's clock reading (aware datetime); the briefing date is its
            Eastern calendar day. None reads the wall clock.

    Returns:
        A BriefingRunResult describing how the run went.
    """
    source = data_source if data_source is not None else get_default_data_source()
    start_time = time.monotonic()
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        raise ValueError("now must be an aware datetime")
    briefing_date = now.astimezone(MARKET_TZ).date()
    date_str = briefing_date.isoformat()

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
        data_source=source,
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
        regime = _fetch_market_regime(source, today=briefing_date)
        scan_results = annotate_regime_conflicts(scan_results, regime)
    except Exception:
        logger.exception("Regime annotation failed; continuing without it")
        regime = {"regime": "unknown", "reason": "regime check failed"}
    logger.info("Regime verdict: %s", regime.get("regime", "unknown"))

    # Step 2: Deep analysis on top N candidates
    top_symbols = _get_top_symbols(scan_results, config.max_deep_analysis)
    logger.info("Step 2: Deep analysis on %d symbols: %s", len(top_symbols), top_symbols)

    deep_analyses = []
    for symbol in top_symbols:
        try:
            data = source.fetch(symbol, period="3mo")
            analysis = run_options_analysis(symbol, data, holding_period=14)
            deep_analyses.append(analysis)
            logger.info("  %s: score=%.1f", symbol, analysis["composite_signal"]["score"])
        except Exception:
            logger.exception("  Failed to analyze %s", symbol)

    elapsed_analysis = time.monotonic() - start_time
    logger.info("Analysis complete in %.1fs", elapsed_analysis)

    # Step 2b: Earnings guard — batch-fetch for all analysed symbols
    analysed_symbols = [a["symbol"] for a in deep_analyses if "symbol" in a]
    earnings_warnings = _fetch_earnings_warnings(analysed_symbols, now, source)
    if earnings_warnings:
        logger.info("Earnings warnings: %s", earnings_warnings)
        for analysis in deep_analyses:
            sym = analysis.get("symbol")
            if sym and sym in earnings_warnings:
                analysis["earnings_warning"] = earnings_warnings[sym]

    # Step 2c: Fixed conviction vocabulary (PDE-69/PDE-150). The picks builder
    # takes the regime, applies regime-conflict annotation itself and derives
    # conviction once; the deep analyses, the AI prompt and the rendered pick
    # table all read the label off this one list, so they cannot disagree and
    # no ordering above is load-bearing.
    picks = build_picks(scan_results, deep_analyses, regime=regime)
    conviction_by_symbol = {p.symbol: p.conviction for p in picks}
    for analysis in deep_analyses:
        sym = analysis.get("symbol")
        if sym in conviction_by_symbol:
            analysis["conviction"] = conviction_by_symbol[sym]

    # Step 3: Generate briefing. model_text None renders the fallback body.
    degraded_reason: str | None = None
    model_text: str | None = None
    if no_ai:
        logger.info("Step 3: Skipping AI (--no-ai mode)")
    else:
        logger.info(
            "Step 3: Generating AI briefing via %s (%s)...",
            config.ai_provider,
            config.ai_model or "default model",
        )
        earnings_preamble = build_earnings_preamble(earnings_warnings)
        try:
            model_text = generate_briefing(
                scan_results=scan_results,
                deep_analyses=deep_analyses,
                provider=PROVIDERS[config.ai_provider],
                model=resolve_model(config.ai_provider, config.ai_model),
                api_key=config.ai_provider_api_key,
                earnings_preamble=earnings_preamble,
                briefing_date=briefing_date,
                picks=picks,
            ).text
        except Exception:
            logger.exception("AI briefing generation failed")
            degraded_reason = (
                f"AI briefing generation failed via {config.ai_provider}; used fallback briefing"
            )
            logger.warning("Using fallback briefing — AI provider was unavailable")

    # Step 4: Deliver. The Briefing module owns the whole document; the no-AI
    # path renders the raw-data body instead.
    elapsed_total = time.monotonic() - start_time
    inputs = BriefingInputs(
        scan_results=scan_results,
        deep_analyses=deep_analyses,
        regime=regime,
        briefing_date=briefing_date,
        elapsed_s=elapsed_total,
        model_text=model_text,
    )
    body = render_raw(inputs) if no_ai else render(inputs)

    if dry_run:
        logger.info("Step 4: Dry run - printing to stdout")
        print(body)
    elif no_ai:
        logger.info("Step 4: Sending raw data email")
        creds = SmtpCreds.from_config(config)
        send_email(
            build_briefing_message(
                creds, subject=f"Morning Market Data (Raw) - {date_str}", body_markdown=body
            ),
            creds,
        )
    else:
        logger.info("Step 4: Sending briefing email")
        creds = SmtpCreds.from_config(config)
        send_email(
            build_briefing_message(
                creds,
                subject=f"Morning Market Briefing - {date_str}",
                body_markdown=body,
                ticker_symbols=_candidate_symbols(scan_results, deep_analyses),
            ),
            creds,
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
EARNINGS_TIMEOUT_SECONDS = 30


def _check_earnings(symbol: str, now: datetime, source: DataSource) -> str | None:
    """Return a warning string if the symbol has earnings within 14 days, else None."""
    try:
        earnings_dt = source.earnings_date(symbol)
        if earnings_dt is None:
            return None
        if earnings_dt.tzinfo is None:
            earnings_dt = earnings_dt.replace(tzinfo=UTC)

        delta = earnings_dt - now
        if timedelta(0) <= delta <= timedelta(days=_EARNINGS_WARN_DAYS):
            days_out = delta.days
            return f"EARNINGS in {days_out} day(s) ({earnings_dt.strftime('%Y-%m-%d')})"
        return None
    except Exception:
        logger.debug("Earnings lookup failed for %s", symbol, exc_info=True)
        return None


def _fetch_earnings_warnings(
    symbols: list[str], now: datetime, source: DataSource
) -> dict[str, str]:
    """Fetch earnings dates for all symbols concurrently. Returns symbol -> warning string."""
    if not symbols:
        return {}
    pool = ThreadPoolExecutor(max_workers=min(len(symbols), _EARNINGS_MAX_WORKERS))
    try:
        futures = {sym: pool.submit(_check_earnings, sym, now, source) for sym in symbols}
        result: dict[str, str] = {}
        for sym, fut in futures.items():
            try:
                warning = fut.result(timeout=EARNINGS_TIMEOUT_SECONDS)
            except TimeoutError:
                fut.cancel()
                logger.debug("Earnings lookup failed for %s", sym, exc_info=True)
                continue
            if warning is not None:
                result[sym] = warning
        return result
    finally:
        # A context manager waits for running lookups, undoing the timeout.
        # Running threads cannot be cancelled; let the briefing continue.
        pool.shutdown(wait=False, cancel_futures=True)


def _fetch_market_regime(source: DataSource, today: date | None = None) -> dict:
    """Fetch SPY history and compute the market regime; failures degrade to unknown.

    ``today`` is the briefing's Eastern calendar day (defaults to the wall
    clock's). Bars dated on or after it are excluded so a manual intraday run
    stays strictly causal — the check always reads the prior session's close.
    """
    try:
        spy_data = source.fetch("SPY", period="3mo")
        if today is None:
            today = datetime.now(MARKET_TZ).date()
        return compute_market_regime(spy_data, today=today)
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
            try:
                creds = SmtpCreds.from_config(config)
                send_email(build_error_message(creds, str(e)), creds)
            except Exception:
                logger.exception("Failed to send error email")
        sys.exit(1)


if __name__ == "__main__":
    main()
