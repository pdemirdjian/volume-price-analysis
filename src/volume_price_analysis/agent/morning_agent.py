"""Morning briefing agent - main orchestrator.

Usage:
    python -m volume_price_analysis.agent.morning_agent [--dry-run] [--no-ai]

Flags:
    --dry-run   Print briefing to stdout instead of sending email
    --no-ai     Skip Claude API, email raw data instead
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime

from ..analysis import run_options_analysis, run_scan
from ..data_fetcher import fetch_stock_data
from .claude_client import generate_briefing
from .config import AgentConfig
from .email_sender import send_briefing_email, send_error_email, send_raw_data_email

# Configure logging to stdout (Docker best practice)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def run_morning_briefing(config: AgentConfig, dry_run: bool = False, no_ai: bool = False):
    """
    Execute the full morning briefing pipeline.

    1. Run market scan
    2. Run deep options analysis on top candidates
    3. Generate AI briefing (unless --no-ai)
    4. Send email (unless --dry-run)
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
        max_iv_percentile=100,
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

    # Step 3: Generate briefing
    if no_ai:
        logger.info("Step 3: Skipping AI (--no-ai mode)")
        briefing = None
    else:
        logger.info("Step 3: Generating AI briefing via %s...", config.claude_model)
        try:
            briefing = generate_briefing(
                scan_results=scan_results,
                deep_analyses=deep_analyses,
                model=config.claude_model,
                api_key=config.anthropic_api_key,
            )
        except Exception:
            logger.exception("Claude API call failed")
            briefing = _fallback_briefing(scan_results, deep_analyses)

    # Step 4: Deliver
    elapsed_total = time.monotonic() - start_time
    stats_line = (
        f"\n\n---\n*Generated in {elapsed_total:.1f}s | "
        f"{total_candidates} candidates scanned | "
        f"{len(deep_analyses)} deep analyses*"
    )

    if dry_run:
        logger.info("Step 4: Dry run - printing to stdout")
        if briefing:
            print(briefing + stats_line)
        else:
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
        )

    logger.info("Morning briefing complete in %.1fs", elapsed_total)


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


def _fallback_briefing(scan_results: dict, deep_analyses: list[dict]) -> str:
    """Generate a basic text briefing when Claude API fails."""
    lines = ["# Morning Market Briefing (Fallback - AI unavailable)\n"]

    summary = scan_results.get("summary", {})
    lines.append(f"**Candidates found:** {summary.get('total_candidates', 0)}")
    lines.append(f"**Bullish setups:** {summary.get('bullish_setups', 0)}")
    lines.append(f"**Bearish setups:** {summary.get('bearish_setups', 0)}")
    lines.append(f"**High conviction:** {summary.get('high_conviction', 0)}\n")

    if deep_analyses:
        lines.append("## Top Candidates\n")
        for a in deep_analyses:
            sym = a.get("symbol", "?")
            score = a.get("composite_signal", {}).get("score", 0)
            rec = a.get("composite_signal", {}).get("recommendation", "?")
            price = a.get("latest_price", 0)
            lines.append(f"- **{sym}** @ ${price:.2f} | Score: {score:.1f} | {rec}")

    return "\n".join(lines)


def main():
    """Entry point for the morning briefing agent."""
    parser = argparse.ArgumentParser(description="Morning market briefing agent")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout, don't email")
    parser.add_argument("--no-ai", action="store_true", help="Skip Claude API, send raw data")
    args = parser.parse_args()

    config = AgentConfig.from_env()

    # Validate config (skip email validation for dry-run)
    if not args.dry_run:
        errors = config.validate()
        if args.no_ai:
            # Only need email config, not API key
            errors = [e for e in errors if "ANTHROPIC" not in e]
        if errors:
            for error in errors:
                logger.error("Config error: %s", error)
            sys.exit(1)
    elif not args.no_ai and not config.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY required (even for --dry-run unless --no-ai)")
        sys.exit(1)

    try:
        asyncio.run(run_morning_briefing(config, dry_run=args.dry_run, no_ai=args.no_ai))
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
