"""Asyncio-based scheduler for the morning briefing agent.

Replaces supercronic with a pure-Python scheduler that sleeps until
the next scheduled time and invokes run_morning_briefing().

Usage:
    morning-scheduler [--time HH:MM] [--skip-holidays]
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import holidays

from .config import AgentConfig
from .email_sender import send_error_email
from .morning_agent import run_morning_briefing

# Configure logging to stdout (Docker best practice)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# Never sleep longer than this in one stretch: asyncio timeouts run on the
# monotonic clock, which drifts from wall time across DST transitions, NTP
# steps, and container suspend/resume. Waking regularly and re-checking wall
# time bounds any such error to one chunk.
_MAX_SLEEP_CHUNK_SECONDS = 15 * 60

# NYSE holidays instance (reused across calls)
_nyse_holidays = holidays.financial_holidays("NYSE")


def _next_run(
    target: time,
    tz: ZoneInfo,
    now: datetime | None = None,
    skip_holidays: bool = False,
) -> datetime:
    """Compute the next scheduled run datetime.

    Pure function — accepts ``now`` for easy testing.

    Args:
        target: Time of day to run (e.g. 08:30).
        tz: Timezone for scheduling.
        now: Current datetime (defaults to now in *tz*).
        skip_holidays: If True, advance past NYSE market holidays.

    Returns:
        The next datetime at which the briefing should fire.
    """
    if now is None:
        now = datetime.now(tz)

    candidate = datetime.combine(now.date(), target, tz)

    # If we're already at or past the target time today, move to tomorrow
    if now >= candidate:
        candidate += timedelta(days=1)

    # Advance past weekends (Sat=5, Sun=6)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)

    # Advance past holidays if requested
    if skip_holidays:
        while candidate.date() in _nyse_holidays:
            candidate += timedelta(days=1)
            # Might land on a weekend after skipping a holiday
            while candidate.weekday() >= 5:
                candidate += timedelta(days=1)

    return candidate


async def _wait_for_next_run(
    next_dt: datetime,
    target: time,
    tz: ZoneInfo,
    stop_event: asyncio.Event,
    skip_holidays: bool = False,
    last_fired: datetime | None = None,
) -> datetime | None:
    """Sleep until *next_dt* in bounded chunks, re-deriving the schedule each wake.

    Returns the schedule datetime that fired, or None if *stop_event* was set
    first. *last_fired* is the most recent schedule that already ran; it floors
    re-derivation so a backward clock jump can never re-fire it.
    """
    while True:
        if stop_event.is_set():
            return None
        now = datetime.now(tz)
        if now >= next_dt:
            return next_dt
        delay = (next_dt - now).total_seconds()

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=min(delay, _MAX_SLEEP_CHUNK_SECONDS))
            return None  # stop_event was set
        except TimeoutError:
            if stop_event.is_set():
                return None
            now = datetime.now(tz)
            # Check against the original schedule first: if the clock jumped
            # past the target while we slept, fire now rather than re-deriving
            # (which would push the run to the next day).
            if now >= next_dt:
                return next_dt
            # Re-derive from no earlier than the last fired schedule, so a
            # backward clock jump cannot repeat a briefing that already ran.
            basis = now if last_fired is None else max(now, last_fired)
            rederived = _next_run(target, tz, now=basis, skip_holidays=skip_holidays)
            if rederived != next_dt:
                logger.info(
                    "Schedule shifted (clock or DST change); next briefing now %s",
                    rederived.strftime("%Y-%m-%d %H:%M"),
                )
                next_dt = rederived


async def _run_loop(
    target: time,
    tz: ZoneInfo,
    stop_event: asyncio.Event,
    skip_holidays: bool = False,
) -> None:
    """Core scheduling loop: compute next run, sleep, execute, repeat."""
    config = AgentConfig.from_env()
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error("Config error: %s", error)
        raise SystemExit(1)

    last_fired: datetime | None = None
    while not stop_event.is_set():
        # Compute from no earlier than the last fired schedule (see
        # _wait_for_next_run) so a backward clock jump can't repeat a run.
        basis = datetime.now(tz) if last_fired is None else max(datetime.now(tz), last_fired)
        next_dt = _next_run(target, tz, now=basis, skip_holidays=skip_holidays)

        # Log next run (with holiday info if applicable)
        holiday_name = _nyse_holidays.get(next_dt.date())
        if holiday_name:
            if skip_holidays:
                # This shouldn't happen since _next_run skips holidays,
                # but log defensively
                logger.info(
                    "Next briefing scheduled for %s (skipping: %s)",
                    next_dt.strftime("%Y-%m-%d %H:%M"),
                    holiday_name,
                )
            else:
                logger.info(
                    "Next briefing scheduled for %s (holiday: %s — still running)",
                    next_dt.strftime("%Y-%m-%d %H:%M"),
                    holiday_name,
                )
        else:
            logger.info(
                "Next briefing scheduled for %s",
                next_dt.strftime("%Y-%m-%d %H:%M"),
            )

        # Sleep until next run or stop signal
        fired = await _wait_for_next_run(
            next_dt, target, tz, stop_event, skip_holidays=skip_holidays, last_fired=last_fired
        )
        if fired is None:
            break
        last_fired = fired

        # Re-load config so rotated credentials/keys are picked up without a
        # container restart; keep the last valid config if the new one is bad.
        fresh_config = AgentConfig.from_env()
        fresh_errors = fresh_config.validate()
        if fresh_errors:
            for error in fresh_errors:
                logger.error("Config error on reload: %s — keeping previous config", error)
        else:
            config = fresh_config

        # Execute briefing
        try:
            logger.info("Running morning briefing...")
            result = await run_morning_briefing(config)
            if result:
                logger.info("Morning briefing completed successfully")
            else:
                logger.warning("Morning briefing completed in degraded mode (AI fallback)")
        except Exception as e:
            logger.exception("Morning briefing failed")
            # Try to send error notification (mirrors morning_agent.py behavior)
            if config.email_from and config.email_password and config.email_to:
                try:
                    send_error_email(
                        error_message=str(e),
                        from_addr=config.email_from,
                        password=config.email_password,
                        to_addr=config.email_to,
                        smtp_host=config.email_smtp_host,
                        smtp_port=config.email_smtp_port,
                    )
                except Exception:
                    logger.exception("Failed to send error email")


async def run_scheduler(
    target: time,
    tz: ZoneInfo,
    skip_holidays: bool = False,
) -> None:
    """Set up signal handlers and run the scheduling loop."""
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    if sys.platform == "win32":

        def _signal_handler_win(signum: int, frame: object) -> None:
            logger.info("Received shutdown signal, stopping...")
            loop.call_soon_threadsafe(stop_event.set)

        signal.signal(signal.SIGINT, _signal_handler_win)
    else:

        def _signal_handler() -> None:
            logger.info("Received shutdown signal, stopping...")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _signal_handler)

    logger.info(
        "Scheduler starting (target=%s, tz=%s, skip_holidays=%s)",
        target.strftime("%H:%M"),
        tz,
        skip_holidays,
    )

    await _run_loop(target, tz, stop_event, skip_holidays=skip_holidays)

    logger.info("Scheduler stopped")


def main() -> None:
    """Entry point for the morning-scheduler CLI."""
    parser = argparse.ArgumentParser(description="Morning briefing scheduler")
    parser.add_argument(
        "--time",
        default="08:30",
        help="Schedule time in HH:MM format (default: 08:30)",
    )
    parser.add_argument(
        "--skip-holidays",
        action="store_true",
        help="Skip NYSE market holidays (default: run every weekday)",
    )
    args = parser.parse_args()

    # Parse time
    try:
        parts = args.time.split(":")
        if len(parts) != 2:
            raise ValueError("Time must be in HH:MM format")
        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour <= 23):
            raise ValueError(f"Hour must be between 0 and 23 (got {hour})")
        if not (0 <= minute <= 59):
            raise ValueError(f"Minute must be between 0 and 59 (got {minute})")
        target = time(hour, minute)
    except ValueError as exc:
        logger.error("Invalid --time value %r: %s", args.time, exc)
        sys.exit(1)

    asyncio.run(run_scheduler(target, ET, skip_holidays=args.skip_holidays))


if __name__ == "__main__":
    main()
