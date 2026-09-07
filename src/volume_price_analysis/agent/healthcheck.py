"""Container liveness probe for the morning scheduler.

The scheduler loop touches a heartbeat file every time it wakes (bounded by
its sleep chunk) and around each briefing run. The Docker ``HEALTHCHECK``
runs :func:`main`, which fails once that file is missing or stale — so a
deadlocked loop, or an inner task that died while PID 1 lingered, is reported
instead of masked.

Kept dependency-free (stdlib only) so the probe starts well inside the
healthcheck timeout; the scheduler imports the helpers from here, not the
other way round.
"""

import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

HEARTBEAT_ENV_VAR = "SCHEDULER_HEARTBEAT_FILE"
DEFAULT_HEARTBEAT_PATH = Path("/tmp/morning-scheduler.heartbeat")

# Longest a healthy scheduler may go without touching the heartbeat. The loop
# wakes at least every 15 minutes (scheduler._MAX_SLEEP_CHUNK_SECONDS) and a
# briefing run normally finishes well inside that; 45 minutes leaves room for
# a slow scan without hiding a real hang for long. tests/test_healthcheck.py
# pins the relationship to the scheduler's chunk.
HEARTBEAT_MAX_AGE_SECONDS = 45 * 60


def heartbeat_path() -> Path:
    """Resolve the heartbeat file location (env override or default)."""
    return Path(os.environ.get(HEARTBEAT_ENV_VAR) or DEFAULT_HEARTBEAT_PATH)


def touch_heartbeat(path: Path) -> None:
    """Record liveness by updating the heartbeat file's mtime.

    Never raises: an unwritable path degrades the healthcheck, not the
    scheduler.
    """
    try:
        path.touch()
    except OSError:
        logger.warning("Could not update heartbeat file %s", path, exc_info=True)


def check_heartbeat(
    path: Path, max_age_seconds: float = HEARTBEAT_MAX_AGE_SECONDS, now: float | None = None
) -> str | None:
    """Return None when the heartbeat is fresh, else a human-readable reason."""
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return f"heartbeat file {path} does not exist"
    except OSError as exc:
        return f"cannot read heartbeat file {path}: {exc}"
    age = (time.time() if now is None else now) - mtime
    if age > max_age_seconds:
        return f"heartbeat is {age:.0f}s old (max {max_age_seconds:.0f}s)"
    return None


def main() -> None:
    """Entry point for the morning-healthcheck CLI: exit 0 healthy, 1 otherwise."""
    reason = check_heartbeat(heartbeat_path())
    if reason is not None:
        print(f"unhealthy: {reason}", file=sys.stderr)
        sys.exit(1)
    print("healthy")


if __name__ == "__main__":
    main()
