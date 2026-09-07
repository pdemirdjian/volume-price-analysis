"""Container liveness probe for the morning scheduler.

The scheduler loop writes a heartbeat file every time it wakes (bounded by
its sleep chunk) and around each briefing run. The Docker ``HEALTHCHECK``
runs :func:`main`, which fails once that file is missing or stale — so a
deadlocked loop, or an inner task that died while PID 1 lingered, is reported
instead of masked.

The file holds a ``time.monotonic()`` reading rather than relying on its
mtime. The scheduler's sleeps and Docker's probe interval both run on the
monotonic clock, so a wall-clock step (NTP correction, DST, a bad RTC) can
neither hide a hang nor flag a healthy loop. Monotonic readings are only
comparable within one boot; a stamp from the future therefore reads as
unhealthy rather than fresh.

Kept dependency-free (stdlib only) so the probe starts well inside the
healthcheck timeout; the scheduler imports the helpers from here, not the
other way round.
"""

import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

HEARTBEAT_ENV_VAR = "SCHEDULER_HEARTBEAT_FILE"
HEARTBEAT_FILENAME = "morning-scheduler.heartbeat"

# Longest a healthy scheduler may go without writing the heartbeat. The loop
# wakes at least every 15 minutes (scheduler._MAX_SLEEP_CHUNK_SECONDS) and a
# briefing run normally finishes well inside that; 45 minutes leaves room for
# a slow scan without hiding a real hang for long. tests/test_healthcheck.py
# pins the relationship to the scheduler's chunk.
HEARTBEAT_MAX_AGE_SECONDS = 45 * 60

# The stamp is written with millisecond precision, so a probe that runs in the
# same instant can read a hair "ahead"; only a clearly future stamp is rejected.
_FUTURE_TOLERANCE_SECONDS = 1.0


def default_heartbeat_path() -> Path:
    """The platform temp dir (``/tmp`` in the container) plus a fixed name."""
    return Path(tempfile.gettempdir()) / HEARTBEAT_FILENAME


def heartbeat_path() -> Path:
    """Resolve the heartbeat file location (env override or default)."""
    return Path(os.environ.get(HEARTBEAT_ENV_VAR) or default_heartbeat_path())


def write_heartbeat(path: Path, *, clock: Callable[[], float] = time.monotonic) -> None:
    """Record liveness by writing the current monotonic reading atomically.

    Never raises: an unwritable path degrades the healthcheck, not the
    scheduler. The failure is logged without a traceback because it repeats
    on every wake.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(f"{clock():.3f}\n", encoding="ascii")
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("Could not write heartbeat file %s: %s", path, exc)


def check_heartbeat(
    path: Path, max_age_seconds: float = HEARTBEAT_MAX_AGE_SECONDS, now: float | None = None
) -> str | None:
    """Return None when the heartbeat is fresh, else a human-readable reason.

    ``now`` is a monotonic reading (defaults to ``time.monotonic()``).
    """
    try:
        stamp = float(path.read_text(encoding="ascii").strip())
    except FileNotFoundError:
        return f"heartbeat file {path} does not exist"
    except OSError as exc:
        return f"cannot read heartbeat file {path}: {exc}"
    except ValueError:
        return f"heartbeat file {path} does not hold a monotonic timestamp"
    age = (time.monotonic() if now is None else now) - stamp
    if age < -_FUTURE_TOLERANCE_SECONDS:
        return f"heartbeat is {-age:.0f}s in the future (written before a reboot?)"
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
