"""Small bounded retry policy shared by core and agent operations."""

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)
_sleep = time.sleep

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 2


def retry_call[T](
    fn: Callable[[], T],
    *,
    attempts: int,
    base_delay: float,
    retry_on: Callable[[Exception], bool],
    sleep: Callable[[float], None] | None = None,
) -> T:
    """Retry eligible failures; re-raise the last exception unchanged."""
    sleep = sleep if sleep is not None else _sleep
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts or not retry_on(exc):
                raise
            logger.warning(
                "Retrying after attempt %d/%d failed (%s)",
                attempt,
                attempts,
                type(exc).__name__,
            )
            sleep(base_delay * 2 ** (attempt - 1))
    raise ValueError("attempts must be at least 1")
