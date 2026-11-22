"""NYSE session calendar and daily-bar completeness helpers (PDE-209).

Pure functions of an instant plus the NYSE holiday/half-day calendar from
``holidays`` — no network access. Used by the data-fetch cache (so intraday
entries expire at session settle) and by the MCP tools (to drop in-progress
daily bars by default).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import holidays

MARKET_TZ = ZoneInfo("America/New_York")
REGULAR_CLOSE = time(16, 0)
HALF_DAY_CLOSE = time(13, 0)
SETTLE_BUFFER = timedelta(minutes=15)


@functools.lru_cache(maxsize=64)
def _nyse_public_holidays(year: int) -> frozenset[date]:
    """Immutable set of NYSE full-day market holidays for ``year``."""
    return frozenset(holidays.financial_holidays("NYSE", years=year, expand=False).keys())


@functools.lru_cache(maxsize=64)
def _nyse_half_days(year: int) -> frozenset[date]:
    """Immutable set of NYSE early-close (13:00 ET) sessions for ``year``."""
    return frozenset(
        holidays.financial_holidays(
            "NYSE", years=year, categories=("half_day",), expand=False
        ).keys()
    )


def is_trading_day(day: date) -> bool:
    """True when the NYSE holds a regular or half-day session on ``day``."""
    return day.weekday() < 5 and day not in _nyse_public_holidays(day.year)


def session_close(day: date) -> time | None:
    """Scheduled NYSE closing time (ET) on ``day``, or None when the market is closed."""
    if not is_trading_day(day):
        return None
    return HALF_DAY_CLOSE if day in _nyse_half_days(day.year) else REGULAR_CLOSE


def previous_trading_day(day: date) -> date:
    """Most recent NYSE trading session strictly before ``day``."""
    candidate = day - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def _normalize_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(MARKET_TZ)
    if now.tzinfo is None:
        raise ValueError("now must be an aware datetime")
    return now.astimezone(MARKET_TZ)


@dataclass(frozen=True, slots=True)
class SessionState:
    """NYSE daily-bar completeness at a single instant.

    ``today`` is the Eastern calendar date of the instant.
    ``last_complete_session`` is the most recent NYSE session whose daily bar
    has settled (regular 16:00 ET or half-day 13:00 ET plus the 15-minute
    settle buffer). ``session_complete`` is True when no in-progress daily bar
    exists at this instant (after today's settle, or on weekends/holidays).
    """

    today: date
    last_complete_session: date
    session_complete: bool

    @property
    def partial_bar_cutoff(self) -> date | None:
        """``as_of`` cutoff dropping today's in-progress bar, or None when settled."""
        return None if self.session_complete else self.today

    def is_bar_complete(self, bar_date: date) -> bool:
        """True when a daily bar dated ``bar_date`` is settled at this instant."""
        if bar_date < self.today:
            return True
        if bar_date == self.today:
            return self.session_complete
        return False


def session_state(now: datetime | None = None) -> SessionState:
    """Return NYSE session completeness and last complete session for ``now``."""
    now_et = _normalize_now(now)
    today = now_et.date()
    close_time = session_close(today)
    if close_time is None:
        return SessionState(
            today=today,
            last_complete_session=previous_trading_day(today),
            session_complete=True,
        )
    settle_at = datetime.combine(today, close_time, tzinfo=MARKET_TZ) + SETTLE_BUFFER
    if now_et >= settle_at:
        return SessionState(
            today=today,
            last_complete_session=today,
            session_complete=True,
        )
    return SessionState(
        today=today,
        last_complete_session=previous_trading_day(today),
        session_complete=False,
    )


def is_bar_complete(bar_date: date, now: datetime | None = None) -> bool:
    """True when a daily bar dated ``bar_date`` is complete as of ``now``.

    A bar counts as complete if its date is before today (ET), or if ``now``
    (ET) is at or past the session close + 15-minute settle buffer.
    """
    return session_state(now).is_bar_complete(bar_date)
