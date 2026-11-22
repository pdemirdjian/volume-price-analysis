"""Unit tests for NYSE session calendar and daily-bar completeness (PDE-209)."""

from datetime import UTC, date, datetime, time

import pytest

from volume_price_analysis.sessions import (
    HALF_DAY_CLOSE,
    MARKET_TZ,
    REGULAR_CLOSE,
    is_bar_complete,
    is_trading_day,
    previous_trading_day,
    session_close,
    session_state,
)


def _et(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=MARKET_TZ)


class TestSessionCalendar:
    def test_regular_weekday_is_trading_day_with_1600_close(self):
        # Wednesday 2026-11-18
        day = date(2026, 11, 18)
        assert is_trading_day(day) is True
        assert session_close(day) == REGULAR_CLOSE

    def test_weekend_is_closed(self):
        assert is_trading_day(date(2026, 11, 21)) is False
        assert is_trading_day(date(2026, 11, 22)) is False
        assert session_close(date(2026, 11, 21)) is None

    @pytest.mark.parametrize(
        "holiday",
        [
            date(2026, 1, 1),  # New Year's Day
            date(2026, 4, 3),  # Good Friday
            date(2026, 7, 3),  # Independence Day (observed)
            date(2026, 11, 26),  # Thanksgiving Day
            date(2026, 12, 25),  # Christmas Day
        ],
    )
    def test_nyse_public_holidays_are_closed(self, holiday: date):
        assert is_trading_day(holiday) is False
        assert session_close(holiday) is None

    @pytest.mark.parametrize(
        "half_day",
        [
            date(2026, 11, 27),  # Day after Thanksgiving
            date(2026, 12, 24),  # Christmas Eve
        ],
    )
    def test_nyse_half_days_close_at_1300(self, half_day: date):
        assert is_trading_day(half_day) is True
        assert session_close(half_day) == HALF_DAY_CLOSE
        assert session_close(half_day) == time(13, 0)

    def test_previous_trading_day_skips_weekends_and_holidays(self):
        # Monday after Good Friday (2026-04-03) -> Thursday 2026-04-02
        assert previous_trading_day(date(2026, 4, 6)) == date(2026, 4, 2)
        # Day after Thanksgiving (2026-11-27) -> Wednesday 2026-11-25
        assert previous_trading_day(date(2026, 11, 27)) == date(2026, 11, 25)
        # Sunday 2026-11-22 -> Friday 2026-11-20
        assert previous_trading_day(date(2026, 11, 22)) == date(2026, 11, 20)


class TestSessionState:
    @pytest.mark.parametrize(
        ("hour", "minute", "second"),
        [
            (8, 30, 0),  # pre-market
            (11, 0, 0),  # intraday
            (16, 0, 0),  # right at close, within settle buffer
            (16, 14, 0),  # 1 minute before settle
            (16, 14, 59),  # 1 second before settle
        ],
    )
    def test_regular_day_before_1615_is_incomplete(self, hour: int, minute: int, second: int):
        now = _et(2026, 11, 18, hour, minute, second)
        state = session_state(now)

        assert state.today == date(2026, 11, 18)
        assert state.session_complete is False
        assert state.last_complete_session == date(2026, 11, 17)
        assert state.partial_bar_cutoff == date(2026, 11, 18)
        assert is_bar_complete(date(2026, 11, 18), now) is False
        assert is_bar_complete(date(2026, 11, 17), now) is True

    @pytest.mark.parametrize(("hour", "minute"), [(16, 15), (17, 0), (23, 59)])
    def test_regular_day_at_or_after_1615_is_complete(self, hour: int, minute: int):
        now = _et(2026, 11, 18, hour, minute)
        state = session_state(now)

        assert state.today == date(2026, 11, 18)
        assert state.session_complete is True
        assert state.last_complete_session == date(2026, 11, 18)
        assert state.partial_bar_cutoff is None
        assert is_bar_complete(date(2026, 11, 18), now) is True

    def test_half_day_settle_boundary_at_1315_et(self):
        # Friday 2026-11-27 (Day after Thanksgiving) — previous trading day is Wed 2026-11-25
        before = _et(2026, 11, 27, 13, 14, 59)
        state_before = session_state(before)
        assert state_before.session_complete is False
        assert state_before.last_complete_session == date(2026, 11, 25)
        assert state_before.partial_bar_cutoff == date(2026, 11, 27)
        assert is_bar_complete(date(2026, 11, 27), before) is False
        assert is_bar_complete(date(2026, 11, 25), before) is True

        at_settle = _et(2026, 11, 27, 13, 15, 0)
        state_after = session_state(at_settle)
        assert state_after.session_complete is True
        assert state_after.last_complete_session == date(2026, 11, 27)
        assert state_after.partial_bar_cutoff is None
        assert is_bar_complete(date(2026, 11, 27), at_settle) is True

    def test_christmas_eve_half_day(self):
        assert session_state(_et(2026, 12, 24, 13, 14)).session_complete is False
        assert session_state(_et(2026, 12, 24, 13, 14)).last_complete_session == date(2026, 12, 23)
        assert session_state(_et(2026, 12, 24, 13, 15)).session_complete is True
        assert session_state(_et(2026, 12, 24, 13, 15)).last_complete_session == date(2026, 12, 24)

    @pytest.mark.parametrize(
        ("now", "expected_session"),
        [
            (_et(2026, 11, 21, 10, 0), date(2026, 11, 20)),  # Saturday -> Friday
            (_et(2026, 11, 22, 18, 0), date(2026, 11, 20)),  # Sunday -> Friday
            (_et(2026, 11, 26, 12, 0), date(2026, 11, 25)),  # Thanksgiving -> Wednesday
            (_et(2026, 4, 3, 15, 0), date(2026, 4, 2)),  # Good Friday -> Thursday
        ],
    )
    def test_weekend_and_holiday_resolve_to_previous_session(
        self, now: datetime, expected_session: date
    ):
        state = session_state(now)
        assert state.session_complete is True
        assert state.last_complete_session == expected_session
        assert state.partial_bar_cutoff is None
        assert is_bar_complete(expected_session, now) is True

    def test_utc_inputs_convert_to_eastern_across_dst(self):
        # EST (UTC-5): 21:15 UTC == 16:15 EST
        assert session_state(datetime(2026, 11, 18, 21, 14, tzinfo=UTC)).session_complete is False
        assert session_state(datetime(2026, 11, 18, 21, 15, tzinfo=UTC)).session_complete is True
        # EDT (UTC-4): 20:15 UTC == 16:15 EDT
        assert session_state(datetime(2026, 7, 8, 20, 14, tzinfo=UTC)).session_complete is False
        assert session_state(datetime(2026, 7, 8, 20, 15, tzinfo=UTC)).session_complete is True

    def test_future_bar_date_is_incomplete(self):
        now = _et(2026, 11, 18, 17, 0)
        assert is_bar_complete(date(2026, 11, 19), now) is False

    def test_naive_datetime_rejected(self):
        with pytest.raises(ValueError, match="aware datetime"):
            session_state(datetime(2026, 11, 18, 12, 0))
