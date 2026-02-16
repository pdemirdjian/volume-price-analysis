"""Tests for the asyncio morning briefing scheduler."""

import asyncio
import logging
from datetime import datetime, time, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from volume_price_analysis.agent.scheduler import (
    _next_run,
    _run_loop,
    run_scheduler,
)

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# _next_run() unit tests — pure function, no mocking needed
# ---------------------------------------------------------------------------


class TestNextRunWeekday:
    """Test scheduling on regular weekdays."""

    def test_before_target_same_day(self):
        """Before target time on a weekday → same day."""
        now = datetime(2026, 2, 16, 7, 0, tzinfo=ET)  # Monday 07:00
        result = _next_run(time(8, 30), ET, now=now)
        assert result == datetime(2026, 2, 16, 8, 30, tzinfo=ET)

    def test_after_target_next_weekday(self):
        """After target time on a weekday → next weekday."""
        now = datetime(2026, 2, 16, 9, 0, tzinfo=ET)  # Monday 09:00
        result = _next_run(time(8, 30), ET, now=now)
        assert result == datetime(2026, 2, 17, 8, 30, tzinfo=ET)  # Tuesday

    def test_at_exact_target_time(self):
        """At exact target time → next weekday."""
        now = datetime(2026, 2, 16, 8, 30, tzinfo=ET)  # Monday 08:30
        result = _next_run(time(8, 30), ET, now=now)
        assert result == datetime(2026, 2, 17, 8, 30, tzinfo=ET)  # Tuesday


class TestNextRunWeekend:
    """Test scheduling across weekends."""

    def test_friday_after_target_to_monday(self):
        """Friday after target → Monday."""
        now = datetime(2026, 2, 20, 9, 0, tzinfo=ET)  # Friday 09:00
        result = _next_run(time(8, 30), ET, now=now)
        assert result == datetime(2026, 2, 23, 8, 30, tzinfo=ET)  # Monday

    def test_saturday_to_monday(self):
        """Saturday → Monday."""
        now = datetime(2026, 2, 21, 10, 0, tzinfo=ET)  # Saturday
        result = _next_run(time(8, 30), ET, now=now)
        assert result == datetime(2026, 2, 23, 8, 30, tzinfo=ET)  # Monday

    def test_sunday_to_monday(self):
        """Sunday → Monday."""
        now = datetime(2026, 2, 22, 10, 0, tzinfo=ET)  # Sunday
        result = _next_run(time(8, 30), ET, now=now)
        assert result == datetime(2026, 2, 23, 8, 30, tzinfo=ET)  # Monday


class TestNextRunHolidays:
    """Test holiday handling."""

    def test_holiday_without_skip_returns_holiday(self):
        """Holiday without skip → runs on the holiday."""
        # MLK Day 2026 is Monday Jan 19
        now = datetime(2026, 1, 18, 10, 0, tzinfo=ET)  # Sunday before MLK
        result = _next_run(time(8, 30), ET, now=now, skip_holidays=False)
        assert result == datetime(2026, 1, 19, 8, 30, tzinfo=ET)  # MLK Day

    def test_holiday_with_skip_advances(self):
        """Holiday with skip → next non-holiday weekday."""
        # MLK Day 2026 is Monday Jan 19
        now = datetime(2026, 1, 18, 10, 0, tzinfo=ET)  # Sunday before MLK
        result = _next_run(time(8, 30), ET, now=now, skip_holidays=True)
        assert result == datetime(2026, 1, 20, 8, 30, tzinfo=ET)  # Tuesday

    def test_thanksgiving_friday_skip(self):
        """Thanksgiving (Thursday) + Black Friday check — skip advances past holiday."""
        # Thanksgiving 2026 is Thursday Nov 26
        now = datetime(2026, 11, 25, 9, 0, tzinfo=ET)  # Wednesday after target
        result = _next_run(time(8, 30), ET, now=now, skip_holidays=True)
        # Nov 26 is Thanksgiving — skip to Nov 27 (Friday, which is a half-day
        # but not an NYSE holiday in the `holidays` package)
        assert result.date() >= datetime(2026, 11, 27).date()
        assert result.weekday() < 5  # Must be a weekday

    def test_non_holiday_weekday_unaffected_by_skip(self):
        """A regular weekday is unaffected by skip_holidays."""
        now = datetime(2026, 3, 2, 7, 0, tzinfo=ET)  # Monday, no holiday
        result_skip = _next_run(time(8, 30), ET, now=now, skip_holidays=True)
        result_no_skip = _next_run(time(8, 30), ET, now=now, skip_holidays=False)
        assert result_skip == result_no_skip


class TestNextRunEdgeCases:
    """Edge cases for _next_run()."""

    def test_custom_time(self):
        """Works with a non-default time."""
        now = datetime(2026, 2, 16, 7, 0, tzinfo=ET)  # Monday 07:00
        result = _next_run(time(9, 0), ET, now=now)
        assert result == datetime(2026, 2, 16, 9, 0, tzinfo=ET)

    def test_defaults_to_now(self):
        """When now is None, uses current time."""
        result = _next_run(time(8, 30), ET)
        assert result.tzinfo is not None
        assert result > datetime.now(ET) - timedelta(seconds=1)


# ---------------------------------------------------------------------------
# Integration tests — async loop behavior
# ---------------------------------------------------------------------------


class TestRunLoop:
    """Test the scheduling loop behavior."""

    @pytest.mark.asyncio
    async def test_stop_event_exits_loop(self):
        """The loop exits when the stop event is set."""
        stop_event = asyncio.Event()
        stop_event.set()  # Pre-set so loop exits immediately

        with patch(
            "volume_price_analysis.agent.scheduler.AgentConfig.from_env"
        ) as mock_config:
            config = MagicMock()
            config.validate.return_value = []
            mock_config.return_value = config

            await _run_loop(time(8, 30), ET, stop_event)

    @pytest.mark.asyncio
    async def test_config_validation_failure_exits_nonzero(self, caplog):
        """Loop exits with non-zero status if config validation fails."""
        stop_event = asyncio.Event()

        with patch(
            "volume_price_analysis.agent.scheduler.AgentConfig.from_env"
        ) as mock_config:
            config = MagicMock()
            config.validate.return_value = ["EMAIL_FROM is required"]
            mock_config.return_value = config

            with pytest.raises(SystemExit, match="1"):
                await _run_loop(time(8, 30), ET, stop_event)

        assert "Config error" in caplog.text

    @pytest.mark.asyncio
    async def test_briefing_failure_continues_loop(self):
        """Loop continues after a briefing failure (doesn't crash out)."""
        call_count = 0

        async def _fake_briefing(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Briefing failed")

        stop_event = asyncio.Event()

        with (
            patch(
                "volume_price_analysis.agent.scheduler.AgentConfig.from_env"
            ) as mock_config,
            patch(
                "volume_price_analysis.agent.scheduler.run_morning_briefing",
                side_effect=_fake_briefing,
            ),
            patch(
                "volume_price_analysis.agent.scheduler.send_error_email"
            ) as mock_error_email,
            patch(
                "volume_price_analysis.agent.scheduler._next_run"
            ) as mock_next_run,
        ):
            config = MagicMock()
            config.validate.return_value = []
            config.email_from = "a@b.com"
            config.email_password = "pass"
            config.email_to = "c@d.com"
            config.email_smtp_host = "smtp.gmail.com"
            config.email_smtp_port = 587
            mock_config.return_value = config

            # Return a time slightly in the future so the sleep fires quickly
            now = datetime.now(ET)

            def _next_run_and_stop(*args, **kwargs):
                # After first briefing call, set stop so loop exits on next iteration
                if call_count >= 1:
                    stop_event.set()
                return now + timedelta(milliseconds=10)

            mock_next_run.side_effect = _next_run_and_stop

            await _run_loop(time(8, 30), ET, stop_event)

        # Briefing was called at least once and the loop didn't crash
        assert call_count >= 1
        mock_error_email.assert_called()


class TestRunScheduler:
    """Test top-level scheduler with signal handling."""

    @pytest.mark.asyncio
    async def test_logs_startup_message(self, caplog):
        """Scheduler logs startup info."""
        with caplog.at_level(logging.INFO, logger="volume_price_analysis.agent.scheduler"):
            with (
                patch(
                    "volume_price_analysis.agent.scheduler._run_loop",
                    new_callable=AsyncMock,
                ),
            ):
                await run_scheduler(time(8, 30), ET)

        assert "Scheduler starting" in caplog.text
        assert "08:30" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_shutdown_message(self, caplog):
        """Scheduler logs shutdown message."""
        with caplog.at_level(logging.INFO, logger="volume_price_analysis.agent.scheduler"):
            with (
                patch(
                    "volume_price_analysis.agent.scheduler._run_loop",
                    new_callable=AsyncMock,
                ),
            ):
                await run_scheduler(time(8, 30), ET)

        assert "Scheduler stopped" in caplog.text
