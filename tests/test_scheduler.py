"""Tests for the asyncio morning briefing scheduler."""

import asyncio
import logging
import signal
from datetime import datetime, time, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from volume_price_analysis.agent.scheduler import (
    _next_run,
    _run_loop,
    _wait_for_next_run,
    main,
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

    def test_skips_holiday_that_lands_on_weekend(self):
        """A Friday NYSE holiday advances past the weekend to Monday."""
        # Good Friday 2025-04-18 is a Friday NYSE holiday.
        # Thursday after target → Friday is a holiday → Saturday → Monday.
        now = datetime(2025, 4, 17, 9, 0, tzinfo=ET)
        result = _next_run(time(8, 30), ET, now=now, skip_holidays=True)
        assert result.date() == datetime(2025, 4, 21, tzinfo=ET).date()
        assert result.weekday() == 0  # Monday

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

    def test_spring_forward_transition(self):
        """Across the spring-forward weekend the target stays 08:30 wall clock."""
        # DST starts Sunday 2026-03-08; Friday is EST, Monday is EDT
        now = datetime(2026, 3, 6, 9, 0, tzinfo=ET)  # Friday 09:00 EST
        result = _next_run(time(8, 30), ET, now=now)
        assert result == datetime(2026, 3, 9, 8, 30, tzinfo=ET)  # Monday
        assert now.utcoffset() == timedelta(hours=-5)
        assert result.utcoffset() == timedelta(hours=-4)

    def test_fall_back_transition(self):
        """Across the fall-back weekend the target stays 08:30 wall clock."""
        # DST ends Sunday 2026-11-01; Friday is EDT, Monday is EST
        now = datetime(2026, 10, 30, 9, 0, tzinfo=ET)  # Friday 09:00 EDT
        result = _next_run(time(8, 30), ET, now=now)
        assert result == datetime(2026, 11, 2, 8, 30, tzinfo=ET)  # Monday
        assert now.utcoffset() == timedelta(hours=-4)
        assert result.utcoffset() == timedelta(hours=-5)


# ---------------------------------------------------------------------------
# Integration tests — async loop behavior
# ---------------------------------------------------------------------------


class TestWaitForNextRun:
    """Test the chunked sleep helper."""

    @pytest.mark.asyncio
    async def test_fires_when_time_already_passed(self):
        stop_event = asyncio.Event()
        next_dt = datetime.now(ET) - timedelta(seconds=1)
        assert await _wait_for_next_run(next_dt, time(8, 30), ET, stop_event) == next_dt

    @pytest.mark.asyncio
    async def test_returns_none_when_stop_event_set(self):
        stop_event = asyncio.Event()
        stop_event.set()
        next_dt = datetime.now(ET) + timedelta(hours=1)
        assert await _wait_for_next_run(next_dt, time(8, 30), ET, stop_event) is None

    @pytest.mark.asyncio
    async def test_stop_wins_over_elapsed_deadline(self):
        """A stop request beats an already-elapsed deadline — no briefing on shutdown."""
        stop_event = asyncio.Event()
        stop_event.set()
        next_dt = datetime.now(ET) - timedelta(seconds=1)
        assert await _wait_for_next_run(next_dt, time(8, 30), ET, stop_event) is None

    @pytest.mark.asyncio
    async def test_rederives_each_wake_and_adopts_corrected_schedule(self):
        """The helper wakes in chunks and adopts a corrected (earlier) schedule.

        The stale deadline is an hour out, so finishing within the test timeout
        is only possible by waking from a chunk and adopting the re-derived
        earlier time — which also keeps the test immune to slow CI machines.
        """
        stop_event = asyncio.Event()
        next_dt = datetime.now(ET) + timedelta(hours=1)
        corrected = datetime.now(ET) - timedelta(seconds=1)
        with (
            patch("volume_price_analysis.agent.scheduler._MAX_SLEEP_CHUNK_SECONDS", 0.01),
            patch(
                "volume_price_analysis.agent.scheduler._next_run",
                return_value=corrected,
            ) as mock_next_run,
        ):
            result = await asyncio.wait_for(
                _wait_for_next_run(next_dt, time(8, 30), ET, stop_event), timeout=5
            )
        assert result == corrected
        assert mock_next_run.call_count >= 1

    @pytest.mark.asyncio
    async def test_rederives_no_earlier_than_last_fired(self):
        """Re-derivation is floored at the last fired schedule.

        Simulates a backward clock jump: last_fired sits ahead of the current
        wall clock, and _next_run must be asked for the schedule after
        last_fired — not after now — so the same briefing can't fire twice.
        """
        stop_event = asyncio.Event()
        next_dt = datetime.now(ET) + timedelta(hours=1)
        last_fired = datetime.now(ET) + timedelta(minutes=30)
        corrected = datetime.now(ET) - timedelta(seconds=1)
        with (
            patch("volume_price_analysis.agent.scheduler._MAX_SLEEP_CHUNK_SECONDS", 0.01),
            patch(
                "volume_price_analysis.agent.scheduler._next_run",
                return_value=corrected,
            ) as mock_next_run,
        ):
            await asyncio.wait_for(
                _wait_for_next_run(next_dt, time(8, 30), ET, stop_event, last_fired=last_fired),
                timeout=5,
            )
        assert mock_next_run.call_args.kwargs["now"] == last_fired


class TestRunLoop:
    """Test the scheduling loop behavior."""

    @pytest.mark.asyncio
    async def test_stop_event_exits_loop(self):
        """The loop exits when the stop event is set."""
        stop_event = asyncio.Event()
        stop_event.set()  # Pre-set so loop exits immediately

        with patch("volume_price_analysis.agent.scheduler.AgentConfig.from_env") as mock_config:
            config = MagicMock()
            config.validate.return_value = []
            mock_config.return_value = config

            await _run_loop(time(8, 30), ET, stop_event)

    @pytest.mark.asyncio
    async def test_stop_landing_as_schedule_fires_skips_briefing(self):
        """A stop signal arriving just as the schedule fires prevents the briefing."""
        stop_event = asyncio.Event()

        async def _fire_and_stop(next_dt, *args, **kwargs):
            stop_event.set()  # signal lands in the window after the fire
            return next_dt

        with (
            patch("volume_price_analysis.agent.scheduler.AgentConfig.from_env") as mock_config,
            patch(
                "volume_price_analysis.agent.scheduler._wait_for_next_run",
                side_effect=_fire_and_stop,
            ),
            patch(
                "volume_price_analysis.agent.scheduler.run_morning_briefing",
                new_callable=AsyncMock,
            ) as mock_briefing,
        ):
            config = MagicMock()
            config.validate.return_value = []
            mock_config.return_value = config

            await _run_loop(time(8, 30), ET, stop_event)

        mock_briefing.assert_not_called()

    @pytest.mark.asyncio
    async def test_config_validation_failure_exits_nonzero(self, caplog):
        """Loop exits with non-zero status if config validation fails."""
        stop_event = asyncio.Event()

        with patch("volume_price_analysis.agent.scheduler.AgentConfig.from_env") as mock_config:
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
            patch("volume_price_analysis.agent.scheduler.AgentConfig.from_env") as mock_config,
            patch(
                "volume_price_analysis.agent.scheduler.run_morning_briefing",
                side_effect=_fake_briefing,
            ),
            patch("volume_price_analysis.agent.scheduler.send_error_email") as mock_error_email,
            patch("volume_price_analysis.agent.scheduler._next_run") as mock_next_run,
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

    @pytest.mark.asyncio
    async def test_reloads_config_before_each_run(self):
        """The config is re-loaded from env before each briefing run."""
        stop_event = asyncio.Event()
        startup_config = MagicMock(name="startup")
        startup_config.validate.return_value = []
        reloaded_config = MagicMock(name="reloaded")
        reloaded_config.validate.return_value = []

        used_configs = []

        async def _fake_briefing(config, *args, **kwargs):
            used_configs.append(config)
            stop_event.set()
            return True

        with (
            patch(
                "volume_price_analysis.agent.scheduler.AgentConfig.from_env",
                side_effect=[startup_config, reloaded_config],
            ),
            patch(
                "volume_price_analysis.agent.scheduler.run_morning_briefing",
                side_effect=_fake_briefing,
            ),
            patch(
                "volume_price_analysis.agent.scheduler._next_run",
                return_value=datetime.now(ET) - timedelta(seconds=1),
            ),
        ):
            await _run_loop(time(8, 30), ET, stop_event)

        assert used_configs == [reloaded_config]

    @pytest.mark.asyncio
    async def test_invalid_reload_keeps_previous_config(self, caplog):
        """An invalid re-loaded config is rejected; the previous one keeps working."""
        stop_event = asyncio.Event()
        startup_config = MagicMock(name="startup")
        startup_config.validate.return_value = []
        bad_config = MagicMock(name="bad")
        bad_config.validate.return_value = ["EMAIL_FROM is required"]

        used_configs = []

        async def _fake_briefing(config, *args, **kwargs):
            used_configs.append(config)
            stop_event.set()
            return True

        with (
            caplog.at_level(logging.ERROR, logger="volume_price_analysis.agent.scheduler"),
            patch(
                "volume_price_analysis.agent.scheduler.AgentConfig.from_env",
                side_effect=[startup_config, bad_config],
            ),
            patch(
                "volume_price_analysis.agent.scheduler.run_morning_briefing",
                side_effect=_fake_briefing,
            ),
            patch(
                "volume_price_analysis.agent.scheduler._next_run",
                return_value=datetime.now(ET) - timedelta(seconds=1),
            ),
        ):
            await _run_loop(time(8, 30), ET, stop_event)

        assert used_configs == [startup_config]
        assert "keeping previous config" in caplog.text

    @pytest.mark.asyncio
    async def test_holiday_log_when_skip_holidays_true(self, caplog):
        """_run_loop logs a defensive skip message if the next run falls on a holiday."""
        stop_event = asyncio.Event()
        holiday_dt = datetime(2025, 4, 18, 8, 30, tzinfo=ET)  # Good Friday

        async def _fake_briefing(*args, **kwargs):
            stop_event.set()
            return True

        with (
            caplog.at_level(logging.INFO, logger="volume_price_analysis.agent.scheduler"),
            patch("volume_price_analysis.agent.scheduler.AgentConfig.from_env") as mock_config,
            patch(
                "volume_price_analysis.agent.scheduler._next_run",
                return_value=holiday_dt,
            ),
            patch(
                "volume_price_analysis.agent.scheduler.run_morning_briefing",
                side_effect=_fake_briefing,
            ),
        ):
            config = MagicMock()
            config.validate.return_value = []
            mock_config.return_value = config
            await _run_loop(time(8, 30), ET, stop_event, skip_holidays=True)

        assert "skipping" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_holiday_log_when_skip_holidays_false(self, caplog):
        """_run_loop logs that it still runs when the next date is a holiday."""
        stop_event = asyncio.Event()
        holiday_dt = datetime(2025, 4, 18, 8, 30, tzinfo=ET)  # Good Friday

        async def _fake_briefing(*args, **kwargs):
            stop_event.set()
            return True

        with (
            caplog.at_level(logging.INFO, logger="volume_price_analysis.agent.scheduler"),
            patch("volume_price_analysis.agent.scheduler.AgentConfig.from_env") as mock_config,
            patch(
                "volume_price_analysis.agent.scheduler._next_run",
                return_value=holiday_dt,
            ),
            patch(
                "volume_price_analysis.agent.scheduler.run_morning_briefing",
                side_effect=_fake_briefing,
            ),
        ):
            config = MagicMock()
            config.validate.return_value = []
            mock_config.return_value = config
            await _run_loop(time(8, 30), ET, stop_event, skip_holidays=False)

        assert "still running" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_successful_briefing_logs_completion(self, caplog):
        """_run_loop logs success when run_morning_briefing returns a truthy result."""
        stop_event = asyncio.Event()

        async def _fake_briefing(*args, **kwargs):
            stop_event.set()
            return True

        with (
            caplog.at_level(logging.INFO, logger="volume_price_analysis.agent.scheduler"),
            patch("volume_price_analysis.agent.scheduler.AgentConfig.from_env") as mock_config,
            patch(
                "volume_price_analysis.agent.scheduler.run_morning_briefing",
                side_effect=_fake_briefing,
            ),
            patch(
                "volume_price_analysis.agent.scheduler._next_run",
                return_value=datetime.now(ET) - timedelta(seconds=1),
            ),
        ):
            config = MagicMock()
            config.validate.return_value = []
            mock_config.return_value = config
            await _run_loop(time(8, 30), ET, stop_event)

        assert "completed successfully" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_error_email_failure_is_logged(self, caplog):
        """_run_loop logs when send_error_email itself raises after a briefing failure."""
        stop_event = asyncio.Event()

        async def _fake_briefing(*args, **kwargs):
            stop_event.set()
            raise RuntimeError("Briefing exploded")

        with (
            caplog.at_level(logging.ERROR, logger="volume_price_analysis.agent.scheduler"),
            patch("volume_price_analysis.agent.scheduler.AgentConfig.from_env") as mock_config,
            patch(
                "volume_price_analysis.agent.scheduler.run_morning_briefing",
                side_effect=_fake_briefing,
            ),
            patch(
                "volume_price_analysis.agent.scheduler.send_error_email",
                side_effect=Exception("SMTP down"),
            ),
            patch(
                "volume_price_analysis.agent.scheduler._next_run",
                return_value=datetime.now(ET) - timedelta(seconds=1),
            ),
        ):
            config = MagicMock()
            config.validate.return_value = []
            config.email_from = "a@b.com"
            config.email_password = "pass"
            config.email_to = "c@d.com"
            mock_config.return_value = config
            await _run_loop(time(8, 30), ET, stop_event)

        assert "error email" in caplog.text.lower()


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

    @pytest.mark.asyncio
    async def test_unix_uses_add_signal_handler(self, caplog):
        """On Unix, loop.add_signal_handler is called for SIGTERM and SIGINT."""
        mock_loop = MagicMock()
        with (
            caplog.at_level(logging.INFO, logger="volume_price_analysis.agent.scheduler"),
            patch("volume_price_analysis.agent.scheduler.sys") as mock_sys,
            patch(
                "volume_price_analysis.agent.scheduler.asyncio.get_running_loop",
                return_value=mock_loop,
            ),
            patch(
                "volume_price_analysis.agent.scheduler._run_loop",
                new_callable=AsyncMock,
            ),
        ):
            mock_sys.platform = "linux"
            await run_scheduler(time(8, 30), ET)

            assert mock_loop.add_signal_handler.call_count == 2
            registered_signals = {
                call.args[0] for call in mock_loop.add_signal_handler.call_args_list
            }
            assert registered_signals == {signal.SIGTERM, signal.SIGINT}

            handler = mock_loop.add_signal_handler.call_args_list[0].args[1]
            handler()
            assert "Received shutdown signal" in caplog.text

    @pytest.mark.asyncio
    async def test_windows_uses_signal_signal(self, caplog):
        """On Windows, signal.signal registers SIGINT and uses call_soon_threadsafe."""
        loop = asyncio.get_running_loop()
        with (
            caplog.at_level(logging.INFO, logger="volume_price_analysis.agent.scheduler"),
            patch("volume_price_analysis.agent.scheduler.sys") as mock_sys,
            patch("volume_price_analysis.agent.scheduler.signal.signal") as mock_signal,
            patch.object(
                loop, "call_soon_threadsafe", wraps=loop.call_soon_threadsafe
            ) as mock_soon,
            patch(
                "volume_price_analysis.agent.scheduler._run_loop",
                new_callable=AsyncMock,
            ),
        ):
            mock_sys.platform = "win32"
            await run_scheduler(time(8, 30), ET)

            mock_signal.assert_called_once()
            assert mock_signal.call_args.args[0] == signal.SIGINT

            handler = mock_signal.call_args.args[1]
            handler(signal.SIGINT, None)
            mock_soon.assert_called()
            assert "Received shutdown signal" in caplog.text


class TestSchedulerMain:
    """Test the scheduler CLI entry point (main)."""

    def test_default_time(self):
        """main() schedules 08:30 ET when no --time is given."""
        with (
            patch("sys.argv", ["morning-scheduler"]),
            patch(
                "volume_price_analysis.agent.scheduler.run_scheduler",
            ) as mock_sched,
            patch("volume_price_analysis.agent.scheduler.asyncio.run"),
        ):
            main()
        mock_sched.assert_called_once_with(time(8, 30), ET, skip_holidays=False)

    def test_custom_time(self):
        """main() parses --time into a datetime.time and passes it to run_scheduler."""
        with (
            patch("sys.argv", ["morning-scheduler", "--time", "10:45"]),
            patch(
                "volume_price_analysis.agent.scheduler.run_scheduler",
            ) as mock_sched,
            patch("volume_price_analysis.agent.scheduler.asyncio.run"),
        ):
            main()
        mock_sched.assert_called_once_with(time(10, 45), ET, skip_holidays=False)

    def test_skip_holidays_flag(self):
        """main() forwards --skip-holidays to run_scheduler."""
        with (
            patch("sys.argv", ["morning-scheduler", "--skip-holidays"]),
            patch(
                "volume_price_analysis.agent.scheduler.run_scheduler",
            ) as mock_sched,
            patch("volume_price_analysis.agent.scheduler.asyncio.run"),
        ):
            main()
        mock_sched.assert_called_once_with(time(8, 30), ET, skip_holidays=True)

    def test_invalid_time_format_exits(self, caplog):
        """main() exits when --time is not parseable."""
        with (
            caplog.at_level(logging.ERROR, logger="volume_price_analysis.agent.scheduler"),
            patch("sys.argv", ["morning-scheduler", "--time", "not-a-time"]),
        ):
            with pytest.raises(SystemExit, match="1"):
                main()
        assert "Invalid --time" in caplog.text

    def test_invalid_time_bad_hour(self):
        """main() exits when --time has an hour outside 0-23."""
        with patch("sys.argv", ["morning-scheduler", "--time", "25:00"]):
            with pytest.raises(SystemExit, match="1"):
                main()

    def test_invalid_time_bad_minute(self):
        """main() exits when --time has a minute outside 0-59."""
        with patch("sys.argv", ["morning-scheduler", "--time", "08:99"]):
            with pytest.raises(SystemExit, match="1"):
                main()

    def test_invalid_time_missing_colon(self):
        """main() exits when --time is missing the HH:MM colon separator."""
        with patch("sys.argv", ["morning-scheduler", "--time", "0830"]):
            with pytest.raises(SystemExit, match="1"):
                main()
