"""Tests for the scheduler liveness heartbeat and healthcheck probe."""

import asyncio
import logging
import os
import tempfile
from datetime import datetime, time, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from volume_price_analysis.agent import healthcheck, scheduler
from volume_price_analysis.agent.healthcheck import (
    HEARTBEAT_ENV_VAR,
    HEARTBEAT_FILENAME,
    HEARTBEAT_MAX_AGE_SECONDS,
    check_heartbeat,
    default_heartbeat_path,
    heartbeat_path,
    write_heartbeat,
)
from volume_price_analysis.agent.scheduler import _run_loop, _wait_for_next_run

ET = ZoneInfo("America/New_York")


class TestCheckHeartbeat:
    def test_missing_file_is_unhealthy(self, tmp_path):
        reason = check_heartbeat(tmp_path / "absent")
        assert reason is not None
        assert "does not exist" in reason

    def test_fresh_stamp_is_healthy(self, tmp_path):
        hb = tmp_path / "hb"
        write_heartbeat(hb, clock=lambda: 1000.0)
        assert check_heartbeat(hb, max_age_seconds=60, now=1030.0) is None

    def test_stale_stamp_is_unhealthy(self, tmp_path):
        hb = tmp_path / "hb"
        write_heartbeat(hb, clock=lambda: 1000.0)
        reason = check_heartbeat(hb, max_age_seconds=60, now=1061.0)
        assert reason is not None
        assert "old" in reason

    def test_future_stamp_is_unhealthy(self, tmp_path):
        # A monotonic reading ahead of ours was written under a different boot;
        # it must not read as fresh.
        hb = tmp_path / "hb"
        write_heartbeat(hb, clock=lambda: 5000.0)
        reason = check_heartbeat(hb, max_age_seconds=60, now=1000.0)
        assert reason is not None
        assert "future" in reason

    def test_garbage_content_is_unhealthy(self, tmp_path):
        hb = tmp_path / "hb"
        hb.write_text("not a number")
        reason = check_heartbeat(hb)
        assert reason is not None
        assert "monotonic timestamp" in reason

    def test_uses_monotonic_clock_by_default(self, tmp_path):
        hb = tmp_path / "hb"
        write_heartbeat(hb)
        assert check_heartbeat(hb) is None

    def test_wall_clock_step_does_not_matter(self, tmp_path):
        # Only the file's contents (monotonic) are consulted, never its mtime.
        hb = tmp_path / "hb"
        write_heartbeat(hb, clock=lambda: 1000.0)

        os.utime(hb, (0, 0))  # mtime at the epoch: "stale" by wall clock
        assert check_heartbeat(hb, max_age_seconds=60, now=1010.0) is None

    def test_unreadable_path_is_unhealthy(self):
        hb = MagicMock()
        hb.read_text.side_effect = PermissionError("nope")
        reason = check_heartbeat(hb)
        assert reason is not None
        assert "cannot read" in reason

    def test_max_age_covers_scheduler_sleep_chunk(self):
        # The loop only writes the file when it wakes; the probe must tolerate
        # at least two missed wakes plus a briefing run before crying wolf.
        assert HEARTBEAT_MAX_AGE_SECONDS >= 2 * scheduler._MAX_SLEEP_CHUNK_SECONDS


class TestHeartbeatPath:
    def test_default_is_in_platform_tempdir(self, monkeypatch):
        monkeypatch.delenv(HEARTBEAT_ENV_VAR, raising=False)
        assert heartbeat_path() == Path(tempfile.gettempdir()) / HEARTBEAT_FILENAME
        assert heartbeat_path() == default_heartbeat_path()

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(HEARTBEAT_ENV_VAR, str(tmp_path / "custom"))
        assert heartbeat_path() == tmp_path / "custom"

    def test_empty_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(HEARTBEAT_ENV_VAR, "")
        assert heartbeat_path() == default_heartbeat_path()


class TestWriteHeartbeat:
    def test_writes_monotonic_reading(self, tmp_path):
        hb = tmp_path / "hb"
        write_heartbeat(hb, clock=lambda: 123.4567)
        assert hb.read_text() == "123.457\n"
        assert not (tmp_path / "hb.tmp").exists()

    def test_overwrites_atomically(self, tmp_path):
        hb = tmp_path / "hb"
        write_heartbeat(hb, clock=lambda: 1.0)
        write_heartbeat(hb, clock=lambda: 2.0)
        assert hb.read_text() == "2.000\n"

    def test_unwritable_path_logs_without_traceback_and_does_not_raise(self, tmp_path, caplog):
        hb = tmp_path / "missing-dir" / "hb"
        with caplog.at_level(logging.WARNING):
            write_heartbeat(hb)
        assert "Could not write heartbeat" in caplog.text
        assert "Traceback" not in caplog.text


class TestMain:
    def test_healthy_exit_zero(self, tmp_path, monkeypatch, capsys):
        hb = tmp_path / "hb"
        write_heartbeat(hb)
        monkeypatch.setenv(HEARTBEAT_ENV_VAR, str(hb))
        healthcheck.main()
        assert "healthy" in capsys.readouterr().out

    def test_unhealthy_exit_one(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv(HEARTBEAT_ENV_VAR, str(tmp_path / "absent"))
        with pytest.raises(SystemExit, match="1"):
            healthcheck.main()
        assert "unhealthy" in capsys.readouterr().err


class TestSchedulerHeartbeat:
    """The loop must keep the heartbeat fresh while sleeping and around runs."""

    @pytest.mark.asyncio
    async def test_wait_writes_heartbeat_on_each_wake(self, tmp_path):
        hb = tmp_path / "hb"
        stop_event = asyncio.Event()
        next_dt = datetime.now(ET) + timedelta(days=365)
        writes = 0
        original = healthcheck.write_heartbeat

        def counting_write(path):
            nonlocal writes
            writes += 1
            original(path)
            if writes >= 3:
                stop_event.set()

        with (
            patch("volume_price_analysis.agent.scheduler._MAX_SLEEP_CHUNK_SECONDS", 0.01),
            patch("volume_price_analysis.agent.scheduler.write_heartbeat", counting_write),
        ):
            result = await asyncio.wait_for(
                _wait_for_next_run(next_dt, time(8, 30), ET, stop_event, heartbeat=hb),
                timeout=5,
            )
        assert result is None
        assert writes >= 3
        assert check_heartbeat(hb) is None

    @pytest.mark.asyncio
    async def test_wait_without_heartbeat_writes_nothing(self):
        stop_event = asyncio.Event()
        stop_event.set()
        with patch("volume_price_analysis.agent.scheduler.write_heartbeat") as write:
            await _wait_for_next_run(datetime.now(ET), time(8, 30), ET, stop_event)
        write.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_loop_writes_heartbeat_around_briefing(self, tmp_path):
        hb = tmp_path / "hb"
        stop_event = asyncio.Event()
        fired = datetime.now(ET)

        async def _fire_once(next_dt, *args, **kwargs):
            return fired

        async def _briefing(config):
            # Heartbeat was written before the run started.
            assert hb.exists()
            hb.unlink()
            stop_event.set()
            return MagicMock(degraded=False)

        with (
            patch("volume_price_analysis.agent.scheduler.AgentConfig.from_env") as mock_config,
            patch(
                "volume_price_analysis.agent.scheduler._wait_for_next_run",
                side_effect=_fire_once,
            ),
            patch(
                "volume_price_analysis.agent.scheduler.run_morning_briefing",
                new_callable=AsyncMock,
                side_effect=_briefing,
            ),
        ):
            config = MagicMock()
            config.validate.return_value = []
            mock_config.return_value = config
            await _run_loop(time(8, 30), ET, stop_event, heartbeat=hb)

        # ...and again after it finished.
        assert hb.exists()

    @pytest.mark.asyncio
    async def test_run_loop_writes_heartbeat_after_failed_briefing(self, tmp_path):
        hb = tmp_path / "hb"
        stop_event = asyncio.Event()

        async def _fire_once(next_dt, *args, **kwargs):
            return datetime.now(ET)

        async def _boom(config):
            hb.unlink()
            stop_event.set()
            raise RuntimeError("scan exploded")

        with (
            patch("volume_price_analysis.agent.scheduler.AgentConfig.from_env") as mock_config,
            patch(
                "volume_price_analysis.agent.scheduler._wait_for_next_run",
                side_effect=_fire_once,
            ),
            patch(
                "volume_price_analysis.agent.scheduler.run_morning_briefing",
                new_callable=AsyncMock,
                side_effect=_boom,
            ),
        ):
            config = MagicMock()
            config.validate.return_value = []
            config.email_from = ""
            mock_config.return_value = config
            await _run_loop(time(8, 30), ET, stop_event, heartbeat=hb)

        assert hb.exists()

    def test_main_passes_heartbeat_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv(HEARTBEAT_ENV_VAR, str(tmp_path / "hb"))
        monkeypatch.setattr("sys.argv", ["morning-scheduler"])
        captured = {}

        async def _fake_run_scheduler(target, tz, skip_holidays=False, heartbeat=None):
            captured["heartbeat"] = heartbeat

        with patch("volume_price_analysis.agent.scheduler.run_scheduler", _fake_run_scheduler):
            scheduler.main()
        assert captured["heartbeat"] == tmp_path / "hb"
