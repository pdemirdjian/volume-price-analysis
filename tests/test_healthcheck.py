"""Tests for the scheduler liveness heartbeat and healthcheck probe."""

import asyncio
import logging
from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from volume_price_analysis.agent import healthcheck, scheduler
from volume_price_analysis.agent.healthcheck import (
    DEFAULT_HEARTBEAT_PATH,
    HEARTBEAT_ENV_VAR,
    HEARTBEAT_MAX_AGE_SECONDS,
    check_heartbeat,
    heartbeat_path,
    touch_heartbeat,
)
from volume_price_analysis.agent.scheduler import _run_loop, _wait_for_next_run

ET = ZoneInfo("America/New_York")


class TestCheckHeartbeat:
    def test_missing_file_is_unhealthy(self, tmp_path):
        reason = check_heartbeat(tmp_path / "absent")
        assert reason is not None
        assert "does not exist" in reason

    def test_fresh_file_is_healthy(self, tmp_path):
        hb = tmp_path / "hb"
        hb.touch()
        assert check_heartbeat(hb, max_age_seconds=60, now=hb.stat().st_mtime + 30) is None

    def test_stale_file_is_unhealthy(self, tmp_path):
        hb = tmp_path / "hb"
        hb.touch()
        reason = check_heartbeat(hb, max_age_seconds=60, now=hb.stat().st_mtime + 61)
        assert reason is not None
        assert "old" in reason

    def test_uses_wall_clock_by_default(self, tmp_path):
        hb = tmp_path / "hb"
        hb.touch()
        assert check_heartbeat(hb) is None

    def test_unreadable_path_is_unhealthy(self, tmp_path):
        # A directory where a file is expected: stat succeeds on POSIX, so
        # force the OSError branch explicitly.
        hb = MagicMock()
        hb.stat.side_effect = PermissionError("nope")
        reason = check_heartbeat(hb)
        assert reason is not None
        assert "cannot read" in reason

    def test_max_age_covers_scheduler_sleep_chunk(self):
        # The loop only touches the file when it wakes; the probe must tolerate
        # at least two missed wakes plus a briefing run before crying wolf.
        assert HEARTBEAT_MAX_AGE_SECONDS >= 2 * scheduler._MAX_SLEEP_CHUNK_SECONDS


class TestHeartbeatPath:
    def test_default(self, monkeypatch):
        monkeypatch.delenv(HEARTBEAT_ENV_VAR, raising=False)
        assert heartbeat_path() == DEFAULT_HEARTBEAT_PATH

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(HEARTBEAT_ENV_VAR, str(tmp_path / "custom"))
        assert heartbeat_path() == tmp_path / "custom"

    def test_empty_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(HEARTBEAT_ENV_VAR, "")
        assert heartbeat_path() == DEFAULT_HEARTBEAT_PATH


class TestTouchHeartbeat:
    def test_creates_and_refreshes(self, tmp_path):
        hb = tmp_path / "hb"
        touch_heartbeat(hb)
        assert hb.exists()
        first = hb.stat().st_mtime_ns
        touch_heartbeat(hb)
        assert hb.stat().st_mtime_ns >= first

    def test_unwritable_path_logs_and_does_not_raise(self, tmp_path, caplog):
        hb = tmp_path / "missing-dir" / "hb"
        with caplog.at_level(logging.WARNING):
            touch_heartbeat(hb)
        assert "Could not update heartbeat" in caplog.text


class TestMain:
    def test_healthy_exit_zero(self, tmp_path, monkeypatch, capsys):
        hb = tmp_path / "hb"
        hb.touch()
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
    async def test_wait_touches_heartbeat_on_each_wake(self, tmp_path):
        hb = tmp_path / "hb"
        stop_event = asyncio.Event()
        next_dt = datetime.now(ET).replace(microsecond=0)
        next_dt = next_dt.replace(year=next_dt.year + 1)
        touches = 0
        original = healthcheck.touch_heartbeat

        def counting_touch(path):
            nonlocal touches
            touches += 1
            original(path)
            if touches >= 3:
                stop_event.set()

        with (
            patch("volume_price_analysis.agent.scheduler._MAX_SLEEP_CHUNK_SECONDS", 0.01),
            patch("volume_price_analysis.agent.scheduler.touch_heartbeat", counting_touch),
        ):
            result = await asyncio.wait_for(
                _wait_for_next_run(next_dt, time(8, 30), ET, stop_event, heartbeat=hb),
                timeout=5,
            )
        assert result is None
        assert touches >= 3
        assert hb.exists()

    @pytest.mark.asyncio
    async def test_wait_without_heartbeat_touches_nothing(self):
        stop_event = asyncio.Event()
        stop_event.set()
        with patch("volume_price_analysis.agent.scheduler.touch_heartbeat") as touch:
            await _wait_for_next_run(datetime.now(ET), time(8, 30), ET, stop_event)
        touch.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_loop_touches_heartbeat_around_briefing(self, tmp_path):
        hb = tmp_path / "hb"
        stop_event = asyncio.Event()
        fired = datetime.now(ET)

        async def _fire_once(next_dt, *args, **kwargs):
            return fired

        async def _briefing(config):
            # Heartbeat was touched before the run started.
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
    async def test_run_loop_touches_heartbeat_after_failed_briefing(self, tmp_path):
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
