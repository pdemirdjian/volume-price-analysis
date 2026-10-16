"""Briefing configuration contracts."""

import pytest

from volume_price_analysis.agent.config import AgentConfig

SCAN_DEFAULTS = {
    "period": "3mo",
    "holding_period": 14,
    "min_score": 2.0,
    "min_adx": 20,
    "max_iv_percentile": 70,
    "min_avg_daily_volume": 500_000,
    "max_results": 15,
}


@pytest.mark.parametrize("name,default", SCAN_DEFAULTS.items())
def test_scan_env_defaults_and_overrides(monkeypatch, name, default):
    key = f"SCAN_{name.upper()}"
    monkeypatch.delenv(key, raising=False)
    assert getattr(AgentConfig.from_env(), f"scan_{name}") == default
    value = "6mo" if name == "period" else "30"
    monkeypatch.setenv(key, value)
    assert getattr(AgentConfig.from_env(), f"scan_{name}") == (value if name == "period" else 30)


@pytest.mark.parametrize("name", [n for n in SCAN_DEFAULTS if n != "period"])
def test_bad_numeric_scan_env_fails(monkeypatch, name):
    key = f"SCAN_{name.upper()}"
    monkeypatch.setenv(key, "invalid")
    with pytest.raises(ValueError, match=key):
        AgentConfig.from_env()


@pytest.mark.parametrize(
    "name,value",
    [
        ("universe", "unknown"),
        ("period", "bad"),
        ("holding_period", 0),
        ("holding_period", 91),
        ("min_score", -1),
        ("min_adx", -1),
        ("max_iv_percentile", 101),
        ("max_iv_percentile", -1),
        ("min_avg_daily_volume", -1),
        ("max_results", 0),
        ("max_results", 101),
        ("min_score", float("nan")),
        ("min_adx", float("inf")),
    ],
)
@pytest.mark.parametrize(
    "dry_run,no_ai", [(False, False), (False, True), (True, False), (True, True)]
)
def test_invalid_scan_settings(name, value, dry_run, no_ai):
    config = AgentConfig(**{f"scan_{name}": value})
    from volume_price_analysis.agent.morning_agent import _config_errors

    errors = _config_errors(config, dry_run=dry_run, no_ai=no_ai)
    assert any(f"SCAN_{name.upper()}" in error for error in errors)


@pytest.mark.parametrize("level", ["DEBUG", "verbose"])
def test_server_startup_logging_uses_stderr(level):
    import os
    import subprocess
    import sys

    script = """
import asyncio
import logging
from contextlib import asynccontextmanager
from volume_price_analysis import server
@asynccontextmanager
async def stdio():
    logging.getLogger("volume_price_analysis.test").debug("startup debug")
    logging.getLogger("volume_price_analysis.test").info("startup info")
    raise RuntimeError("stop before serving")
    yield
server.stdio_server = stdio
try:
    asyncio.run(server.main())
except RuntimeError:
    pass
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "LOG_LEVEL": level},
        check=True,
    )
    assert result.stdout == ""
    assert ("startup debug" in result.stderr) == (level == "DEBUG")
    if level == "verbose":
        assert "Invalid LOG_LEVEL=" in result.stderr
        assert "using INFO" in result.stderr
    assert "startup info" in result.stderr


@pytest.mark.parametrize("no_ai", [False, True])
@pytest.mark.parametrize("value", [0, 21])
def test_dry_run_rejects_invalid_deep_analysis_limit(no_ai, value):
    from volume_price_analysis.agent.morning_agent import _config_errors

    errors = _config_errors(AgentConfig(max_deep_analysis=value), dry_run=True, no_ai=no_ai)
    assert any("max_deep_analysis" in error for error in errors)
