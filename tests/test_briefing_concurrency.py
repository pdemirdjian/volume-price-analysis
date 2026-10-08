"""The briefing keeps the event loop responsive while gathering ordered analyses."""

import asyncio
from threading import Event, Lock, get_ident

from volume_price_analysis.agent.ai_client import PROVIDERS
from volume_price_analysis.agent.config import AgentConfig
from volume_price_analysis.agent.morning_agent import run_morning_briefing
from volume_price_analysis.data_fetcher import InMemoryDataSource


async def test_deep_analysis_is_bounded_concurrent_and_ordered(mocker, sample_stock_data, caplog):
    symbols = ["AAPL", "MSFT", "FAIL", "NVDA", "AMD", "META"]
    release = Event()
    started = asyncio.Event()
    loop = asyncio.get_running_loop()
    lock = Lock()
    active = 0
    peak = 0
    timed_out = False
    completed = []

    class BlockingSource(InMemoryDataSource):
        def fetch(self, symbol, **kwargs):
            nonlocal active, peak, timed_out
            if symbol == "SPY":
                return super().fetch(symbol, **kwargs)
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 4:
                    loop.call_soon_threadsafe(started.set)
            try:
                # A finite wait keeps regressions from hanging the test suite.
                if not release.wait(2):
                    timed_out = True
                # Force the first symbol to finish after the other initial workers.
                if symbol == "AAPL":
                    assert last_initial_finished.wait(2)
                result = super().fetch(symbol, **kwargs)
                with lock:
                    completed.append(symbol)
                return result
            finally:
                if symbol == "NVDA":
                    last_initial_finished.set()
                with lock:
                    active -= 1

    last_initial_finished = Event()
    source = BlockingSource(
        frames=dict.fromkeys(["SPY", *symbols], sample_stock_data),
        errors={"FAIL": ValueError("unavailable")},
    )
    mocker.patch(
        "volume_price_analysis.agent.morning_agent.run_scan",
        return_value={
            "summary": {
                "total_candidates": 6,
                "bullish_setups": 6,
                "bearish_setups": 0,
                "high_conviction": 0,
            },
            "high_conviction_setups": [],
            "top_bullish": [{"symbol": symbol} for symbol in symbols],
            "top_bearish": [],
        },
    )

    async def release_from_event_loop():
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            # Give any unbounded extra workers time to enter before releasing the batch.
            await asyncio.sleep(0.05)
        finally:
            release.set()

    result, _ = await asyncio.gather(
        run_morning_briefing(
            AgentConfig(max_deep_analysis=6), dry_run=True, no_ai=True, data_source=source
        ),
        release_from_event_loop(),
    )

    assert not timed_out, "The event loop could not release the blocking fetches"
    assert peak == 4
    assert completed.index("NVDA") < completed.index("AAPL")
    assert result.symbols_analyzed == ["AAPL", "MSFT", "NVDA", "AMD", "META"]
    assert "Failed to analyze FAIL" in caplog.text


async def test_network_stages_and_ai_backoff_run_off_event_loop(mocker, sample_stock_data):
    loop_thread = get_ident()
    stages = []

    def record(stage):
        assert get_ident() != loop_thread, f"{stage} blocked the event loop"
        stages.append(stage)

    class CheckingSource(InMemoryDataSource):
        def fetch(self, symbol, **kwargs):
            record("regime" if symbol == "SPY" else "deep analysis")
            return super().fetch(symbol, **kwargs)

        def earnings_date(self, symbol):
            record("earnings")
            return super().earnings_date(symbol)

    mocker.patch(
        "volume_price_analysis.agent.morning_agent.run_scan",
        return_value={
            "summary": {
                "total_candidates": 1,
                "bullish_setups": 1,
                "bearish_setups": 0,
                "high_conviction": 0,
            },
            "high_conviction_setups": [],
            "top_bullish": [{"symbol": "AAPL"}],
            "top_bearish": [],
        },
    )

    def provider(*args, **kwargs):
        record("AI")
        if stages.count("AI") == 1:
            raise TimeoutError("retry this request")
        return "Briefing prose"

    mocker.patch.dict(PROVIDERS, {"anthropic": provider})
    mocker.patch("volume_price_analysis.retry._sleep", side_effect=lambda _: record("backoff"))
    mocker.patch(
        "volume_price_analysis.agent.morning_agent.send_email",
        side_effect=lambda *args: record("SMTP"),
    )
    source = CheckingSource(frames=dict.fromkeys(["SPY", "AAPL"], sample_stock_data))
    config = AgentConfig(
        ai_provider="anthropic",
        email_from="sender@example.test",
        email_to="recipient@example.test",
    )
    result = await run_morning_briefing(config, data_source=source)
    assert not result.degraded
    assert result.email_sent
    assert stages == ["regime", "deep analysis", "earnings", "AI", "backoff", "AI", "SMTP"]

    stages.clear()
    result = await run_morning_briefing(config, no_ai=True, data_source=source)
    assert result.email_sent
    assert stages == ["regime", "deep analysis", "earnings", "SMTP"]
