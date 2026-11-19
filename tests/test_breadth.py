"""Tests for market breadth (PDE-210): per-symbol observations and their summary."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from volume_price_analysis.analysis import scan_symbol
from volume_price_analysis.breadth import BreadthObservation, observe_breadth, summarize_breadth
from volume_price_analysis.data_fetcher import InMemoryDataSource


def _obs(
    symbol: str,
    *,
    sma20: bool | None = True,
    sma50: bool | None = True,
    up: bool | None = True,
    volume: float = 1_000.0,
    high: bool = False,
    low: bool = False,
) -> BreadthObservation:
    return BreadthObservation(
        symbol=symbol,
        above_sma20=sma20,
        above_sma50=sma50,
        up_close=up,
        volume=volume,
        new_high=high,
        new_low=low,
    )


def _frame(closes: list[float], volume: float = 1_000.0) -> pd.DataFrame:
    n = len(closes)
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-01-02", periods=n),
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": [volume] * n,
        }
    )


class TestSummarizeBreadth:
    def test_mixed_observations(self):
        observations = [
            _obs("AAA", sma20=True, sma50=True, up=True, volume=300, high=True),
            _obs("BBB", sma20=True, sma50=False, up=False, volume=100),
            _obs("CCC", sma20=False, sma50=None, up=False, volume=50, low=True),
            _obs("DDD", sma20=False, sma50=False, up=None, volume=999),
        ]
        summary = summarize_breadth(observations, {"AAA", "BBB", "CCC", "DDD"})
        assert summary == {
            "universe": "sp500",
            "symbols_counted": 4,
            "pct_above_sma20": 50.0,
            # CCC has too little history for SMA50: it leaves that denominator.
            "pct_above_sma50": 33.3,
            "up_volume": 300.0,
            "down_volume": 150.0,
            "up_down_volume_ratio": 2.0,
            "new_highs": 1,
            "new_lows": 1,
            "net_new_highs": 0,
        }

    def test_all_up(self):
        observations = [_obs(s, high=True) for s in ("AAA", "BBB")]
        summary = summarize_breadth(observations, {"AAA", "BBB"})
        assert summary["pct_above_sma20"] == 100.0
        assert summary["pct_above_sma50"] == 100.0
        assert summary["down_volume"] == 0.0
        # No down volume: the ratio is undefined, never infinite (it must serialise).
        assert summary["up_down_volume_ratio"] is None
        assert summary["net_new_highs"] == 2

    def test_empty(self):
        summary = summarize_breadth([], {"AAA"})
        assert summary == {
            "universe": "sp500",
            "symbols_counted": 0,
            "pct_above_sma20": None,
            "pct_above_sma50": None,
            "up_volume": 0.0,
            "down_volume": 0.0,
            "up_down_volume_ratio": None,
            "new_highs": 0,
            "new_lows": 0,
            "net_new_highs": 0,
        }
        json.dumps(summary, allow_nan=False)

    def test_etf_symbols_are_excluded(self):
        observations = [
            _obs("AAPL", sma20=False, up=False, volume=10, low=True),
            # Leveraged, inverse, and plain ETFs are not sp500 members: never counted.
            _obs("TQQQ", volume=1e9, high=True),
            _obs("SQQQ", volume=1e9, high=True),
            _obs("SPY", volume=1e9, high=True),
        ]
        summary = summarize_breadth(observations, {"AAPL", "MSFT"})
        assert summary["symbols_counted"] == 1
        assert summary["pct_above_sma20"] == 0.0
        assert summary["up_volume"] == 0.0
        assert summary["down_volume"] == 10.0
        assert summary["net_new_highs"] == -1


class TestObserveBreadth:
    def test_rising_frame(self):
        obs = observe_breadth("AAA", _frame([float(i) for i in range(1, 61)], volume=7.0))
        assert obs == _obs("AAA", sma20=True, sma50=True, up=True, volume=7.0, high=True)

    def test_falling_frame(self):
        obs = observe_breadth("AAA", _frame([float(i) for i in range(60, 0, -1)]))
        assert obs is not None
        assert (obs.above_sma20, obs.above_sma50, obs.up_close) == (False, False, False)
        assert (obs.new_high, obs.new_low) == (False, True)

    def test_short_history_leaves_sma_unknown(self):
        obs = observe_breadth("AAA", _frame([1.0, 2.0, 2.0]))
        assert obs is not None
        assert obs.above_sma20 is None
        assert obs.above_sma50 is None
        # Unchanged close: neither up nor down.
        assert obs.up_close is None

    def test_too_little_data(self):
        assert observe_breadth("AAA", _frame([1.0])) is None
        assert observe_breadth("AAA", _frame([1.0, float("nan")])) is None

    def test_observation_holds_scalars_only(self):
        """Workers return observations, never frames, so scan memory stays bounded."""
        obs = observe_breadth("AAA", _frame([float(i) for i in range(1, 61)]))
        for value in dataclasses.asdict(obs).values():  # type: ignore[arg-type]
            assert value is None or isinstance(value, (str, bool, float, int))


class TestScanSymbolBreadthIsCausal:
    def test_as_of_cutoff_excludes_future_bars(self):
        rng = np.random.default_rng(7)
        closes = list(100 + np.cumsum(rng.normal(0, 1, 80)))
        extended = _frame(closes)
        n = 60
        # A violent future bar would flip every field if it leaked in.
        extended.loc[n:, ["Close", "High", "Low"]] = 10_000.0
        as_of = extended["Date"].iloc[n].date()
        source = InMemoryDataSource({"AAA": extended})

        _, obs = scan_symbol(
            "AAA", "1y", 14, 0.0, 0.0, 100.0, "any", data_source=source, as_of=as_of
        )

        assert obs == observe_breadth("AAA", extended.iloc[:n].reset_index(drop=True))


class TestRunScanBreadth:
    @pytest.mark.asyncio
    async def test_summary_carries_breadth_over_sp500_members(self):
        from volume_price_analysis.analysis import run_scan

        up = _frame([float(i) for i in range(1, 61)])
        down = _frame([float(i) for i in range(60, 0, -1)])
        source = InMemoryDataSource({"AAA": up, "BBB": down, "TQQQ": up})
        result = await run_scan(
            universe="full_market",
            universes={"sp500": ["AAA", "BBB"], "full_market": ["AAA", "BBB", "TQQQ"]},
            data_source=source,
        )
        breadth = result["summary"]["breadth"]
        assert breadth["universe"] == "sp500"
        assert breadth["symbols_counted"] == 2
        assert breadth["pct_above_sma20"] == 50.0
        assert breadth["net_new_highs"] == 0
        json.dumps(result, allow_nan=False)

    @pytest.mark.asyncio
    async def test_skipped_and_errored_symbols_are_not_counted(self):
        from volume_price_analysis.analysis import run_scan

        source = InMemoryDataSource({"AAA": _frame([1.0, 2.0])})  # too short to scan
        result = await run_scan(
            symbols=["AAA", "MISSING"],
            universes={"sp500": ["AAA", "MISSING"], "full_market": []},
            data_source=source,
        )
        assert result["summary"]["breadth"]["symbols_counted"] == 0
