"""Tests for sector context (PDE-248): sector tags, sector relative strength, conflicts."""

from datetime import date

import pandas as pd
import pytest

from volume_price_analysis.analysis import _build_sp500_members, get_symbol_sector, run_scan
from volume_price_analysis.breadth import BreadthObservation, observe_breadth
from volume_price_analysis.data_fetcher import InMemoryDataSource
from volume_price_analysis.sectors import (
    SECTOR_ETFS,
    sector_conflict,
    sector_from_industries,
    sector_relative_strength,
)

SESSION = date(2026, 3, 27)


def _obs(
    symbol: str,
    r20: float | None,
    r5: float | None = 0.0,
    session: date | None = SESSION,
) -> BreadthObservation:
    return BreadthObservation(
        symbol=symbol,
        session=session,
        above_sma20=None,
        above_sma50=None,
        up_close=None,
        volume=0.0,
        new_high=False,
        new_low=False,
        return_5d=r5,
        return_20d=r20,
    )


def _frame(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-01-02", periods=len(closes)),
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": [1_000.0] * len(closes),
        }
    )


class TestSectorFromIndustries:
    def test_gics_sector_entry(self):
        assert sector_from_industries(["Information Technology", "Semiconductors"]) == (
            "Information Technology"
        )

    def test_aliases_map_to_one_sector(self):
        assert sector_from_industries(["Healthcare"]) == "Health Care"
        assert sector_from_industries(["Technology Equipment", "Technology"]) == (
            "Information Technology"
        )
        assert sector_from_industries(["Real estate"]) == "Real Estate"

    def test_last_sector_entry_wins(self):
        # The payload tags utilities ['Energy', 'Utilities']; GICS says Utilities.
        assert sector_from_industries(["Energy", "Utilities"]) == "Utilities"

    def test_free_text_industries_map_to_none(self):
        assert sector_from_industries(["Conglomerate", "Retail"]) is None
        assert sector_from_industries([]) is None
        assert sector_from_industries(None) is None
        assert sector_from_industries([None, 3]) is None

    def test_every_sector_has_an_etf_in_the_etf_universe(self):
        from volume_price_analysis.analysis import get_universes

        assert set(SECTOR_ETFS.values()) <= set(get_universes()["etfs"])


class TestSp500Members:
    def test_bundled_payload_tags_members_with_sectors(self):
        members = _build_sp500_members()
        assert members["NVDA"] == "Information Technology"
        assert set(members.values()) - {None} <= set(SECTOR_ETFS)

    def test_etfs_and_unknown_symbols_have_no_sector(self):
        assert get_symbol_sector("XLK") is None
        assert get_symbol_sector("SPY") is None
        assert get_symbol_sector("NOT_A_SYMBOL") is None

    def test_fallback_members_have_no_sector(self, mocker):
        mocker.patch(
            "volume_price_analysis.analysis.PyTickerSymbols", side_effect=RuntimeError("boom")
        )
        members = _build_sp500_members()
        assert "AAPL" in members
        assert set(members.values()) == {None}


class TestSectorRelativeStrength:
    def test_ranked_by_20_day_rs_strongest_first(self):
        rows = sector_relative_strength(
            [
                _obs("SPY", r20=2.0, r5=1.0),
                _obs("XLE", r20=-3.0, r5=-1.5),
                _obs("XLK", r20=6.5, r5=0.5),
                _obs("XLF", r20=2.5, r5=2.0),
                _obs("AAPL", r20=40.0),  # not a sector ETF: ignored
            ]
        )
        assert [r["etf"] for r in rows] == ["XLK", "XLF", "XLE"]
        assert [r["rank"] for r in rows] == [1, 2, 3]
        assert rows[0] == {
            "sector": "Information Technology",
            "etf": "XLK",
            "return_5d": 0.5,
            "return_20d": 6.5,
            "rs_5d": -0.5,
            "rs_20d": 4.5,
            "rank": 1,
        }
        assert rows[-1]["rs_20d"] == -5.0

    def test_ties_break_on_etf(self):
        rows = sector_relative_strength([_obs("SPY", 0.0), _obs("XLU", 1.0), _obs("XLB", 1.0)])
        assert [r["etf"] for r in rows] == ["XLB", "XLU"]

    def test_missing_spy_yields_nothing(self):
        assert sector_relative_strength([_obs("XLK", 5.0), _obs("XLE", -1.0)]) == []

    def test_spy_with_insufficient_bars_yields_nothing(self):
        assert sector_relative_strength([_obs("SPY", None, None), _obs("XLK", 5.0)]) == []

    def test_sector_etf_with_insufficient_bars_is_skipped(self):
        rows = sector_relative_strength(
            [_obs("SPY", 1.0), _obs("XLK", None, r5=3.0), _obs("XLE", 2.0, r5=None)]
        )
        assert [r["etf"] for r in rows] == ["XLE"]
        assert rows[0]["rs_5d"] is None

    def test_sector_etf_on_a_stale_session_is_skipped(self):
        rows = sector_relative_strength(
            [_obs("SPY", 1.0), _obs("XLK", 5.0, session=date(2026, 3, 26)), _obs("XLE", 2.0)]
        )
        assert [r["etf"] for r in rows] == ["XLE"]

    def test_observations_carry_the_returns(self):
        # 21 closes are the minimum for a 20-session return; 20 are too few.
        enough = observe_breadth("SPY", _frame([100.0] * 20 + [110.0]))
        short = observe_breadth("SPY", _frame([100.0] * 20))
        assert enough is not None and short is not None
        assert enough.return_20d == pytest.approx(10.0)
        assert enough.return_5d == pytest.approx(10.0)
        assert short.return_20d is None
        assert short.return_5d == 0.0


STRENGTH = [
    {"sector": "Information Technology", "etf": "XLK", "rs_20d": 4.0, "rank": 1},
    {"sector": "Financials", "etf": "XLF", "rs_20d": 0.5, "rank": 2},
    {"sector": "Energy", "etf": "XLE", "rs_20d": -3.0, "rank": 3},
]


class TestSectorConflict:
    def test_bullish_pick_in_weakest_sector(self):
        candidate = {"symbol": "XOM", "composite_score": 4.2, "sector": "Energy"}
        assert sector_conflict(candidate, STRENGTH) == (
            "bullish setup in the weakest sector (Energy, XLE)"
        )

    def test_bearish_pick_in_strongest_sector(self):
        candidate = {"symbol": "NVDA", "composite_score": -3.0, "sector": "Information Technology"}
        assert sector_conflict(candidate, STRENGTH) == (
            "bearish setup in the strongest sector (Information Technology, XLK)"
        )

    @pytest.mark.parametrize(
        "candidate",
        [
            {"composite_score": 4.2, "sector": "Information Technology"},  # aligned
            {"composite_score": -4.2, "sector": "Energy"},  # aligned
            {"composite_score": 4.2, "sector": "Financials"},  # middle of the pack
            {"composite_score": 4.2, "sector": None},
            {"composite_score": None, "sector": "Energy"},
            {"composite_score": "x", "sector": "Energy"},
        ],
    )
    def test_no_conflict(self, candidate):
        assert sector_conflict(candidate, STRENGTH) is None

    def test_needs_two_ranked_sectors(self):
        candidate = {"composite_score": 4.2, "sector": "Energy"}
        assert sector_conflict(candidate, STRENGTH[2:]) is None
        assert sector_conflict(candidate, []) is None


class TestRunScanSectors:
    @pytest.mark.asyncio
    async def test_candidates_tagged_and_summary_ranks_sectors(self):
        rising = _frame([100.0 + i for i in range(80)])
        flat = _frame([100.0 + (i % 2) * 0.1 for i in range(80)])
        source = InMemoryDataSource({"NVDA": rising, "XLK": rising, "XLE": flat, "SPY": flat})
        result = await run_scan(
            universe="full_market",
            universes={"sp500": ["NVDA"], "full_market": ["NVDA", "XLK", "XLE", "SPY"]},
            data_source=source,
            min_score=0,
            min_adx=0,
        )
        sectors = {c["symbol"]: c.get("sector") for c in result["top_bullish"]}
        assert sectors.get("NVDA") == "Information Technology"
        assert sectors.get("XLK", None) is None
        strength = result["summary"]["sector_strength"]
        assert [r["etf"] for r in strength] == ["XLK", "XLE"]

    @pytest.mark.asyncio
    async def test_universe_without_sector_etfs_ranks_nothing(self):
        source = InMemoryDataSource({"NVDA": _frame([100.0 + i for i in range(80)])})
        result = await run_scan(symbols=["NVDA"], data_source=source)
        assert result["summary"]["sector_strength"] == []
