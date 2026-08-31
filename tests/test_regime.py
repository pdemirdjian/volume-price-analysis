"""Tests for the market-regime annotation applied to morning briefings (PDE-66)."""

import numpy as np
import pandas as pd
import pytest

from volume_price_analysis.agent.regime import (
    REGIME_SMA_PERIOD,
    annotate_regime_conflicts,
    compute_market_regime,
    format_regime_header,
)


def make_spy_data(closes: list[float], start: str = "2026-07-01") -> pd.DataFrame:
    """Build a minimal SPY daily DataFrame with the columns fetch_stock_data returns."""
    dates = pd.bdate_range(start=start, periods=len(closes))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        }
    )


def make_candidate(symbol: str, score: float, **extra) -> dict:
    return {"symbol": symbol, "composite_score": score, "adx": 30, "iv_percentile": 40, **extra}


class TestComputeMarketRegime:
    """compute_market_regime: SPY prior close vs its 20-day SMA, strictly causal."""

    def test_bullish_when_close_above_sma(self):
        # 29 flat sessions at 600, final close well above -> above SMA
        regime = compute_market_regime(make_spy_data([600.0] * 29 + [650.0]))
        assert regime["regime"] == "bullish"
        assert regime["spy_close"] == pytest.approx(650.0)
        assert regime["sma20"] == pytest.approx((600.0 * 19 + 650.0) / 20)
        assert regime["close_vs_sma_pct"] > 0

    def test_bearish_when_close_below_sma(self):
        regime = compute_market_regime(make_spy_data([600.0] * 29 + [550.0]))
        assert regime["regime"] == "bearish"
        assert regime["close_vs_sma_pct"] < 0

    def test_uses_last_window_only(self):
        # Old history is noise; the SMA must come from the final 20 closes.
        closes = [100.0] * 30 + [600.0] * 20
        regime = compute_market_regime(make_spy_data(closes))
        assert regime["sma20"] == pytest.approx(600.0)
        assert regime["regime"] == "bullish"  # tie counts as bullish

    def test_tie_counts_as_bullish(self):
        regime = compute_market_regime(make_spy_data([600.0] * REGIME_SMA_PERIOD))
        assert regime["spy_close"] == pytest.approx(regime["sma20"])
        assert regime["regime"] == "bullish"

    def test_exactly_period_rows_is_enough(self):
        regime = compute_market_regime(make_spy_data([600.0] * REGIME_SMA_PERIOD))
        assert regime["regime"] in ("bullish", "bearish")

    def test_short_history_is_unknown(self):
        regime = compute_market_regime(make_spy_data([600.0] * (REGIME_SMA_PERIOD - 1)))
        assert regime["regime"] == "unknown"
        assert "reason" in regime

    def test_none_and_empty_are_unknown(self):
        assert compute_market_regime(None)["regime"] == "unknown"
        assert compute_market_regime(pd.DataFrame())["regime"] == "unknown"

    def test_nan_closes_are_dropped_not_counted(self):
        # 25 rows but 6 NaN -> only 19 valid closes -> unknown (PDE-12 lesson:
        # never let NaN flow into the arithmetic).
        closes = [600.0] * 25
        data = make_spy_data(closes)
        data.loc[data.index[:6], "Close"] = np.nan
        regime = compute_market_regime(data)
        assert regime["regime"] == "unknown"

    def test_as_of_reflects_last_session(self):
        data = make_spy_data([600.0] * 30, start="2026-07-01")
        regime = compute_market_regime(data)
        assert regime["as_of"] == data["Date"].iloc[-1].strftime("%Y-%m-%d")

    def test_missing_close_column_is_unknown(self):
        data = pd.DataFrame({"Date": pd.bdate_range("2026-07-01", periods=30)})
        assert compute_market_regime(data)["regime"] == "unknown"

    def test_today_excludes_in_progress_session(self):
        # 30 flat sessions then an intraday bar dated "today": the live bar
        # must not enter the check — spec requires the prior session's close.
        data = make_spy_data([600.0] * 30 + [550.0])
        today = data["Date"].iloc[-1].date()
        regime = compute_market_regime(data, today=today)
        assert regime["spy_close"] == pytest.approx(600.0)
        assert regime["regime"] == "bullish"
        assert regime["as_of"] == data["Date"].iloc[-2].strftime("%Y-%m-%d")

    def test_today_after_all_bars_changes_nothing(self):
        data = make_spy_data([600.0] * 29 + [650.0])
        today = (data["Date"].iloc[-1] + pd.Timedelta(days=1)).date()
        assert compute_market_regime(data, today=today) == compute_market_regime(data)

    def test_today_excluding_everything_is_unknown(self):
        data = make_spy_data([600.0] * 30)
        today = data["Date"].iloc[0].date()
        assert compute_market_regime(data, today=today)["regime"] == "unknown"


class TestAnnotateRegimeConflicts:
    """annotate_regime_conflicts: counter-regime picks are flagged, never demoted."""

    def make_scan(self):
        bull = make_candidate("AAPL", 4.5)
        bear = make_candidate("XOM", -4.2)
        return {
            "summary": {
                "total_candidates": 2,
                "bullish_setups": 1,
                "bearish_setups": 1,
                "high_conviction": 2,
                "errors": 0,
            },
            "high_conviction_setups": [bull, bear],
            "top_bullish": [bull],
            "top_bearish": [bear],
        }

    def test_bearish_regime_flags_bullish_picks(self):
        regime = {"regime": "bearish", "spy_close": 550.0, "sma20": 600.0}
        annotated = annotate_regime_conflicts(self.make_scan(), regime)

        # Membership and order unchanged; only the counter-regime pick is flagged.
        assert [c["symbol"] for c in annotated["high_conviction_setups"]] == ["AAPL", "XOM"]
        aapl, xom = annotated["high_conviction_setups"]
        assert "bullish setup against a bearish tape" in aapl["regime_conflict"]
        assert "regime_conflict" not in xom
        # Annotation only: the summary count is untouched.
        assert annotated["summary"]["high_conviction"] == 2

    def test_bullish_regime_flags_bearish_picks(self):
        regime = {"regime": "bullish", "spy_close": 650.0, "sma20": 600.0}
        annotated = annotate_regime_conflicts(self.make_scan(), regime)

        aapl, xom = annotated["high_conviction_setups"]
        assert "regime_conflict" not in aapl
        assert "bearish setup against a bullish tape" in xom["regime_conflict"]

    def test_unknown_regime_annotates_nothing(self):
        regime = {"regime": "unknown", "reason": "SPY data unavailable"}
        scan = self.make_scan()
        annotated = annotate_regime_conflicts(scan, regime)

        assert annotated["high_conviction_setups"] == scan["high_conviction_setups"]
        assert all("regime_conflict" not in c for c in annotated["high_conviction_setups"])
        assert annotated["summary"]["high_conviction"] == 2

    def test_regime_verdict_attached_to_results(self):
        regime = {"regime": "bullish", "spy_close": 650.0, "sma20": 600.0}
        annotated = annotate_regime_conflicts(self.make_scan(), regime)
        assert annotated["market_regime"] is regime

    def test_input_not_mutated(self):
        scan = self.make_scan()
        original_hc = [dict(c) for c in scan["high_conviction_setups"]]
        original_summary = dict(scan["summary"])

        annotate_regime_conflicts(scan, {"regime": "bearish", "spy_close": 550.0, "sma20": 600.0})

        assert scan["high_conviction_setups"] == original_hc
        assert scan["summary"] == original_summary
        assert "regime_conflict" not in scan["high_conviction_setups"][0]
        assert "market_regime" not in scan

    def test_candidates_without_score_are_kept_unflagged(self):
        scan = self.make_scan()
        scan["high_conviction_setups"].append({"symbol": "MYSTERY"})
        annotated = annotate_regime_conflicts(
            scan, {"regime": "bearish", "spy_close": 550.0, "sma20": 600.0}
        )
        mystery = next(c for c in annotated["high_conviction_setups"] if c["symbol"] == "MYSTERY")
        assert "regime_conflict" not in mystery

    def test_handles_sparse_scan_results(self):
        annotated = annotate_regime_conflicts(
            {}, {"regime": "bullish", "spy_close": 650.0, "sma20": 600.0}
        )
        assert annotated["high_conviction_setups"] == []
        assert annotated["market_regime"]["regime"] == "bullish"

    def test_flag_reaches_top_list_duplicates(self):
        # run_scan shares candidate dicts between high_conviction_setups and the
        # top lists; the flag must be visible wherever the pick appears, without
        # mutating the originals.
        scan = self.make_scan()
        annotated = annotate_regime_conflicts(
            scan, {"regime": "bearish", "spy_close": 550.0, "sma20": 600.0}
        )
        (aapl_top,) = annotated["top_bullish"]
        assert "bullish setup against a bearish tape" in aapl_top["regime_conflict"]
        # XOM is regime-aligned: its top_bearish entry stays unflagged.
        assert "regime_conflict" not in annotated["top_bearish"][0]
        # Originals untouched.
        assert "regime_conflict" not in scan["top_bullish"][0]


class TestFormatRegimeHeader:
    def test_bullish_header_names_values(self):
        header = format_regime_header(
            {
                "regime": "bullish",
                "spy_close": 652.31,
                "sma20": 643.1,
                "close_vs_sma_pct": 1.43,
                "as_of": "2026-08-28",
            }
        )
        assert "BULLISH" in header
        assert "652.31" in header
        assert "643.10" in header
        assert "above" in header
        assert "2026-08-28" in header

    def test_bearish_header_says_below(self):
        header = format_regime_header(
            {
                "regime": "bearish",
                "spy_close": 550.0,
                "sma20": 600.0,
                "close_vs_sma_pct": -8.33,
                "as_of": "2026-08-28",
            }
        )
        assert "BEARISH" in header
        assert "below" in header

    def test_unknown_header_carries_reason(self):
        header = format_regime_header({"regime": "unknown", "reason": "SPY data unavailable"})
        assert "UNKNOWN" in header
        assert "SPY data unavailable" in header

    def test_conflict_count_is_mentioned(self):
        header = format_regime_header(
            {
                "regime": "bearish",
                "spy_close": 550.0,
                "sma20": 600.0,
                "close_vs_sma_pct": -8.33,
                "as_of": "2026-08-28",
            },
            conflict_count=2,
        )
        assert "2" in header
        assert "flagged" in header.lower()

    def test_no_conflict_note_when_zero(self):
        header = format_regime_header(
            {
                "regime": "bullish",
                "spy_close": 650.0,
                "sma20": 600.0,
                "close_vs_sma_pct": 8.33,
                "as_of": "2026-08-28",
            },
            conflict_count=0,
        )
        assert "flagged" not in header.lower()
