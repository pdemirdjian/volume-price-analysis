"""Tests for the deterministic pick block (PDE-69)."""

from volume_price_analysis.agent.picks import (
    CONVICTIONS,
    PICK_TABLE_HEADER,
    Pick,
    annotate_conviction,
    build_picks,
    conviction_for,
    render_picks_table,
)


def _scan(high=(), bull=(), bear=()):
    return {
        "high_conviction_setups": list(high),
        "top_bullish": list(bull),
        "top_bearish": list(bear),
    }


class TestConvictionFor:
    def test_high_conviction_gate_wins(self):
        assert conviction_for({"composite_score": 2.5}, high_conviction=True) == "HIGH"

    def test_strong_score_is_medium(self):
        assert conviction_for({"composite_score": -4.2}, high_conviction=False) == "MEDIUM"

    def test_high_signal_quality_is_medium(self):
        c = {"composite_score": 2.1, "signal_quality": "high"}
        assert conviction_for(c, high_conviction=False) == "MEDIUM"

    def test_marginal_is_low(self):
        c = {"composite_score": 2.1, "signal_quality": "medium"}
        assert conviction_for(c, high_conviction=False) == "LOW"

    def test_missing_fields_is_low(self):
        assert conviction_for({}, high_conviction=False) == "LOW"

    def test_always_one_of_the_fixed_vocabulary(self):
        for c in ({}, {"composite_score": 9}, {"signal_quality": "high"}):
            for high in (True, False):
                assert conviction_for(c, high_conviction=high) in CONVICTIONS


class TestAnnotateConviction:
    def test_labels_every_candidate_in_every_list(self):
        aapl = {"symbol": "AAPL", "composite_score": 5.0}
        scan = _scan(high=[aapl], bull=[aapl, {"symbol": "MSFT", "composite_score": 2.2}])
        out = annotate_conviction(scan)
        assert out["high_conviction_setups"][0]["conviction"] == "HIGH"
        assert out["top_bullish"][0]["conviction"] == "HIGH"
        assert out["top_bullish"][1]["conviction"] == "LOW"

    def test_does_not_mutate_input(self):
        aapl = {"symbol": "AAPL", "composite_score": 5.0}
        scan = _scan(high=[aapl], bull=[aapl])
        annotate_conviction(scan)
        assert "conviction" not in aapl

    def test_keeps_other_keys_and_tolerates_junk(self):
        scan = {**_scan(bull=["not-a-dict"]), "summary": {"x": 1}, "top_bearish": None}
        out = annotate_conviction(scan)
        assert out["summary"] == {"x": 1}
        assert out["top_bullish"] == ["not-a-dict"]
        assert out["top_bearish"] is None


class TestBuildPicks:
    def test_priority_order_and_dedupe(self):
        aapl = {"symbol": "AAPL", "composite_score": 5.0}
        msft = {"symbol": "MSFT", "composite_score": 2.5}
        tsla = {"symbol": "TSLA", "composite_score": -4.5}
        picks = build_picks(_scan(high=[aapl], bull=[msft, aapl], bear=[tsla]))
        assert [p.symbol for p in picks] == ["AAPL", "MSFT", "TSLA"]
        assert [p.conviction for p in picks] == ["HIGH", "LOW", "MEDIUM"]
        assert [p.direction for p in picks] == ["bullish", "bullish", "bearish"]

    def test_deep_analysis_supplies_price_and_earnings(self):
        scan = _scan(bull=[{"symbol": "AAPL", "composite_score": 3.0, "latest_price": 100.0}])
        deep = [
            {"symbol": "AAPL", "latest_price": 101.5, "earnings_warning": "EARNINGS in 3 day(s)"}
        ]
        (pick,) = build_picks(scan, deep)
        assert pick.price == 101.5
        assert pick.earnings_warning == "EARNINGS in 3 day(s)"

    def test_scan_price_when_no_deep_analysis(self):
        scan = _scan(bull=[{"symbol": "AAPL", "composite_score": 3.0, "latest_price": 100.0}])
        (pick,) = build_picks(scan)
        assert pick.price == 100.0
        assert pick.earnings_warning is None

    def test_regime_conflict_flag(self):
        scan = _scan(high=[{"symbol": "AAPL", "composite_score": 5.0, "regime_conflict": "x"}])
        assert build_picks(scan)[0].regime_conflict is True

    def test_skips_junk_entries(self):
        scan = _scan(bull=["junk", {"no": "symbol"}, {"symbol": "A", "composite_score": 2}])
        assert [p.symbol for p in build_picks(scan)] == ["A"]

    def test_empty_scan(self):
        assert build_picks({}) == []


class TestRenderPicksTable:
    def test_fixed_header_and_rows(self):
        picks = [
            Pick("AAPL", "bullish", "HIGH", 5.0, 101.5, True, "EARNINGS in 3 day(s)"),
            Pick("TSLA", "bearish", "LOW", -2.25),
        ]
        table = render_picks_table(picks).splitlines()
        assert table[0] == PICK_TABLE_HEADER
        assert table[1] == "|---|---|---|---|---|---|"
        assert table[2] == (
            "| AAPL | bullish | HIGH | +5.00 | 101.50 | counter-regime; EARNINGS in 3 day(s) |"
        )
        assert table[3] == "| TSLA | bearish | LOW | -2.25 | — | — |"

    def test_empty_still_emits_block(self):
        table = render_picks_table([]).splitlines()
        assert table[0] == PICK_TABLE_HEADER
        assert len(table) == 3
        assert "no candidates" in table[2]
