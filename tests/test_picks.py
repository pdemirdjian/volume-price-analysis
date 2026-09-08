"""Tests for the deterministic pick block (PDE-69)."""

from volume_price_analysis.agent.picks import (
    CONVICTIONS,
    PICK_TABLE_HEADER,
    Pick,
    build_picks,
    render_picks_table,
)
from volume_price_analysis.agent.regime import annotate_regime_conflicts


def _scan(high=(), bull=(), bear=()):
    return {
        "high_conviction_setups": list(high),
        "top_bullish": list(bull),
        "top_bearish": list(bear),
    }


def _conviction(candidate, *, high=False, regime=None):
    """Conviction of one candidate, read off the pick the builder derives."""
    named = {"symbol": "SYM", **candidate}
    scan = _scan(high=[named] if high else (), bull=[named])
    return build_picks(scan, regime=regime)[0].conviction


class TestConvictionRules:
    """The conviction vocabulary, exercised through its only call path."""

    def test_high_conviction_listing_wins(self):
        assert _conviction({"composite_score": 2.5}, high=True) == "HIGH"

    def test_strong_score_is_medium(self):
        assert _conviction({"composite_score": -4.2}) == "MEDIUM"

    def test_gate_reevaluated_from_fields(self):
        # Not in high_conviction_setups (the scan caps that list at five), but
        # the candidate itself clears the gate.
        assert _conviction({"composite_score": -4.2, "adx": 31.0, "iv_percentile": 22.0}) == "HIGH"

    def test_gate_fails_on_any_leg(self):
        base = {"composite_score": 4.5, "adx": 30.0, "hv_percentile": 40.0}
        assert _conviction(base) == "HIGH"
        assert _conviction({**base, "adx": 27.9}) == "MEDIUM"
        assert _conviction({**base, "hv_percentile": 50.1}) == "MEDIUM"
        assert _conviction({**base, "composite_score": 3.9, "adx": 40}) == "LOW"

    def test_non_numeric_score_is_low(self):
        assert _conviction({"composite_score": "n/a"}) == "LOW"

    def test_high_signal_quality_is_medium(self):
        assert _conviction({"composite_score": 2.1, "signal_quality": "high"}) == "MEDIUM"

    def test_marginal_is_low(self):
        assert _conviction({"composite_score": 2.1, "signal_quality": "medium"}) == "LOW"

    def test_missing_fields_is_low(self):
        assert _conviction({}) == "LOW"

    def test_always_one_of_the_fixed_vocabulary(self):
        for c in ({}, {"composite_score": 9}, {"signal_quality": "high"}):
            for high in (True, False):
                assert _conviction(c, high=high) in CONVICTIONS

    def test_sixth_qualifier_beyond_scan_cap_is_still_high(self):
        qualifiers = [
            {"symbol": f"S{i}", "composite_score": 5.0, "adx": 35.0, "iv_percentile": 10.0}
            for i in range(6)
        ]
        scan = _scan(high=qualifiers[:5], bull=qualifiers)
        assert [p.conviction for p in build_picks(scan)] == ["HIGH"] * 6

    def test_one_label_per_symbol_across_lists(self):
        aapl = {"symbol": "AAPL", "composite_score": 5.0}
        scan = _scan(high=[aapl], bull=[aapl, {"symbol": "MSFT", "composite_score": 2.2}])
        assert [(p.symbol, p.conviction) for p in build_picks(scan)] == [
            ("AAPL", "HIGH"),
            ("MSFT", "LOW"),
        ]

    def test_tolerates_junk_lists(self):
        scan = {**_scan(bull=["not-a-dict"]), "summary": {"x": 1}, "top_bearish": None}
        assert build_picks(scan) == []


class TestBuildPicks:
    def test_priority_order_and_dedupe(self):
        aapl = {"symbol": "AAPL", "composite_score": 5.0}
        msft = {"symbol": "MSFT", "composite_score": 2.5}
        tsla = {"symbol": "TSLA", "composite_score": -4.5}
        picks = build_picks(_scan(high=[aapl], bull=[msft, aapl], bear=[tsla]))
        assert [p.symbol for p in picks] == ["AAPL", "MSFT", "TSLA"]
        assert [p.conviction for p in picks] == ["HIGH", "LOW", "MEDIUM"]
        assert [p.direction for p in picks] == ["bullish", "bullish", "bearish"]

    def test_deep_analysis_supplies_price_score_and_earnings(self):
        scan = _scan(bull=[{"symbol": "AAPL", "composite_score": 3.0, "latest_price": 100.0}])
        deep = [
            {
                "symbol": "AAPL",
                "latest_price": 101.5,
                "composite_signal": {"score": 3.4567},
                "earnings_warning": "EARNINGS in 3 day(s)",
            }
        ]
        (pick,) = build_picks(scan, deep)
        assert pick.price == 101.5
        assert pick.score == 3.46
        assert pick.earnings_warning == "EARNINGS in 3 day(s)"

    def test_deep_analysis_without_score_keeps_scan_score(self):
        scan = _scan(bull=[{"symbol": "AAPL", "composite_score": 3.0}])
        (pick,) = build_picks(scan, [{"symbol": "AAPL", "composite_signal": "junk"}])
        assert pick.score == 3.0

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


class TestRegimeFoldedIntoConviction:
    """PDE-150: the builder takes the regime and derives conviction once."""

    def test_conflict_downgrades_conviction_with_no_prior_annotation(self):
        # Raw scan results — the orchestrator has NOT run any annotator first.
        aapl = {"symbol": "AAPL", "composite_score": 5.0, "adx": 35.0, "hv_percentile": 10.0}
        scan = _scan(high=[aapl], bull=[aapl])

        assert [p.conviction for p in build_picks(scan, regime={"regime": "bullish"})] == ["HIGH"]

        (pick,) = build_picks(scan, regime={"regime": "bearish"})
        assert pick.conviction == "MEDIUM"
        assert pick.regime_conflict is True

    def test_unknown_regime_annotates_nothing(self):
        aapl = {"symbol": "AAPL", "composite_score": 5.0, "adx": 35.0, "hv_percentile": 10.0}
        (pick,) = build_picks(_scan(high=[aapl]), regime={"regime": "unknown"})
        assert pick.conviction == "HIGH"
        assert pick.regime_conflict is False

    def test_does_not_mutate_the_caller_s_candidates(self):
        aapl = {"symbol": "AAPL", "composite_score": 5.0, "adx": 35.0, "hv_percentile": 10.0}
        build_picks(_scan(high=[aapl], bull=[aapl]), regime={"regime": "bearish"})
        assert "regime_conflict" not in aapl

    def test_qualifier_beyond_the_scan_cap_is_still_conflict_checked(self):
        # `annotate_regime_conflicts` only reaches the scan's five-entry
        # high_conviction_setups list; the gate is re-evaluated here uncapped,
        # so the sixth qualifier must not slip through as HIGH on a hostile tape.
        qualifiers = [
            {"symbol": f"S{i}", "composite_score": 5.0, "adx": 35.0, "iv_percentile": 10.0}
            for i in range(6)
        ]
        scan = _scan(high=qualifiers[:5], bull=qualifiers)
        picks = build_picks(scan, regime={"regime": "bearish"})
        assert [p.conviction for p in picks] == ["MEDIUM"] * 6
        assert all(p.regime_conflict for p in picks)

    def test_non_qualifying_counter_trend_pick_is_not_flagged(self):
        # Only gate qualifiers are conflict-checked; an ordinary counter-trend
        # candidate keeps its unflagged row, as before.
        scan = _scan(bull=[{"symbol": "MSFT", "composite_score": 2.2}])
        (pick,) = build_picks(scan, regime={"regime": "bearish"})
        assert pick.regime_conflict is False
        assert pick.conviction == "LOW"

    def test_already_annotated_scan_is_unchanged_by_reannotation(self):
        # Ordering is not load-bearing: annotating twice yields the same picks.
        aapl = {"symbol": "AAPL", "composite_score": 5.0, "adx": 35.0, "hv_percentile": 10.0}
        regime = {"regime": "bearish"}
        pre = annotate_regime_conflicts(_scan(high=[aapl], bull=[aapl]), regime)
        assert build_picks(pre, regime=regime) == build_picks(
            _scan(high=[aapl], bull=[aapl]), regime=regime
        )


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
