"""Tests for the Briefing module: the one owner of the delivered document."""

import dataclasses
from datetime import date
from pathlib import Path

import pytest

from volume_price_analysis.agent.briefing import BriefingInputs, render, render_raw

_GOLDEN = Path(__file__).parent / "golden" / "briefing_body.md"

_BULLISH_REGIME = {
    "regime": "bullish",
    "spy_close": 500.0,
    "sma20": 494.07,
    "close_vs_sma_pct": 1.2,
    "as_of": "2026-09-03",
}


def golden_inputs(**overrides) -> BriefingInputs:
    """The inputs that render tests/golden/briefing_body.md."""
    scan = {
        "scan_parameters": {"symbols_scanned": 540},
        "summary": {"total_candidates": 3},
        "high_conviction_setups": [
            {"symbol": "AAPL", "composite_score": 5.2, "latest_price": 190.1, "regime_conflict": ""}
        ],
        "top_bullish": [
            {"symbol": "AAPL", "composite_score": 5.2, "latest_price": 190.1},
            {"symbol": "MSFT", "composite_score": 2.4, "signal_quality": "medium"},
        ],
        "top_bearish": [
            {
                "symbol": "TSLA",
                "composite_score": -4.1,
                "latest_price": 240.55,
                "regime_conflict": "bearish setup against a bullish tape",
            }
        ],
    }
    deep = [
        {"symbol": "AAPL", "latest_price": 190.25, "earnings_warning": "EARNINGS in 5 day(s)"},
    ]
    inputs = BriefingInputs(
        scan_results=scan,
        deep_analyses=deep,
        regime=_BULLISH_REGIME,
        briefing_date=date(2026, 9, 4),
        elapsed_s=42.0,
        model_text=(
            "## Executive Summary\n\nTape is constructive.\n\n"
            "## Top Picks\n\n- **AAPL** @ $190.25 | Score +5.2 | bullish | Conviction: MEDIUM\n"
        ),
    )
    return dataclasses.replace(inputs, **overrides)


class TestRender:
    def test_matches_golden_file(self):
        # Normalise CRLF so a Windows checkout with autocrlf still compares.
        expected = _GOLDEN.read_text(encoding="utf-8").replace("\r\n", "\n")
        assert render(golden_inputs()) == expected, (
            "Email body layout changed. If intentional, regenerate tests/golden/"
            "briefing_body.md from render(golden_inputs()) and review the diff."
        )

    def test_inputs_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            golden_inputs().model_text = "x"  # type: ignore[misc]


class TestRenderFallback:
    """model_text None: the same template wraps a programmatic body."""

    def _body(self) -> str:
        return render(
            golden_inputs(
                model_text=None,
                scan_results={
                    "summary": {
                        "total_candidates": 2,
                        "bullish_setups": 1,
                        "bearish_setups": 1,
                        "high_conviction": 1,
                    },
                    "high_conviction_setups": [{"symbol": "AAPL", "composite_score": 5.2}],
                    "top_bullish": [{"symbol": "AAPL", "composite_score": 5.2}],
                    "top_bearish": [{"symbol": "TSLA", "composite_score": -2.0}],
                },
                deep_analyses=[
                    {
                        "symbol": "AAPL",
                        "latest_price": 150.0,
                        "composite_signal": {"score": 5.5, "recommendation": "strong_bullish"},
                    },
                    {
                        "symbol": "TSLA",
                        "latest_price": 240.5,
                        "composite_signal": {"score": -2.0, "recommendation": "bearish"},
                    },
                ],
                regime={
                    "regime": "bearish",
                    "spy_close": 550.0,
                    "sma20": 560.0,
                    "close_vs_sma_pct": -1.8,
                    "as_of": "2026-09-03",
                },
            )
        )

    def test_fallback_body_sits_between_table_and_footer(self):
        body = self._body()
        expected = (
            "## Fallback Briefing (AI unavailable)\n\n"
            "**Candidates found:** 2\n"
            "**Bullish setups:** 1\n"
            "**Bearish setups:** 1\n"
            "**High conviction:** 1\n\n"
            "## Top Candidates\n\n"
            "- **AAPL** @ $150.00 | Score: 5.5 | strong_bullish | ⚠️ counter-regime setup\n"
            "- **TSLA** @ $240.50 | Score: -2.0 | bearish"
            "\n\n---\n"
        )
        assert expected in body
        assert body.index("## Pick Summary") < body.index("## Fallback Briefing")

    def test_one_h1_and_the_dated_title(self):
        body = self._body()
        assert body.startswith("# Morning Market Briefing — Friday, September 4, 2026\n\n")
        assert sum(line.startswith("# ") for line in body.splitlines()) == 1

    def test_header_counts_the_flagged_pick(self):
        body = self._body()
        assert body.splitlines()[2].startswith("**Market Regime: BEARISH**")
        assert "1 high-conviction pick flagged as counter-regime." in body


class TestRenderFooter:
    def test_reports_scanned_and_found_separately(self):
        body = render(golden_inputs())
        assert "540 symbols scanned | 3 candidates found | 1 deep analyses*" in body

    def test_omits_scanned_count_when_absent(self):
        body = render(golden_inputs(scan_results={"summary": {"total_candidates": 3}}))
        assert "symbols scanned" not in body
        assert "*Generated in 42.0s | 3 candidates found | 1 deep analyses*" in body


class TestRenderUnknownRegime:
    def test_header_says_unknown_and_nothing_is_flagged(self):
        body = render(golden_inputs(regime={"regime": "unknown", "reason": "regime check failed"}))
        assert body.splitlines()[2] == (
            "**Market Regime: UNKNOWN** — regime check unavailable (regime check failed)."
        )
        assert "| TSLA | bearish | MEDIUM | -4.10 | 240.55 | counter-regime |" in body


class TestRenderRaw:
    """The no-AI body: regime header, then the raw scan and deep-analysis JSON."""

    def test_exact_body(self):
        inputs = golden_inputs(
            model_text=None,
            scan_results={"summary": {"total_candidates": 1}, "top_bullish": []},
            deep_analyses=[{"symbol": "AAPL", "score": 4.5}, {"score": 2.0}],
            regime={"regime": "unknown", "reason": "SPY data unavailable"},
        )
        expected = (
            "**Market Regime: UNKNOWN** — regime check unavailable (SPY data unavailable).\n\n"
            "# Morning Market Scan Results\n\n"
            "```json\n"
            "{\n"
            '  "summary": {\n'
            '    "total_candidates": 1\n'
            "  },\n"
            '  "top_bullish": [],\n'
            '  "market_regime": {\n'
            '    "regime": "unknown",\n'
            '    "reason": "SPY data unavailable"\n'
            "  },\n"
            '  "high_conviction_setups": []\n'
            "}\n"
            "```\n\n"
            "# Deep Analysis Results\n\n"
            "## AAPL\n"
            "```json\n"
            "{\n"
            '  "symbol": "AAPL",\n'
            '  "score": 4.5\n'
            "}\n"
            "```\n\n"
            "## Unknown\n"
            "```json\n"
            "{\n"
            '  "score": 2.0\n'
            "}\n"
            "```\n\n"
        )
        assert render_raw(inputs) == expected

    def test_no_deep_section_without_analyses(self):
        body = render_raw(golden_inputs(deep_analyses=[]))
        assert "Deep Analysis Results" not in body
        assert body.startswith("**Market Regime: BULLISH**")
