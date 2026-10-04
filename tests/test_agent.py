"""Tests for the morning briefing agent."""

import email
import json
import logging
import smtplib
import ssl
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from volume_price_analysis.agent.ai_client import (
    _TRUNCATION_WARNING,
    MAX_OUTPUT_TOKENS,
    PROVIDERS,
    SYSTEM_PROMPT,
    BriefingResult,
    _drop_superseded_scan_fields,
    _project_deep_analysis,
    _project_scan_results,
    build_briefing_prompt,
    find_ungrounded_tickers,
    format_briefing_date,
    generate_anthropic,
    generate_briefing,
    generate_gemini,
    is_transient_ai_error,
    resolve_model,
    strip_date_placeholders,
)
from volume_price_analysis.agent.config import MAX_DEEP_ANALYSIS_CAP, AgentConfig
from volume_price_analysis.agent.email_sender import (
    SmtpCreds,
    _linkify_tickers,
    _parse_recipients,
    build_briefing_message,
    build_error_message,
    send_email,
)
from volume_price_analysis.agent.morning_agent import (
    _HOLDING_PERIOD_DAYS,
    BriefingRunResult,
    _candidate_symbols,
    _check_earnings,
    _config_errors,
    _fetch_earnings_warnings,
    _get_top_symbols,
    build_earnings_preamble,
    main,
    run_morning_briefing,
)
from volume_price_analysis.agent.picks import build_picks
from volume_price_analysis.data_fetcher import InMemoryDataSource

# A minimal frame standing in for fetched history. These tests mock
# run_options_analysis, so only the fetch succeeding matters.
_STUB_FRAME = pd.DataFrame(
    {
        "Date": pd.date_range("2026-01-01", periods=5),
        "Open": [100.0] * 5,
        "High": [101.0] * 5,
        "Low": [99.0] * 5,
        "Close": [100.0] * 5,
        "Volume": [1_000_000] * 5,
    }
)


def _briefing_result(degraded=False, reason=None):
    """Stand-in for run_morning_briefing's return value in main() tests."""
    return BriefingRunResult(
        degraded=degraded,
        reason=reason,
        regime={"regime": "bullish"},
        symbols_analyzed=["AAPL"],
        email_sent=True,
    )


def agent_source(symbols=(), *, spy=None, earnings=None, errors=None):
    """Build the DataSource run_morning_briefing should use.

    Injecting one keeps the whole pipeline — scan, SPY regime fetch, per-symbol
    fetches, earnings guard — off the network without patching import paths.
    Symbols absent from ``frames`` fail exactly as they would in production.
    """
    frames = dict.fromkeys(symbols, _STUB_FRAME)
    if spy is not None:
        frames["SPY"] = spy
    return InMemoryDataSource(frames=frames, earnings=earnings, errors=errors)


class TestAgentConfig:
    """Test configuration loading and validation."""

    def test_from_env_loads_required_vars(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        monkeypatch.setenv("AI_PROVIDER_API_KEY", "test-key")
        monkeypatch.setenv("EMAIL_FROM", "test@example.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "test-password")
        monkeypatch.setenv("EMAIL_TO", "recipient@example.com")

        config = AgentConfig.from_env()
        assert config.ai_provider == "gemini"
        assert config.ai_provider_api_key == "test-key"
        assert config.email_from == "test@example.com"
        assert config.email_password == "test-password"
        assert config.email_to == "recipient@example.com"

    def test_from_env_anthropic_provider(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        monkeypatch.setenv("AI_PROVIDER_API_KEY", "sk-test-key")

        config = AgentConfig.from_env()
        assert config.ai_provider == "anthropic"
        assert config.ai_provider_api_key == "sk-test-key"

    def test_from_env_uses_defaults(self, monkeypatch):
        for key in [
            "AI_PROVIDER",
            "AI_PROVIDER_API_KEY",
            "EMAIL_FROM",
            "EMAIL_PASSWORD",
            "EMAIL_TO",
            "EMAIL_SMTP_HOST",
            "SCAN_UNIVERSE",
            "MAX_DEEP_ANALYSIS",
            "AI_MODEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = AgentConfig.from_env()
        assert config.ai_provider == "gemini"
        assert config.email_smtp_host == "smtp.gmail.com"
        assert config.email_smtp_port == 587
        assert config.scan_universe == "full_market"
        assert config.max_deep_analysis == 5
        assert config.ai_model == ""

    def test_from_env_overrides_defaults(self, monkeypatch):
        monkeypatch.setenv("SCAN_UNIVERSE", "tech")
        monkeypatch.setenv("MAX_DEEP_ANALYSIS", "10")
        monkeypatch.setenv("EMAIL_SMTP_PORT", "465")

        config = AgentConfig.from_env()
        assert config.scan_universe == "tech"
        assert config.max_deep_analysis == 10
        assert config.email_smtp_port == 465

    def test_from_env_handles_invalid_int(self, monkeypatch):
        monkeypatch.setenv("EMAIL_SMTP_PORT", "not_a_number")
        monkeypatch.setenv("MAX_DEEP_ANALYSIS", "abc")

        config = AgentConfig.from_env()
        assert config.email_smtp_port == 587
        assert config.max_deep_analysis == 5

    def test_from_env_rejects_nonpositive_max_deep(self, monkeypatch):
        monkeypatch.setenv("MAX_DEEP_ANALYSIS", "0")
        assert AgentConfig.from_env().max_deep_analysis == 5

        monkeypatch.setenv("MAX_DEEP_ANALYSIS", "-3")
        assert AgentConfig.from_env().max_deep_analysis == 5

    def test_from_env_clamps_oversized_max_deep(self, monkeypatch):
        monkeypatch.setenv("MAX_DEEP_ANALYSIS", "1000")
        assert AgentConfig.from_env().max_deep_analysis == MAX_DEEP_ANALYSIS_CAP

    def test_validate_missing_api_key(self):
        config = AgentConfig(
            ai_provider="gemini",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        errors = config.validate()
        assert len(errors) == 1
        assert "AI_PROVIDER_API_KEY" in errors[0]

    def test_validate_invalid_provider(self):
        config = AgentConfig(
            ai_provider="openai",
            ai_provider_api_key="key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        errors = config.validate()
        assert any("AI_PROVIDER" in e for e in errors)

    def test_validate_missing_email(self):
        config = AgentConfig(ai_provider="gemini", ai_provider_api_key="key")
        errors = config.validate()
        assert any("EMAIL_FROM" in e for e in errors)
        assert any("EMAIL_PASSWORD" in e for e in errors)
        assert any("EMAIL_TO" in e for e in errors)

    def test_validate_all_present(self):
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_rejects_out_of_range_max_deep(self):
        base = {
            "ai_provider": "gemini",
            "ai_provider_api_key": "test-key",
            "email_from": "a@b.com",
            "email_password": "pass",
            "email_to": "c@d.com",
        }
        errors = AgentConfig(**base, max_deep_analysis=0).validate()
        assert any("max_deep_analysis" in e for e in errors)

        errors = AgentConfig(**base, max_deep_analysis=MAX_DEEP_ANALYSIS_CAP + 1).validate()
        assert any("max_deep_analysis" in e for e in errors)

        assert AgentConfig(**base, max_deep_analysis=MAX_DEEP_ANALYSIS_CAP).validate() == []

    def test_validate_all_present_anthropic(self):
        config = AgentConfig(
            ai_provider="anthropic",
            ai_provider_api_key="sk-test",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        errors = config.validate()
        assert len(errors) == 0


class TestGetTopSymbols:
    """Test symbol extraction from scan results."""

    def test_prioritizes_high_conviction(self):
        scan_results = {
            "high_conviction_setups": [
                {"symbol": "NVDA"},
                {"symbol": "AAPL"},
            ],
            "top_bullish": [
                {"symbol": "MSFT"},
                {"symbol": "NVDA"},  # duplicate
            ],
            "top_bearish": [
                {"symbol": "TSLA"},
            ],
        }
        symbols = _get_top_symbols(scan_results, 3)
        assert symbols == ["NVDA", "AAPL", "MSFT"]

    def test_deduplicates(self):
        scan_results = {
            "high_conviction_setups": [{"symbol": "AAPL"}],
            "top_bullish": [{"symbol": "AAPL"}, {"symbol": "MSFT"}],
            "top_bearish": [{"symbol": "MSFT"}, {"symbol": "TSLA"}],
        }
        symbols = _get_top_symbols(scan_results, 10)
        assert len(symbols) == len(set(symbols))

    def test_respects_max_count(self):
        scan_results = {
            "high_conviction_setups": [{"symbol": f"SYM{i}"} for i in range(10)],
            "top_bullish": [],
            "top_bearish": [],
        }
        symbols = _get_top_symbols(scan_results, 3)
        assert len(symbols) == 3

    def test_handles_empty_results(self):
        scan_results = {
            "high_conviction_setups": [],
            "top_bullish": [],
            "top_bearish": [],
        }
        symbols = _get_top_symbols(scan_results, 5)
        assert symbols == []


class TestCandidateSymbols:
    """Test symbol collection for ticker linkification."""

    def test_collects_from_all_lists(self):
        scan_results = {
            "high_conviction_setups": [{"symbol": "NVDA"}],
            "top_bullish": [{"symbol": "AAPL"}, {"symbol": "NVDA"}],
            "top_bearish": [{"symbol": "TSLA"}],
        }
        deep_analyses = [{"symbol": "MSFT"}, {"score": 1.0}]
        assert _candidate_symbols(scan_results, deep_analyses) == {
            "NVDA",
            "AAPL",
            "TSLA",
            "MSFT",
        }

    def test_handles_empty_results(self):
        assert _candidate_symbols({}, []) == set()


def _full_deep_analysis():
    """A representative run_options_analysis-shaped dict for projection tests."""
    return {
        "symbol": "AAPL",
        "analysis_type": "Options Trading (14-Day Optimized)",
        "period": "2024-01-01 to 2024-03-01",
        "latest_price": 150.25,
        "headline": {
            "recommendation": "bullish",
            "composite_score": 4.2,
            "signal_quality": "high",
            "rationale": (
                "Bullish (score +4.2/10, high signal quality): price vs VWAP aligned bullish."
            ),
        },
        "parameters": {"holding_period": 14, "mfi_period": 7, "volume_window": 10},
        "composite_signal": {
            "score": 4.2,
            "recommendation": "bullish",
            "action": "Consider call options or call spreads",
            "signal_quality": "high",
            "quality_note": "Strong trend supports directional trades",
            "score_breakdown": {"price_vs_vwap": 2, "obv_momentum": 2, "rsi": 1},
        },
        "trend_analysis": {
            "adx": {
                "value": 31.5,
                "plus_di": 28.0,
                "minus_di": 12.0,
                "trend_strength": "strong",
                "trend_direction": "bullish",
                "adx_slope": "rising",
                "interpretation": "Strong bullish trend",
            },
            "rsi": {
                "value": 58.0,
                "condition": "neutral",
                "divergence_type": "none",
                "divergence_signal": "none",
                "interpretation": "Momentum neutral",
            },
        },
        "volume_indicators": {
            "obv": {"value": 123456789.0, "trend": "increasing", "short_term_momentum": "bullish"},
            "accumulation_distribution": {
                "value": 98765.0,
                "trend": "increasing",
                "signal": "institutional_buying",
            },
            "vpt": {"value": 4567.0, "trend": "increasing", "volume_conviction": "strong"},
            "mfi": {"value": 62.0, "condition": "Neutral", "options_signal": "neutral"},
            "cmf": {"value": 0.15, "signal": "neutral"},
            "relative_volume": {"current_rvol": 1.4, "significance": "elevated"},
            "volume_breakout": {"is_breakout": False, "direction": "none"},
        },
        "volatility_analysis": {
            "iv_percentile_proxy": {
                "percentile": 45.0,
                "hv_percentile": 45.0,
                "basis": "historical_volatility",
                "is_proxy": True,
                "current_hv": 0.22,
                "hv_range": "15.0% - 30.0%",
                "interpretation": "mid",
                "options_implication": "fairly priced",
                "strategy_suggestion": "debit spreads",
            },
            "expected_move": {
                "dollars": 7.5,
                "percent": 5.0,
                "upper_target": 157.75,
                "lower_target": 142.75,
                "targets": {},
                "strike_guidance": "near ATM",
                "interpretation": "moderate move",
            },
            "atr": {
                "value": 3.2,
                "daily_range": "±$3.20",
                "stop_loss_suggestion": "$143.85 to $145.45",
            },
            "bollinger_bands": {
                "upper": 158.0,
                "middle": 150.0,
                "lower": 142.0,
                "percent_b": 0.6,
                "bandwidth": 0.1,
                "squeeze_detected": False,
                "position": "neutral",
            },
        },
        "volume_profile": {
            "point_of_control": 149.0,
            "value_area_high": 153.0,
            "value_area_low": 145.0,
            "current_position": "in_value_area",
            "interpretation": "balanced",
            "strike_selection_guidance": {"poc_strike": "$149.00 - Highest probability"},
        },
        "time_decay": {"days_to_expiration": 14, "theta_risk": "moderate", "theta_note": "monitor"},
        "volume_trends": {"divergence_detected": False, "divergence_type": "none"},
        "options_insights": [
            "BULLISH: Composite score 4.2/10 - Consider call options or bull spreads",
            "STRONG TREND: ADX at 31.5 (bullish)",
        ],
    }


class TestProjectScanResults:
    """Test the curated scan-results projection (O3)."""

    def test_keeps_curated_keys(self):
        scan = {
            "scan_parameters": {"universe": "full_market"},
            "summary": {"total_candidates": 2},
            "high_conviction_setups": [{"symbol": "NVDA"}],
            "top_bullish": [{"symbol": "AAPL"}],
            "top_bearish": [{"symbol": "TSLA"}],
        }
        projected = _project_scan_results(scan)
        assert set(projected) == {
            "scan_parameters",
            "summary",
            "high_conviction_setups",
            "top_bullish",
            "top_bearish",
        }

    def test_drops_raw_error_list(self):
        scan = {
            "summary": {"total_candidates": 1, "errors": 3},
            "errors": [{"symbol": "BADX", "error": "delisted traceback noise"}],
        }
        projected = _project_scan_results(scan)
        # The verbose per-symbol error list is noise; the count stays in summary.
        assert "errors" not in projected
        assert projected["summary"]["errors"] == 3

    def test_omits_absent_keys(self):
        assert _project_scan_results({"summary": {}}) == {"summary": {}}

    def test_keeps_regime_fields(self):
        scan = {
            "summary": {"total_candidates": 1},
            "market_regime": {"regime": "bearish", "spy_close": 550.0},
            "high_conviction_setups": [
                {"symbol": "AAPL", "regime_conflict": "bullish setup against a bearish tape"}
            ],
        }
        projected = _project_scan_results(scan)
        assert projected["market_regime"]["regime"] == "bearish"
        assert projected["high_conviction_setups"][0]["regime_conflict"]


class TestDropSupersededScanFields:
    """Scan-level targets must be dropped for symbols that have a deep analysis."""

    def _scan(self):
        return {
            "summary": {"total_candidates": 2},
            "top_bullish": [
                {
                    "symbol": "BSX",
                    "composite_score": -4.0,
                    "expected_move_pct": 6.4,
                    "key_levels": {"upper_target": 45.8, "lower_target": 40.32},
                },
                {
                    "symbol": "AAPL",
                    "composite_score": 3.0,
                    "expected_move_pct": 4.1,
                    "key_levels": {"upper_target": 160.0, "lower_target": 148.0},
                },
            ],
        }

    def test_strips_targets_for_deep_analyzed_symbols(self):
        result = _drop_superseded_scan_fields(self._scan(), [{"symbol": "BSX"}])
        bsx, aapl = result["top_bullish"]
        # BSX has a deep analysis: its scan targets would conflict, so drop them.
        assert "key_levels" not in bsx
        assert "expected_move_pct" not in bsx
        assert bsx["composite_score"] == -4.0
        # AAPL has no deep analysis: its scan targets are the only ones, keep them.
        assert aapl["key_levels"]["lower_target"] == 148.0
        assert aapl["expected_move_pct"] == 4.1

    def test_does_not_mutate_input(self):
        scan = self._scan()
        _drop_superseded_scan_fields(scan, [{"symbol": "BSX"}])
        assert "key_levels" in scan["top_bullish"][0]

    def test_no_deep_analyses_returns_unchanged(self):
        scan = self._scan()
        assert _drop_superseded_scan_fields(scan, []) is scan

    def test_handles_malformed_entries(self):
        scan = {"top_bullish": "not-a-list", "top_bearish": [None, {"symbol": "BSX"}]}
        result = _drop_superseded_scan_fields(scan, [{"symbol": "BSX"}, "junk", {}])
        assert result["top_bullish"] == "not-a-list"
        assert result["top_bearish"][0] is None


class TestProjectDeepAnalysis:
    """Test the curated deep-analysis projection (O3)."""

    def test_keeps_high_signal_sections(self):
        projected = _project_deep_analysis(_full_deep_analysis())
        assert projected["symbol"] == "AAPL"
        assert projected["latest_price"] == 150.25
        for key in (
            "headline",
            "composite",
            "trend",
            "volatility",
            "key_levels",
            "volume_signals",
            "insights",
        ):
            assert key in projected

    def test_drops_internal_noise(self):
        projected = _project_deep_analysis(_full_deep_analysis())
        # Internal scoring detail and tuning params are not briefing-relevant.
        assert "parameters" not in projected
        assert "score_breakdown" not in projected.get("composite", {})
        # Raw indicator magnitudes (OBV/AD absolute values) are dropped.
        flat = json.dumps(projected)
        assert "123456789" not in flat
        assert "plus_di" not in flat

    def test_carries_hv_honest_levels(self):
        projected = _project_deep_analysis(_full_deep_analysis())
        # Key tradeable levels survive: support/resistance and HV percentile.
        assert projected["key_levels"]["point_of_control"] == 149.0
        assert projected["volatility"]["hv_percentile"] == 45.0
        assert projected["volatility"]["upper_target"] == 157.75

    def test_handles_minimal_dict(self):
        # Sparse dicts (e.g. fallback paths/tests) must not raise.
        assert _project_deep_analysis({"symbol": "MSFT"}) == {"symbol": "MSFT"}

    def test_handles_missing_symbol(self):
        projected = _project_deep_analysis({"latest_price": 10.0})
        assert projected["symbol"] == "Unknown"

    def test_handles_malformed_nested_values(self):
        # A section present but holding non-dict (scalar) values must degrade
        # gracefully, not raise AttributeError.
        malformed = {
            "symbol": "X",
            "trend_analysis": {"adx": 14.5, "rsi": "n/a"},
            "volatility_analysis": "unexpected",
            "volume_indicators": {"obv": 123, "cmf": None},
        }
        projected = _project_deep_analysis(malformed)
        assert projected["symbol"] == "X"
        assert projected["trend"]["adx"] is None
        assert projected["volume_signals"]["obv_trend"] is None


class TestBuildBriefingPrompt:
    """Test that _build_user_message emits the curated projection (O3)."""

    def test_curated_message_is_smaller_than_raw_dump(self, sample_stock_data):
        from volume_price_analysis.analysis import run_options_analysis

        analysis = run_options_analysis("TEST", sample_stock_data)
        scan = {
            "scan_parameters": {"universe": "full_market"},
            "summary": {"total_candidates": 1, "high_conviction": 1, "errors": 0},
            "high_conviction_setups": [{"symbol": "TEST", "composite_score": 4.0}],
            "top_bullish": [{"symbol": "TEST", "composite_score": 4.0}],
            "top_bearish": [],
            "errors": [],
        }
        curated = build_briefing_prompt(scan, [analysis])
        raw_dump = f"{json.dumps(scan, default=str)}{json.dumps(analysis, default=str)}"
        # The whole point of O3: meaningfully less text than the raw JSON dump.
        assert len(curated) < len(raw_dump)
        # But the essential symbol and headline call still survive.
        assert "TEST" in curated
        assert analysis["headline"]["recommendation"] in curated

    def test_excludes_score_breakdown_noise(self, sample_stock_data):
        from volume_price_analysis.analysis import run_options_analysis

        analysis = run_options_analysis("TEST", sample_stock_data)
        curated = build_briefing_prompt({"summary": {}}, [analysis])
        # Internal scoring detail is dropped from the model-facing prompt.
        assert "score_breakdown" not in curated
        assert "plus_di" not in curated

    def test_deep_analyzed_symbol_gets_single_target_source(self):
        # A symbol with a deep analysis must not also carry scan-level targets:
        # the two are computed from different data windows and the model would
        # cite both (e.g. "$40.32 scan target" vs "$39.75 lower target").
        scan = {
            "summary": {"total_candidates": 1},
            "top_bearish": [
                {
                    "symbol": "BSX",
                    "expected_move_pct": 6.4,
                    "key_levels": {"upper_target": 45.8, "lower_target": 40.32},
                }
            ],
        }
        deep = {
            "symbol": "BSX",
            "volatility_analysis": {
                "expected_move": {"upper_target": 46.37, "lower_target": 39.75},
            },
        }
        curated = build_briefing_prompt(scan, [deep])
        assert "40.32" not in curated
        assert "39.75" in curated


class _FakeProvider:
    """Records the arguments generate_briefing hands a provider."""

    def __init__(self, text="# Morning Briefing\nTest content"):
        self.text = text
        self.calls = []

    def __call__(self, user_content, model, api_key):
        self.calls.append((user_content, model, api_key))
        return self.text

    @property
    def user_content(self):
        return self.calls[-1][0]


class TestGenerateBriefing:
    """generate_briefing orchestrates prompt + provider + grounding check."""

    def test_returns_provider_text(self):
        provider = _FakeProvider()

        result = generate_briefing(
            scan_results={"summary": {"total_candidates": 5}},
            deep_analyses=[],
            provider=provider,
            model="test-model",
            api_key="test-key",
        )

        assert isinstance(result, BriefingResult)
        assert "Morning Briefing" in result.text
        assert result.ungrounded_tickers == []

    def test_passes_model_and_key_through(self):
        provider = _FakeProvider()

        generate_briefing(
            scan_results={},
            deep_analyses=[],
            provider=provider,
            model="some-model",
            api_key="sk-test",
        )

        _, model, api_key = provider.calls[0]
        assert model == "some-model"
        assert api_key == "sk-test"

    def test_includes_scan_data_in_prompt(self):
        provider = _FakeProvider(text="briefing")

        generate_briefing(
            scan_results={
                "summary": {"total_candidates": 1, "high_conviction": 1},
                "high_conviction_setups": [{"symbol": "NVDA", "composite_score": 6.2}],
            },
            deep_analyses=[{"symbol": "AAPL"}],
            provider=provider,
            model="m",
            api_key="k",
        )

        # Curated high-signal data reaches the model: candidate + deep symbols.
        assert "AAPL" in provider.user_content
        assert "NVDA" in provider.user_content
        assert "total_candidates" in provider.user_content

    def test_includes_earnings_preamble(self):
        provider = _FakeProvider(text="briefing")

        generate_briefing(
            scan_results={},
            deep_analyses=[],
            provider=provider,
            model="m",
            api_key="k",
            earnings_preamble="**EARNINGS EVENT RISK** — NVDA reports tomorrow.",
        )

        assert "EARNINGS EVENT RISK" in provider.user_content

    def test_provider_error_propagates(self):
        def boom(user_content, model, api_key):
            raise RuntimeError("provider down")

        with pytest.raises(RuntimeError, match="provider down"):
            generate_briefing(
                scan_results={},
                deep_analyses=[],
                provider=boom,
                model="m",
                api_key="k",
            )


class TestProviderRegistry:
    """PROVIDERS maps AI_PROVIDER names onto the production adapters."""

    def test_registry_contents(self):
        assert PROVIDERS["anthropic"] is generate_anthropic
        assert PROVIDERS["gemini"] is generate_gemini

    def test_registry_covers_every_valid_config_provider(self):
        # config.AgentConfig.validate() is the single validator of names.
        assert set(PROVIDERS) == {"gemini", "anthropic"}

    def test_resolve_model_defaults_per_provider(self):
        assert resolve_model("gemini") == "gemini-2.5-pro"
        assert resolve_model("anthropic") == "claude-sonnet-4-6"

    def test_resolve_model_honours_explicit_override(self):
        assert resolve_model("gemini", "gemini-flash") == "gemini-flash"


class TestGenerateAnthropicAdapter:
    """SDK-level tests for the Anthropic adapter itself."""

    @staticmethod
    def _mock_client(mocker, text="# Morning Briefing", stop_reason="end_turn"):
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [SimpleNamespace(type="text", text=text)]
        mock_message.usage.input_tokens = 100
        mock_message.usage.output_tokens = 200
        mock_message.stop_reason = stop_reason
        mock_client.messages.create.return_value = mock_message
        mocker.patch("anthropic.Anthropic", return_value=mock_client)
        return mock_client

    @pytest.mark.parametrize("has_thinking", [False, True])
    def test_returns_first_text_block(self, mocker, has_thinking):
        client = self._mock_client(mocker)
        blocks = [
            SimpleNamespace(type="text", text="First briefing"),
            SimpleNamespace(type="text", text="Second briefing"),
        ]
        if has_thinking:
            blocks.insert(0, SimpleNamespace(type="thinking", thinking="Reasoning"))
        client.messages.create.return_value.content = blocks

        assert generate_anthropic("prompt", "model", "key") == "First briefing"

    @pytest.mark.parametrize(
        "blocks", [[], [SimpleNamespace(type="thinking", thinking="Reasoning")]]
    )
    def test_missing_text_is_non_transient_failure(self, mocker, blocks):
        client = self._mock_client(mocker)
        client.messages.create.return_value.content = blocks

        with pytest.raises(ValueError, match="no text block") as raised:
            generate_anthropic("prompt", "model", "key")

        assert not is_transient_ai_error(raised.value)
        client.messages.create.assert_called_once()

    def test_request_has_finite_timeout(self, mocker):
        mock_client = self._mock_client(mocker)
        with patch("anthropic.Anthropic", return_value=mock_client) as client:
            generate_anthropic("user content", "claude-test", "sk-test")
        client.assert_called_once_with(api_key="sk-test", timeout=120, max_retries=0)

    def test_calls_messages_api(self, mocker):
        mock_client = self._mock_client(mocker)

        result = generate_anthropic("user content", "claude-test", "sk-test")

        assert "Morning Briefing" in result
        assert _TRUNCATION_WARNING not in result
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-test"
        assert kwargs["system"] == SYSTEM_PROMPT
        assert kwargs["max_tokens"] == MAX_OUTPUT_TOKENS
        assert kwargs["messages"][0]["content"] == "user content"

    def test_appends_warning_on_truncation(self, mocker):
        self._mock_client(mocker, text="# Truncated", stop_reason="max_tokens")

        result = generate_anthropic("user content", "claude-test", "sk-test")

        assert _TRUNCATION_WARNING in result

    def test_max_tokens_override(self, mocker):
        mock_client = self._mock_client(mocker)

        generate_anthropic("user content", "claude-test", "sk-test", max_tokens=512)

        assert mock_client.messages.create.call_args.kwargs["max_tokens"] == 512


class TestGenerateGeminiAdapter:
    """SDK-level tests for the Gemini adapter itself."""

    @staticmethod
    def _mock_client(mocker, text="# Morning Briefing", finish_reason=None):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = text
        if finish_reason is None:
            mock_response.candidates = []
        else:
            mock_candidate = MagicMock(spec=[])
            mock_candidate.finish_reason = finish_reason
            mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response
        mocker.patch("google.genai.Client", return_value=mock_client)
        return mock_client

    def test_request_has_finite_timeout(self, mocker):
        mock_client = self._mock_client(mocker)
        with patch("google.genai.Client", return_value=mock_client) as client:
            generate_gemini("user content", "gemini-test", "test-key")
        client.assert_called_once_with(
            api_key="test-key", http_options={"timeout": 120_000, "retry_options": {"attempts": 1}}
        )

    def test_calls_generate_content(self, mocker):
        mock_client = self._mock_client(mocker)

        result = generate_gemini("user content", "gemini-test", "test-key")

        assert "Morning Briefing" in result
        assert _TRUNCATION_WARNING not in result
        kwargs = mock_client.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-test"
        assert kwargs["contents"] == "user content"
        assert kwargs["config"]["system_instruction"] == SYSTEM_PROMPT
        assert kwargs["config"]["max_output_tokens"] == MAX_OUTPUT_TOKENS

    def test_appends_warning_on_truncation(self, mocker):
        """Enum-like finish_reason exposing a .name attribute."""
        enum_like = MagicMock()
        enum_like.name = "MAX_TOKENS"
        self._mock_client(mocker, text="# Truncated", finish_reason=enum_like)

        result = generate_gemini("user content", "gemini-test", "test-key")

        assert _TRUNCATION_WARNING in result

    def test_truncation_detection_with_string_finish_reason(self, mocker):
        """Handles SDK versions where finish_reason is a plain string."""
        self._mock_client(mocker, text="# Truncated", finish_reason="MAX_TOKENS")

        result = generate_gemini("user content", "gemini-test", "test-key")

        assert _TRUNCATION_WARNING in result

    def test_max_tokens_override(self, mocker):
        mock_client = self._mock_client(mocker)

        generate_gemini("user content", "gemini-test", "test-key", max_tokens=512)

        kwargs = mock_client.models.generate_content.call_args.kwargs
        assert kwargs["config"]["max_output_tokens"] == 512


class TestAgentConfigRepr:
    """Test that AgentConfig.__repr__ masks secrets."""

    def test_repr_masks_secrets(self):
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="super-secret-api-key-12345",
            ai_model="gemini-2.5-flash",
            email_from="sender@test.com",
            email_password="my-secret-password",
            email_to="recipient@test.com",
            email_smtp_host="smtp.gmail.com",
            email_smtp_port=587,
            scan_universe="full_market",
            max_deep_analysis=5,
        )

        result = repr(config)

        # Secrets must be masked
        assert "super-secret-api-key-12345" not in result
        assert "my-secret-password" not in result
        assert "***" in result

        # Non-secret fields must still be visible
        assert "gemini" in result
        assert "gemini-2.5-flash" in result
        assert "sender@test.com" in result
        assert "recipient@test.com" in result
        assert config.email_smtp_host in result
        assert "587" in result
        assert "full_market" in result
        assert "5" in result


class FakeSmtp:
    """Minimal stand-in for ``smtplib.SMTP`` usable as a ``send_email`` factory."""

    def __init__(self, sendmail_error: Exception | None = None):
        self.sendmail_error = sendmail_error
        self.host: str | None = None
        self.port: int | None = None
        self.starttls_calls = 0
        self.login_args: tuple[str, str] | None = None
        self.sent: list[tuple[str, list[str], str]] = []

    def __call__(self, host, port, *, timeout=None):
        self.timeout = timeout
        self.host = host
        self.port = port
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def starttls(self, context=None):
        self.starttls_calls += 1

    def login(self, user, password):
        self.login_args = (user, password)

    def sendmail(self, from_addr, to_addrs, message):
        if self.sendmail_error is not None:
            raise self.sendmail_error
        self.sent.append((from_addr, to_addrs, message))


@pytest.fixture
def fake_smtp(mocker) -> FakeSmtp:
    """Intercept real SMTP delivery so send sites can be observed end to end."""
    smtp = FakeSmtp()
    mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", smtp)
    return smtp


def _sent_message(smtp: FakeSmtp):
    """Parse the single message the fake SMTP captured."""
    assert len(smtp.sent) == 1, f"expected exactly one sent message, got {len(smtp.sent)}"
    _from_addr, _to_addrs, raw = smtp.sent[0]
    return email.message_from_string(raw)


def _sent_plain_body(smtp: FakeSmtp) -> str:
    """The text/plain part of the single message the fake SMTP captured."""
    for part in _sent_message(smtp).walk():
        if part.get_content_type() == "text/plain":
            return part.get_payload(decode=True).decode()
    raise AssertionError("sent message has no text/plain part")


def _creds(to_addr: str = "recipient@test.com", **kwargs) -> SmtpCreds:
    """Build test credentials from a comma-separated recipient string."""
    return SmtpCreds.from_parts(
        from_addr="sender@test.com", password="test-pass", to_addr=to_addr, **kwargs
    )


def _body_text(message) -> str:
    """Concatenate the decoded payloads of every part of a built message."""
    return "\n".join(
        part.get_payload(decode=True).decode() for part in message.walk() if not part.is_multipart()
    )


class TestSmtpCreds:
    """Test SmtpCreds construction."""

    def test_from_config(self):
        config = AgentConfig(
            email_from="sender@test.com",
            email_password="test-pass",
            email_to="alice@test.com, bob@test.com",
            email_smtp_host="custom.smtp.com",
            email_smtp_port=465,
        )

        creds = SmtpCreds.from_config(config)

        assert creds.from_addr == "sender@test.com"
        assert creds.password == "test-pass"
        assert creds.to_addrs == ["alice@test.com", "bob@test.com"]
        assert creds.smtp_host == "custom.smtp.com"
        assert creds.smtp_port == 465

    def test_defaults_smtp_host_and_port(self):
        creds = SmtpCreds.from_parts("a@b.com", "pw", "c@d.com")
        assert creds.smtp_host == "smtp.gmail.com"
        assert creds.smtp_port == 587

    def test_raises_on_empty_recipients(self):
        with pytest.raises(ValueError, match="No valid recipient"):
            _parse_recipients("")

        with pytest.raises(ValueError, match="No valid recipient"):
            _parse_recipients(",,")

        with pytest.raises(ValueError, match="No valid recipient"):
            SmtpCreds.from_parts("a@b.com", "pw", ",,")

    def test_filters_empty_recipients_from_trailing_comma(self):
        creds = _creds("alice@test.com,bob@test.com,")
        assert creds.to_addrs == ["alice@test.com", "bob@test.com"]


class TestBuildErrorMessage:
    """Test error message construction and sanitization."""

    def test_truncates_long_message(self):
        msg = build_error_message(_creds(), "A" * 1000)

        body = _body_text(msg)
        # The 1000-char message should be truncated to 500 chars
        assert "A" * 501 not in body
        assert "A" * 500 in body

    def test_redacts_urls(self):
        msg = build_error_message(
            _creds(), "Failed at https://api.example.com/key=abc123 during request"
        )

        body = _body_text(msg)
        assert "https://api.example.com/key=abc123" not in body
        assert "[URL redacted]" in body

    def test_redacts_secrets(self):
        msg = build_error_message(
            _creds(), "Error: key=sk-12345 and password: mysecret were exposed"
        )

        body = _body_text(msg)
        assert "sk-12345" not in body
        assert "mysecret" not in body
        assert "[REDACTED]" in body

    def test_empty_message(self):
        assert "Unknown error" in _body_text(build_error_message(_creds(), ""))

    def test_subject_and_recipients(self):
        msg = build_error_message(_creds("alice@test.com,bob@test.com"), "Something failed")

        assert msg["Subject"] == "Morning Briefing - ERROR"
        assert msg["From"] == "sender@test.com"
        assert msg["To"] == "alice@test.com, bob@test.com"


class TestBuildBriefingMessage:
    """Test briefing message construction."""

    def test_multipart_with_plain_and_html(self):
        msg = build_briefing_message(_creds(), "Test Briefing", "# Hello\n\nThis is a **test**.")

        assert msg["Subject"] == "Test Briefing"
        assert msg["From"] == "sender@test.com"
        assert msg["To"] == "recipient@test.com"
        content_types = [p.get_content_type() for p in msg.walk() if not p.is_multipart()]
        assert content_types == ["text/plain", "text/html"]

    def test_recipient_header_lists_every_address(self):
        msg = build_briefing_message(
            _creds("alice@test.com,bob@test.com,carol@test.com"), "Test", "# Hello"
        )
        assert msg["To"] == "alice@test.com, bob@test.com, carol@test.com"

    def test_sanitizes_html(self):
        # Markdown that contains an XSS script tag
        malicious_markdown = "# Hello\n\n<script>alert('xss')</script>\n\nSafe content here."

        msg = build_briefing_message(_creds(), "Test Briefing", malicious_markdown)

        html_part = next(p for p in msg.walk() if p.get_content_type() == "text/html")
        html = html_part.get_payload(decode=True).decode()
        # The script tag must be stripped from the HTML part by nh3
        assert "<script>" not in html
        assert "alert('xss')" not in html
        # But safe content should remain in the HTML part
        assert "Safe content here" in html

    def test_newlines_stripped_from_subject(self):
        msg = build_briefing_message(
            _creds(),
            "Test\r\nBcc: evil@attacker.com\r\nSubject: Injected",
            "# Hello",
        )

        assert msg["Subject"] == "TestBcc: evil@attacker.comSubject: Injected"
        # Newlines removed: no injected Bcc header on its own line
        assert "\nBcc:" not in msg.as_string()
        assert "\r\nBcc:" not in msg.as_string()

    def test_linkifies_ticker_symbols(self):
        msg = build_briefing_message(
            _creds(), "Test", "AAPL looks strong, RSI is high.", ticker_symbols=["AAPL"]
        )

        html_part = next(p for p in msg.walk() if p.get_content_type() == "text/html")
        html = html_part.get_payload(decode=True).decode()
        assert 'href="https://www.tradingview.com/chart/?symbol=AAPL"' in html
        assert ">AAPL</a>" in html
        # Non-candidate acronyms stay unlinked
        assert ">RSI</a>" not in html

    def test_no_linkification_without_symbols(self):
        msg = build_briefing_message(_creds(), "Test", "AAPL looks strong.")

        html_part = next(p for p in msg.walk() if p.get_content_type() == "text/html")
        assert "tradingview.com" not in html_part.get_payload(decode=True).decode()


class TestSendEmail:
    """Test the single SMTP transport function via an injected factory."""

    def test_sends_via_factory(self):
        creds = _creds("alice@test.com,bob@test.com")
        message = build_briefing_message(creds, "Test Briefing", "# Hello")
        smtp = FakeSmtp()

        send_email(message, creds, smtp_factory=smtp)

        assert smtp.host == "smtp.gmail.com"
        assert smtp.port == 587
        assert smtp.timeout == 30
        assert smtp.starttls_calls == 1
        assert smtp.login_args == ("sender@test.com", "test-pass")
        from_addr, to_addrs, sent = smtp.sent[0]
        assert from_addr == "sender@test.com"
        # sendmail must receive a LIST of addresses, not a comma-separated string
        assert to_addrs == ["alice@test.com", "bob@test.com"]
        assert "text/plain" in sent
        assert "text/html" in sent

    def test_uses_custom_host_and_port(self):
        creds = _creds(smtp_host="custom.smtp.com", smtp_port=465)
        smtp = FakeSmtp()

        send_email(build_error_message(creds, "boom"), creds, smtp_factory=smtp)

        assert (smtp.host, smtp.port) == ("custom.smtp.com", 465)

    def test_raises_and_logs_smtp_exception(self, caplog):
        creds = _creds()
        smtp = FakeSmtp(sendmail_error=smtplib.SMTPException("Connection refused"))

        with caplog.at_level(logging.ERROR, logger="volume_price_analysis.agent.email_sender"):
            with pytest.raises(smtplib.SMTPException, match="Connection refused"):
                send_email(build_error_message(creds, "boom"), creds, smtp_factory=smtp)

        assert "Failed to send email" in caplog.text


class TestRunMorningBriefing:
    """Test the full orchestrator (all external calls mocked)."""

    @pytest.mark.asyncio
    async def test_dry_run_prints_to_stdout(self, mocker, capsys):
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_scan",
            return_value={
                "summary": {
                    "total_candidates": 2,
                    "bullish_setups": 1,
                    "bearish_setups": 1,
                    "high_conviction": 0,
                    "errors": 0,
                },
                "high_conviction_setups": [],
                "top_bullish": [{"symbol": "AAPL"}],
                "top_bearish": [{"symbol": "TSLA"}],
            },
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_options_analysis",
            return_value={
                "symbol": "AAPL",
                "composite_signal": {"score": 4.2},
            },
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value=BriefingResult(text="# Test Briefing\nLooks good!"),
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )

        await run_morning_briefing(config, dry_run=True, data_source=agent_source(["AAPL"]))

        captured = capsys.readouterr()
        assert "Test Briefing" in captured.out

    @pytest.mark.asyncio
    async def test_no_ai_mode_skips_generation(self, mocker, fake_smtp):
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_scan",
            return_value={
                "summary": {
                    "total_candidates": 0,
                    "bullish_setups": 0,
                    "bearish_setups": 0,
                    "high_conviction": 0,
                    "errors": 0,
                },
                "high_conviction_setups": [],
                "top_bullish": [],
                "top_bearish": [],
            },
        )

        mock_generate = mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )

        await run_morning_briefing(config, dry_run=False, no_ai=True, data_source=agent_source())

        mock_generate.assert_not_called()
        assert _sent_message(fake_smtp)["Subject"].startswith("Morning Market Data (Raw) - ")
        # The raw-data renderer produced the body (layout pinned in test_briefing.py).
        assert "# Morning Market Scan Results" in _sent_plain_body(fake_smtp)

    @pytest.mark.asyncio
    async def test_ai_failure_uses_fallback(self, mocker, fake_smtp):
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_scan",
            return_value={
                "summary": {
                    "total_candidates": 1,
                    "bullish_setups": 1,
                    "bearish_setups": 0,
                    "high_conviction": 0,
                    "errors": 0,
                },
                "high_conviction_setups": [],
                "top_bullish": [{"symbol": "AAPL"}],
                "top_bearish": [],
            },
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_options_analysis",
            return_value={"symbol": "AAPL", "composite_signal": {"score": 3.0}},
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            side_effect=Exception("API Error"),
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )

        await run_morning_briefing(
            config, dry_run=False, no_ai=False, data_source=agent_source(["AAPL"])
        )

        assert "Fallback" in _sent_plain_body(fake_smtp)

    @pytest.mark.asyncio
    async def test_regime_verdict_flags_picks_and_heads_email(self, mocker, fake_smtp):
        """PDE-66: a bearish tape flags bullish picks as counter-regime (keeping
        their high-conviction billing) and the regime verdict heads the email body."""
        import pandas as pd

        bull = {"symbol": "AAPL", "composite_score": 4.5}
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_scan",
            return_value={
                "summary": {
                    "total_candidates": 1,
                    "bullish_setups": 1,
                    "bearish_setups": 0,
                    "high_conviction": 1,
                    "errors": 0,
                },
                "high_conviction_setups": [bull],
                "top_bullish": [bull],
                "top_bearish": [],
            },
        )

        # Fixed fixture window: 29 sessions at 600 then a close at 550 puts
        # SPY below its 20-day SMA -> bearish regime.
        spy_data = pd.DataFrame(
            {
                "Date": pd.bdate_range("2026-07-01", periods=30),
                "Close": [600.0] * 29 + [550.0],
            }
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_options_analysis",
            return_value={"symbol": "AAPL", "composite_signal": {"score": 4.5}},
        )
        mock_generate = mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value=BriefingResult(text="# Briefing"),
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )

        await run_morning_briefing(
            config, dry_run=False, no_ai=False, data_source=agent_source(["AAPL"], spy=spy_data)
        )

        annotated_scan = mock_generate.call_args.kwargs["scan_results"]
        assert annotated_scan["market_regime"]["regime"] == "bearish"
        # Annotation only: the pick keeps its high-conviction billing and count.
        assert [c["symbol"] for c in annotated_scan["high_conviction_setups"]] == ["AAPL"]
        assert annotated_scan["high_conviction_setups"][0]["regime_conflict"]
        assert annotated_scan["summary"]["high_conviction"] == 1

        # Delivery only; the body's layout is pinned in tests/test_briefing.py.
        assert "## Pick Summary" in _sent_plain_body(fake_smtp)

    @pytest.mark.asyncio
    async def test_earnings_from_source_warn_the_analysis_and_the_prompt(self, mocker, fake_smtp):
        """The earnings guard runs end-to-end against the injected data source."""
        bull = {"symbol": "AAPL", "composite_score": 4.5}
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_scan",
            return_value={
                "summary": {
                    "total_candidates": 1,
                    "bullish_setups": 1,
                    "bearish_setups": 0,
                    "high_conviction": 0,
                    "errors": 0,
                },
                "high_conviction_setups": [],
                "top_bullish": [bull],
                "top_bearish": [],
            },
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_options_analysis",
            return_value={"symbol": "AAPL", "composite_signal": {"score": 4.5}},
        )
        mock_generate = mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value="# Briefing",
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )
        source = agent_source(["AAPL"], earnings={"AAPL": datetime.now(UTC) + timedelta(days=5)})

        await run_morning_briefing(config, dry_run=False, no_ai=False, data_source=source)

        preamble = mock_generate.call_args.kwargs["earnings_preamble"]
        assert "EARNINGS EVENT RISK" in preamble
        assert "AAPL" in preamble
        deep = mock_generate.call_args.kwargs["deep_analyses"]
        assert "EARNINGS" in deep[0]["earnings_warning"]


# ---------------------------------------------------------------------------
# Morning agent coverage: lines 82-83, 127-129, 212-252, 256
# ---------------------------------------------------------------------------


class TestDeepAnalysisException:
    """Test exception handling during deep analysis loop (lines 82-83)."""

    @pytest.mark.asyncio
    async def test_exception_in_fetch_is_caught_and_logged(self, mocker, capsys):
        """When fetch_stock_data raises, the exception is caught and analysis continues."""
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_scan",
            return_value={
                "summary": {
                    "total_candidates": 2,
                    "bullish_setups": 1,
                    "bearish_setups": 1,
                    "high_conviction": 0,
                    "errors": 0,
                },
                "high_conviction_setups": [],
                "top_bullish": [{"symbol": "AAPL"}],
                "top_bearish": [{"symbol": "TSLA"}],
            },
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value=BriefingResult(text="# Briefing with no deep data"),
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=2,
        )

        # Should NOT raise despite fetch_stock_data failing for every symbol
        await run_morning_briefing(
            config,
            dry_run=True,
            data_source=agent_source(
                errors=dict.fromkeys(["AAPL", "TSLA"], ValueError("Yahoo Finance unavailable"))
            ),
        )

        captured = capsys.readouterr()
        assert "Briefing with no deep data" in captured.out

    @pytest.mark.asyncio
    async def test_exception_in_options_analysis_is_caught(self, mocker, capsys):
        """When run_options_analysis raises, the exception is caught and analysis continues."""
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_scan",
            return_value={
                "summary": {
                    "total_candidates": 1,
                    "bullish_setups": 1,
                    "bearish_setups": 0,
                    "high_conviction": 0,
                    "errors": 0,
                },
                "high_conviction_setups": [],
                "top_bullish": [{"symbol": "AAPL"}],
                "top_bearish": [],
            },
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_options_analysis",
            side_effect=ValueError("Bad data"),
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value=BriefingResult(text="# Briefing after analysis error"),
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )

        await run_morning_briefing(config, dry_run=True, data_source=agent_source(["AAPL"]))

        captured = capsys.readouterr()
        assert "Briefing after analysis error" in captured.out


class TestDryRunNoAi:
    """Test dry_run=True + no_ai=True path (lines 127-129)."""

    @pytest.mark.asyncio
    async def test_prints_json_to_stdout(self, mocker, capsys):
        """dry_run + no_ai prints scan JSON and deep analyses JSON to stdout."""
        scan_data = {
            "summary": {
                "total_candidates": 1,
                "bullish_setups": 1,
                "bearish_setups": 0,
                "high_conviction": 0,
                "errors": 0,
            },
            "high_conviction_setups": [],
            "top_bullish": [{"symbol": "MSFT"}],
            "top_bearish": [],
        }
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_scan",
            return_value=scan_data,
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_options_analysis",
            return_value={
                "symbol": "MSFT",
                "composite_signal": {"score": 4.0},
            },
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )

        await run_morning_briefing(
            config, dry_run=True, no_ai=True, data_source=agent_source(["MSFT"])
        )

        captured = capsys.readouterr()
        # The raw-data body is what dry run prints in --no-ai mode.
        assert "# Morning Market Scan Results" in captured.out
        assert "total_candidates" in captured.out
        # deep analysis JSON printed (lines 128-129)
        assert "MSFT" in captured.out
        assert "composite_signal" in captured.out

    @pytest.mark.asyncio
    async def test_prints_json_with_no_candidates(self, mocker, capsys):
        """dry_run + no_ai with no candidates still prints scan JSON."""
        scan_data = {
            "summary": {
                "total_candidates": 0,
                "bullish_setups": 0,
                "bearish_setups": 0,
                "high_conviction": 0,
                "errors": 0,
            },
            "high_conviction_setups": [],
            "top_bullish": [],
            "top_bearish": [],
        }
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_scan",
            return_value=scan_data,
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )

        await run_morning_briefing(config, dry_run=True, no_ai=True, data_source=agent_source())

        captured = capsys.readouterr()
        assert "total_candidates" in captured.out


class TestMain:
    """Test the main() entry point (lines 212-252, 256)."""

    def test_main_dry_run_no_ai_succeeds(self, mocker):
        """main() with --dry-run --no-ai parses args and runs briefing."""
        mocker.patch("sys.argv", ["morning-briefing", "--dry-run", "--no-ai"])
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.AgentConfig.from_env",
            return_value=AgentConfig(
                ai_provider="gemini",
                ai_provider_api_key="test-key",
                email_from="a@b.com",
                email_password="pass",
                email_to="c@d.com",
            ),
        )
        mock_run = mocker.patch(
            "volume_price_analysis.agent.morning_agent.asyncio.run",
            return_value=_briefing_result(),
        )

        main()

        mock_run.assert_called_once()
        # Close the real coroutine main() handed to the mocked asyncio.run,
        # else it warns "coroutine was never awaited" at GC time.
        mock_run.call_args[0][0].close()

    def test_main_config_validation_failure_exits(self, mocker):
        """Config validation errors cause sys.exit(1)."""
        mocker.patch("sys.argv", ["morning-briefing"])
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.AgentConfig.from_env",
            return_value=AgentConfig(
                ai_provider="gemini",
                ai_provider_api_key="",
                email_from="",
                email_password="",
                email_to="",
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_main_no_ai_skips_api_key_errors(self, mocker):
        """--no-ai filters out API_KEY validation errors."""
        mocker.patch("sys.argv", ["morning-briefing", "--no-ai"])
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.AgentConfig.from_env",
            return_value=AgentConfig(
                ai_provider="gemini",
                ai_provider_api_key="",
                email_from="a@b.com",
                email_password="pass",
                email_to="c@d.com",
            ),
        )
        mock_run = mocker.patch(
            "volume_price_analysis.agent.morning_agent.asyncio.run",
            return_value=_briefing_result(),
        )

        main()

        mock_run.assert_called_once()
        # Close the real coroutine main() handed to the mocked asyncio.run,
        # else it warns "coroutine was never awaited" at GC time.
        mock_run.call_args[0][0].close()

    def test_main_dry_run_with_ai_needs_api_key(self, mocker):
        """--dry-run without --no-ai still validates AI config."""
        mocker.patch("sys.argv", ["morning-briefing", "--dry-run"])
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.AgentConfig.from_env",
            return_value=AgentConfig(
                ai_provider="gemini",
                ai_provider_api_key="",
                email_from="",
                email_password="",
                email_to="",
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_main_dry_run_no_ai_skips_all_validation(self, mocker):
        """--dry-run + --no-ai skips all config validation."""
        mocker.patch("sys.argv", ["morning-briefing", "--dry-run", "--no-ai"])
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.AgentConfig.from_env",
            return_value=AgentConfig(
                ai_provider="gemini",
                ai_provider_api_key="",
                email_from="",
                email_password="",
                email_to="",
            ),
        )
        mock_run = mocker.patch(
            "volume_price_analysis.agent.morning_agent.asyncio.run",
            return_value=_briefing_result(),
        )

        main()

        mock_run.assert_called_once()
        # Close the real coroutine main() handed to the mocked asyncio.run,
        # else it warns "coroutine was never awaited" at GC time.
        mock_run.call_args[0][0].close()

    def test_main_dry_run_with_valid_ai_config_succeeds(self, mocker):
        """--dry-run with valid AI config passes validation."""
        mocker.patch("sys.argv", ["morning-briefing", "--dry-run"])
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.AgentConfig.from_env",
            return_value=AgentConfig(
                ai_provider="gemini",
                ai_provider_api_key="valid-key",
                email_from="",
                email_password="",
                email_to="",
            ),
        )
        mock_run = mocker.patch(
            "volume_price_analysis.agent.morning_agent.asyncio.run",
            return_value=_briefing_result(),
        )

        main()

        mock_run.assert_called_once()
        # Close the real coroutine main() handed to the mocked asyncio.run,
        # else it warns "coroutine was never awaited" at GC time.
        mock_run.call_args[0][0].close()

    def test_main_critical_failure_sends_error_email(self, mocker, fake_smtp):
        """Critical exception sends the error report and exits 1."""
        mocker.patch("sys.argv", ["morning-briefing"])
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.AgentConfig.from_env",
            return_value=AgentConfig(
                ai_provider="gemini",
                ai_provider_api_key="test-key",
                email_from="a@b.com",
                email_password="pass",
                email_to="c@d.com",
            ),
        )
        mock_run = mocker.patch(
            "volume_price_analysis.agent.morning_agent.asyncio.run",
            side_effect=RuntimeError("Critical failure"),
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        assert _sent_message(fake_smtp)["Subject"] == "Morning Briefing - ERROR"
        assert "Critical failure" in _sent_plain_body(fake_smtp)
        # Close the real coroutine main() handed to the mocked asyncio.run.
        mock_run.call_args[0][0].close()

    def test_main_error_email_failure_is_logged_and_exit_code_kept(self, mocker, caplog):
        """A failing error-report send is logged; main still exits 1."""
        mocker.patch("sys.argv", ["morning-briefing"])
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.AgentConfig.from_env",
            return_value=AgentConfig(
                ai_provider="gemini",
                ai_provider_api_key="test-key",
                email_from="a@b.com",
                email_password="pass",
                email_to="c@d.com",
            ),
        )
        mock_run = mocker.patch(
            "volume_price_analysis.agent.morning_agent.asyncio.run",
            side_effect=RuntimeError("Critical failure"),
        )
        mocker.patch(
            "volume_price_analysis.agent.email_sender.smtplib.SMTP",
            FakeSmtp(sendmail_error=smtplib.SMTPException("SMTP down")),
        )

        with caplog.at_level(logging.ERROR, logger="volume_price_analysis.agent.morning_agent"):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        assert "Failed to send error email" in caplog.text
        mock_run.call_args[0][0].close()

    def test_main_critical_failure_dry_run_no_error_email(self, mocker, fake_smtp):
        """dry-run critical failure does NOT send error email."""
        mocker.patch("sys.argv", ["morning-briefing", "--dry-run", "--no-ai"])
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.AgentConfig.from_env",
            return_value=AgentConfig(
                ai_provider="gemini",
                ai_provider_api_key="test-key",
                email_from="a@b.com",
                email_password="pass",
                email_to="c@d.com",
            ),
        )
        mock_run = mocker.patch(
            "volume_price_analysis.agent.morning_agent.asyncio.run",
            side_effect=RuntimeError("Critical failure"),
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        assert fake_smtp.sent == []
        # Close the real coroutine main() handed to the mocked asyncio.run.
        mock_run.call_args[0][0].close()

    def test_main_critical_failure_missing_email_password_skips_error_email(self, mocker):
        """Missing email_password means error email is not sent on failure."""
        mocker.patch("sys.argv", ["morning-briefing", "--no-ai"])
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.AgentConfig.from_env",
            return_value=AgentConfig(
                ai_provider="gemini",
                ai_provider_api_key="test-key",
                email_from="a@b.com",
                email_password="",
                email_to="c@d.com",
            ),
        )

        # email_password is empty, validation will fail at lines 225-228
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_main_degraded_briefing_exits_code_2(self, mocker):
        """A degraded BriefingRunResult makes main() exit with code 2."""
        mocker.patch("sys.argv", ["morning-briefing"])
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.AgentConfig.from_env",
            return_value=AgentConfig(
                ai_provider="gemini",
                ai_provider_api_key="test-key",
                email_from="a@b.com",
                email_password="pass",
                email_to="c@d.com",
            ),
        )
        mock_run = mocker.patch(
            "volume_price_analysis.agent.morning_agent.asyncio.run",
            return_value=_briefing_result(degraded=True, reason="AI provider unavailable"),
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        # Close the real coroutine main() handed to the mocked asyncio.run.
        mock_run.call_args[0][0].close()


class TestEmailFormatValidation:
    """Test email address format validation in AgentConfig.validate()."""

    def test_invalid_email_from_format(self):
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="not-an-email",
            email_password="pass",
            email_to="valid@example.com",
        )
        errors = config.validate()
        assert any("EMAIL_FROM" in e and "not a valid email" in e for e in errors)

    def test_invalid_email_to_format(self):
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="valid@example.com",
            email_password="pass",
            email_to="bad-address,also-bad",
        )
        errors = config.validate()
        assert any("EMAIL_TO" in e and "not a valid email" in e for e in errors)

    def test_valid_emails_pass_validation(self):
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="user@example.com",
            email_password="pass",
            email_to="a@b.co,c@d.org",
        )
        errors = config.validate()
        assert not errors

    def test_missing_dot_after_at(self):
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="user@localhost",
            email_password="pass",
            email_to="valid@example.com",
        )
        errors = config.validate()
        assert any("not a valid email" in e for e in errors)


class TestRunMorningBriefingDegradedReturn:
    """run_morning_briefing reports degradation (and why) via BriefingRunResult."""

    @pytest.mark.asyncio
    async def test_ai_failure_returns_degraded_result(self, mocker, fake_smtp):
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_scan",
            return_value={
                "summary": {
                    "total_candidates": 1,
                    "bullish_setups": 1,
                    "bearish_setups": 0,
                    "high_conviction": 0,
                    "errors": 0,
                },
                "high_conviction_setups": [],
                "top_bullish": [{"symbol": "AAPL"}],
                "top_bearish": [],
            },
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_options_analysis",
            return_value={"symbol": "AAPL", "composite_signal": {"score": 3.0}},
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            side_effect=Exception("API Error"),
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )

        result = await run_morning_briefing(
            config, dry_run=False, no_ai=False, data_source=agent_source(["AAPL"])
        )
        assert result.degraded is True
        assert result.reason is not None
        assert "gemini" in result.reason
        assert result.symbols_analyzed == ["AAPL"]
        assert result.email_sent is True
        assert "regime" in result.regime

    @pytest.mark.asyncio
    async def test_successful_briefing_returns_healthy_result(self, mocker, fake_smtp):
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_scan",
            return_value={
                "summary": {
                    "total_candidates": 0,
                    "bullish_setups": 0,
                    "bearish_setups": 0,
                    "high_conviction": 0,
                    "errors": 0,
                },
                "high_conviction_setups": [],
                "top_bullish": [],
                "top_bearish": [],
            },
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value=BriefingResult(text="# Briefing"),
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )

        result = await run_morning_briefing(
            config, dry_run=False, no_ai=False, data_source=agent_source()
        )
        assert result.degraded is False
        assert result.reason is None
        assert result.symbols_analyzed == []
        assert result.email_sent is True


class TestSystemPromptVolatilityLabeling:
    """The briefing prompt must label volatility honestly as an HV proxy (HOM-39)."""

    def test_prompt_flags_volatility_as_hv_proxy(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "historical volatility" in lowered
        assert "hv" in lowered

    def test_prompt_warns_against_implied_volatility_framing(self):
        """The prompt must tell the model the metric is not options-implied vol."""
        lowered = SYSTEM_PROMPT.lower()
        assert "implied volatility" in lowered


class TestSystemPromptGrounding:
    """HOM-45: SYSTEM_PROMPT must instruct the model to only use provided data."""

    def test_prompt_has_grounding_clause(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "use only" in lowered
        assert "never invent" in lowered

    def test_prompt_mentions_tickers_must_come_from_data(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "ticker" in lowered


class TestSystemPromptConsistency:
    """The prompt must forbid conflicting values and overstated target labels."""

    def test_prompt_requires_one_value_per_metric(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "one value per metric" in lowered
        assert "deep-analysis values" in lowered

    def test_prompt_labels_targets_as_one_sigma_expected_move(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "standard deviation" in lowered
        assert "not as predictions" in lowered

    def test_prompt_clarifies_hv_percentile_is_relative(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "own recent history" in lowered


class TestFindUngroundedTickers:
    """HOM-45: flag ticker-like tokens that are absent from the scan/analysis input."""

    @staticmethod
    def _scan(*, bullish=(), bearish=(), high_conviction=(), errors=()):
        return {
            "summary": {},
            "high_conviction_setups": [{"symbol": s} for s in high_conviction],
            "top_bullish": [{"symbol": s} for s in bullish],
            "top_bearish": [{"symbol": s} for s in bearish],
            "errors": [{"symbol": s, "error": "boom"} for s in errors],
        }

    def test_grounded_briefing_returns_empty(self):
        scan = self._scan(bullish=["AAPL"], bearish=["TSLA"])
        briefing = "## Top Picks\n- AAPL looks bullish\n- TSLA looks bearish"
        assert find_ungrounded_tickers(briefing, scan, []) == []

    def test_detects_hallucinated_ticker(self):
        scan = self._scan(bullish=["AAPL"])
        briefing = "AAPL is great. Also consider ZZZZ for a play."
        assert find_ungrounded_tickers(briefing, scan, []) == ["ZZZZ"]

    def test_ignores_indicator_acronyms(self):
        scan = self._scan(bullish=["AAPL"])
        briefing = (
            "AAPL: ADX 31, RSI 62, VWAP above, POC/VAH/VAL aligned, "
            "HV percentile high, OBV rising, MFI 70, CMF positive, ATR wide, BB squeeze, "
            "14-day DTE, ROC up."
        )
        assert find_ungrounded_tickers(briefing, scan, []) == []

    def test_ignores_emphasis_and_common_words(self):
        scan = self._scan(bullish=["AAPL"])
        briefing = (
            "STRONG BULLISH on AAPL. NO TREND elsewhere. VOLUME BREAKOUT, "
            "EXTREME OVERBOUGHT. WARNING: IMPORTANT risk. US markets, ETF flows, AI theme."
        )
        assert find_ungrounded_tickers(briefing, scan, []) == []

    def test_recognizes_symbols_from_deep_analyses(self):
        scan = self._scan()
        deep = [{"symbol": "NVDA"}]
        briefing = "NVDA shows accumulation into the close."
        assert find_ungrounded_tickers(briefing, scan, deep) == []

    def test_recognizes_symbols_from_error_list(self):
        scan = self._scan(errors=["GOOGL"])
        briefing = "Note: GOOGL failed to fetch but was in scope."
        assert find_ungrounded_tickers(briefing, scan, []) == []

    def test_cashtag_grounded(self):
        scan = self._scan(bullish=["AAPL"])
        assert find_ungrounded_tickers("Buy $AAPL calls.", scan, []) == []

    def test_cashtag_hallucinated_is_flagged(self):
        scan = self._scan(bullish=["AAPL"])
        assert find_ungrounded_tickers("Rotate into $ZM here.", scan, []) == ["ZM"]

    def test_ignores_bare_single_letters(self):
        scan = self._scan(bullish=["AAPL"])
        briefing = "Plan B is fine; option A too; F was not analyzed."
        assert find_ungrounded_tickers(briefing, scan, []) == []

    def test_handles_class_share_symbol(self):
        scan = self._scan(bullish=["BRK-B"])
        briefing = "BRK.B is consolidating near its POC."
        assert find_ungrounded_tickers(briefing, scan, []) == []

    def test_dedupes_and_sorts(self):
        scan = self._scan(bullish=["AAPL"])
        briefing = "WXYZ and MNOP and WXYZ again, plus AAPL."
        assert find_ungrounded_tickers(briefing, scan, []) == ["MNOP", "WXYZ"]

    def test_handles_empty_briefing(self):
        scan = self._scan(bullish=["AAPL"])
        assert find_ungrounded_tickers("", scan, []) == []

    def test_handles_empty_scan_data(self):
        assert find_ungrounded_tickers("Consider ZZZZ today.", {}, []) == ["ZZZZ"]


class TestGenerateBriefingGrounding:
    """HOM-45: generate_briefing reports tickers the briefing invented."""

    @staticmethod
    def _scan():
        return {
            "summary": {"total_candidates": 1},
            "high_conviction_setups": [],
            "top_bullish": [{"symbol": "AAPL"}],
            "top_bearish": [],
            "errors": [],
        }

    def test_reports_and_logs_hallucinated_ticker(self, caplog):
        import logging

        provider = _FakeProvider(text="AAPL is bullish. Also buy ZZZZ calls.")

        with caplog.at_level(logging.WARNING, logger="volume_price_analysis.agent.ai_client"):
            result = generate_briefing(
                scan_results=self._scan(),
                deep_analyses=[],
                provider=provider,
                model="m",
                api_key="k",
            )

        assert "ZZZZ" in result.text
        assert result.ungrounded_tickers == ["ZZZZ"]
        assert "ZZZZ" in caplog.text
        assert "hallucination" in caplog.text.lower()

    def test_no_warning_when_grounded(self, caplog):
        import logging

        provider = _FakeProvider(text="AAPL is bullish today; watch the VWAP.")

        with caplog.at_level(logging.WARNING, logger="volume_price_analysis.agent.ai_client"):
            result = generate_briefing(
                scan_results=self._scan(),
                deep_analyses=[],
                provider=provider,
                model="m",
                api_key="k",
            )

        assert result.ungrounded_tickers == []
        assert "hallucination" not in caplog.text.lower()


# ---------------------------------------------------------------------------
# Earnings guard tests
# ---------------------------------------------------------------------------

NOW_UTC = datetime(2026, 6, 28, 12, 0, tzinfo=UTC)


class TestCheckEarnings:
    """Unit tests for _check_earnings against an injected data source.

    Parsing Yahoo's raw ``.info`` payload now lives in YFinanceDataSource, so
    those cases are covered in test_data_fetcher.py; here the source hands back
    a datetime (or nothing) and only the 14-day window logic is under test.
    """

    def _source(self, earnings_dt):
        return InMemoryDataSource(earnings={"AAPL": earnings_dt})

    def test_no_earnings_date_returns_none(self):
        assert _check_earnings("AAPL", NOW_UTC, self._source(None)) is None

    def test_unknown_symbol_returns_none(self):
        assert _check_earnings("AAPL", NOW_UTC, InMemoryDataSource()) is None

    def test_earnings_within_14_days_returns_warning(self):
        result = _check_earnings("AAPL", NOW_UTC, self._source(NOW_UTC + timedelta(days=7)))
        assert result is not None
        assert "EARNINGS" in result
        assert "7 day" in result

    def test_earnings_in_past_returns_none(self):
        past = NOW_UTC - timedelta(days=3)
        assert _check_earnings("AAPL", NOW_UTC, self._source(past)) is None

    def test_earnings_more_than_14_days_out_returns_none(self):
        far_future = NOW_UTC + timedelta(days=30)
        assert _check_earnings("AAPL", NOW_UTC, self._source(far_future)) is None

    def test_boundary_exactly_14_days_out_warns(self):
        """The window is inclusive at both ends."""
        edge = NOW_UTC + timedelta(days=_HOLDING_PERIOD_DAYS)
        assert _check_earnings("AAPL", NOW_UTC, self._source(edge)) is not None

    def test_after_hours_earnings_uses_market_calendar_day(self):
        now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
        earnings = datetime(2026, 8, 18, 0, 30, tzinfo=UTC)  # August 17 at 20:30 ET
        result = _check_earnings("AAPL", now, self._source(earnings))
        assert result == "EARNINGS in 14 day(s) (2026-08-17)"

    def test_naive_datetime_uses_calendar_day_as_is(self):
        """A naive earnings date keeps its calendar day without timezone conversion."""
        naive_upcoming = datetime(2026, 7, 3)  # naive midnight, 5 days out
        result = _check_earnings("AAPL", NOW_UTC, self._source(naive_upcoming))
        assert result == "EARNINGS in 5 day(s) (2026-07-03)"

    def test_data_source_failure_returns_none(self):
        """A provider blowing up must not sink the briefing."""

        class FailingSource(InMemoryDataSource):
            def earnings_date(self, symbol):
                raise RuntimeError("provider down")

        assert _check_earnings("AAPL", NOW_UTC, FailingSource()) is None


class TestFetchEarningsWarnings:
    """Tests for the concurrent batch fetch helper."""

    def test_empty_symbols_returns_empty_dict(self):
        assert _fetch_earnings_warnings([], NOW_UTC, InMemoryDataSource()) == {}

    def test_returns_only_symbols_with_warnings(self):
        source = InMemoryDataSource(earnings={"NVDA": NOW_UTC + timedelta(days=5)})

        result = _fetch_earnings_warnings(["AAPL", "NVDA", "MSFT"], NOW_UTC, source)

        assert set(result) == {"NVDA"}
        assert "EARNINGS in 5 day(s)" in result["NVDA"]

    def test_slow_lookup_times_out_and_keeps_other_symbols(self, mocker, caplog):
        release = Event()
        finished = Event()

        class SlowSource(InMemoryDataSource):
            def earnings_date(self, symbol):
                if symbol == "AAPL":
                    try:
                        release.wait()
                    finally:
                        finished.set()
                return NOW_UTC + timedelta(days=5)

        mocker.patch(
            "volume_price_analysis.agent.morning_agent.EARNINGS_TIMEOUT_SECONDS",
            0.01,
        )
        caller = ThreadPoolExecutor(max_workers=1)
        try:
            with caplog.at_level(logging.DEBUG):
                future = caller.submit(
                    _fetch_earnings_warnings, ["AAPL", "NVDA"], NOW_UTC, SlowSource()
                )
                result = future.result(timeout=2)
            assert not finished.is_set(), "briefing waited for the timed-out lookup"
            assert set(result) == {"NVDA"}
            assert "Earnings lookup failed for AAPL" in caplog.text
            assert any(
                record.exc_info and isinstance(record.exc_info[1], TimeoutError)
                for record in caplog.records
            )
        finally:
            release.set()
            caller.shutdown(wait=True)
            assert finished.wait(timeout=2)

    def test_thread_pool_is_bounded(self):
        from concurrent.futures import ThreadPoolExecutor

        from volume_price_analysis.agent.morning_agent import _EARNINGS_MAX_WORKERS

        symbols = [f"SYM{i}" for i in range(_EARNINGS_MAX_WORKERS * 5)]
        with patch(
            "volume_price_analysis.agent.morning_agent.ThreadPoolExecutor",
            wraps=ThreadPoolExecutor,
        ) as mock_pool:
            _fetch_earnings_warnings(symbols, NOW_UTC, InMemoryDataSource())

        mock_pool.assert_called_once_with(max_workers=_EARNINGS_MAX_WORKERS)


# ---------------------------------------------------------------------------
# Ticker linkify tests
# ---------------------------------------------------------------------------


class TestLinkifyTickers:
    """Tests for _linkify_tickers in email_sender."""

    def test_wraps_ticker_in_tradingview_link(self):
        html = "<p><strong>AAPL</strong> looks bullish.</p>"
        result = _linkify_tickers(html, {"AAPL"})
        assert 'href="https://www.tradingview.com/chart/?symbol=AAPL"' in result
        assert ">AAPL<" in result

    def test_no_symbols_returns_unchanged(self):
        html = "<p>AAPL and NVDA look bullish.</p>"
        assert _linkify_tickers(html) == html
        assert _linkify_tickers(html, set()) == html

    def test_only_links_whitelisted_symbols(self):
        html = "<p>AAPL is up but NVDA is down.</p>"
        result = _linkify_tickers(html, {"AAPL"})
        assert 'symbol=AAPL"' in result
        assert 'symbol=NVDA"' not in result

    def test_skips_indicator_acronyms(self):
        html = "<p>RSI is 26.7 and ADX is 47.0 with HV at 21.5%.</p>"
        result = _linkify_tickers(html, {"BSX"})
        assert "<a " not in result

    def test_does_not_truncate_long_caps_words(self):
        html = "<p>EXTREME OVERSOLD CONDITION.</p>"
        result = _linkify_tickers(html, {"EXTRE", "OVERS", "CONDI"})
        assert "<a " not in result

    def test_does_not_link_option_strikes(self):
        html = "<p>Buy the 430C and sell the 475C.</p>"
        result = _linkify_tickers(html, {"C"})
        assert "<a " not in result

    def test_links_common_word_ticker_when_candidate(self):
        html = "<p><strong>HAS</strong> shows a bullish divergence.</p>"
        result = _linkify_tickers(html, {"HAS"})
        assert 'symbol=HAS"' in result

    def test_longest_symbol_wins_overlap(self):
        html = "<p>GOOGL broke out.</p>"
        result = _linkify_tickers(html, {"GOOG", "GOOGL"})
        assert 'symbol=GOOGL"' in result
        assert 'symbol=GOOG"' not in result

    def test_does_not_double_link(self):
        html = '<p><a href="https://example.com">TSLA</a> is volatile.</p>'
        result = _linkify_tickers(html, {"TSLA"})
        assert result.count("<a ") == 1

    def test_skips_html_tag_content(self):
        html = '<div CLASS="foo">NVDA is up</div>'
        result = _linkify_tickers(html, {"CLASS", "NVDA"})
        assert 'symbol=CLASS"' not in result
        assert 'symbol=NVDA"' in result

    def test_hyphenated_symbol(self):
        html = "<p>BRK-B is consolidating.</p>"
        result = _linkify_tickers(html, {"BRK-B"})
        assert 'symbol=BRK-B"' in result

    def test_empty_string(self):
        assert _linkify_tickers("", {"AAPL"}) == ""


class TestBuildEarningsPreamble:
    """build_earnings_preamble renders the AI prompt's earnings-risk block."""

    def test_empty_warnings_render_nothing(self):
        assert build_earnings_preamble({}) == ""

    def test_warnings_are_sorted_and_labelled(self):
        preamble = build_earnings_preamble(
            {"MSFT": "EARNINGS in 3 day(s)", "AAPL": "EARNINGS in 7 day(s)"}
        )
        assert "EARNINGS EVENT RISK" in preamble
        assert "within 14 days" in preamble
        assert preamble.index("AAPL") < preamble.index("MSFT")
        assert "  - AAPL: EARNINGS in 7 day(s)" in preamble
        assert preamble.startswith("\n\n")
        assert preamble.endswith("\n")


class TestConfigErrors:
    """_config_errors keeps only the errors that can actually block a run mode."""

    @staticmethod
    def _config(**overrides):
        base = {
            "ai_provider": "gemini",
            "ai_provider_api_key": "",
            "email_from": "",
            "email_password": "",
            "email_to": "",
        }
        base.update(overrides)
        return AgentConfig(**base)

    def test_dry_run_no_ai_ignores_everything(self):
        assert _config_errors(self._config(), dry_run=True, no_ai=True) == []

    def test_dry_run_with_ai_keeps_only_ai_errors(self):
        errors = _config_errors(self._config(), dry_run=True, no_ai=False)
        assert errors == ["AI_PROVIDER_API_KEY is required"]

    def test_no_ai_keeps_only_email_errors(self):
        errors = _config_errors(self._config(), dry_run=False, no_ai=True)
        assert all("API_KEY" not in e and "AI_PROVIDER" not in e for e in errors)
        assert "EMAIL_FROM is required" in errors

    def test_full_run_keeps_all_errors(self):
        errors = _config_errors(self._config(), dry_run=False, no_ai=False)
        assert "AI_PROVIDER_API_KEY is required" in errors
        assert "EMAIL_FROM is required" in errors

    def test_valid_config_has_no_errors(self):
        config = self._config(
            ai_provider_api_key="k", email_from="a@b.com", email_password="p", email_to="c@d.com"
        )
        assert _config_errors(config, dry_run=False, no_ai=False) == []


# ---------------------------------------------------------------------------
# PDE-69: fixed conviction vocabulary and programmatic date
# ---------------------------------------------------------------------------


class TestSystemPromptDateAndConviction:
    def test_prompt_forbids_date_placeholders(self):
        assert "[Today's Date]" in SYSTEM_PROMPT
        assert "do NOT write" in SYSTEM_PROMPT

    def test_prompt_fixes_conviction_vocabulary(self):
        assert '"conviction" field' in SYSTEM_PROMPT
        assert "HIGH, MEDIUM, or LOW" in SYSTEM_PROMPT
        assert "Signal Quality" in SYSTEM_PROMPT  # named as a forbidden scale


class TestFormatBriefingDate:
    def test_no_zero_padding(self):
        assert format_briefing_date(date(2026, 9, 4)) == "Friday, September 4, 2026"

    def test_two_digit_day(self):
        assert format_briefing_date(date(2026, 12, 25)) == "Friday, December 25, 2026"


class TestStripDatePlaceholders:
    def test_removes_bracketed_placeholder_lines(self):
        text = (
            "# Briefing\n"
            "Date: [Today's Date]\n"
            "**Date:** [Insert Date]\n"
            "*Date*: [DATE]\n"
            "**Date: [Insert Date]**\n"
            "## Date: [Today's Date]\n"
            "- Date: [Today's Date]\n"
            "Real content\n"
        )
        assert strip_date_placeholders(text) == "# Briefing\nReal content\n"

    def test_keeps_real_dates_and_other_brackets(self):
        text = "Date: September 4, 2026\nSee [link](x) for the date\n"
        assert strip_date_placeholders(text) == text

    def test_leaves_links_and_trailing_prose_intact(self):
        # A bracket that is a markdown link or is followed by prose is not a
        # placeholder line; stripping a prefix would leave debris behind.
        text = (
            "Date: [September 4, 2026](https://example.com/calendar) — earnings\n"
            "Date: [Today's Date] - Market Open\n"
        )
        assert strip_date_placeholders(text) == text

    def test_empty(self):
        assert strip_date_placeholders("") == ""


class TestBriefingDateInPrompt:
    def test_date_is_stated_up_front(self):
        prompt = build_briefing_prompt({"summary": {}}, [], briefing_date=date(2026, 9, 4))
        assert "Briefing date: Friday, September 4, 2026 (2026-09-04, US/Eastern)." in prompt
        assert prompt.index("Briefing date:") < prompt.index("## Scan Results")

    def test_omitted_when_unknown(self):
        assert "Briefing date" not in build_briefing_prompt({"summary": {}}, [])

    def test_generate_briefing_passes_date_and_scrubs_output(self):
        provider = _FakeProvider(text="Date: [Today's Date]\n# Briefing\nBody")
        result = generate_briefing(
            scan_results={"summary": {}},
            deep_analyses=[],
            provider=provider,
            model="m",
            api_key="k",
            briefing_date=date(2026, 9, 4),
        )
        assert "Briefing date: Friday, September 4, 2026" in provider.user_content
        assert result.text == "# Briefing\nBody"


class TestConvictionReachesTheModel:
    def test_deep_projection_carries_conviction(self):
        assert _project_deep_analysis({"symbol": "A", "conviction": "HIGH"})["conviction"] == "HIGH"
        assert "conviction" not in _project_deep_analysis({"symbol": "A"})

    def test_prompt_contains_conviction_for_scan_candidates(self):
        scan = {
            "summary": {},
            "high_conviction_setups": [{"symbol": "AAPL", "composite_score": 5.0}],
            "top_bullish": [{"symbol": "AAPL", "composite_score": 5.0}],
        }
        prompt = build_briefing_prompt(scan, [], picks=build_picks(scan))
        assert '"conviction": "HIGH"' in prompt


class TestRunMorningBriefingDateAndPicks:
    @pytest.mark.asyncio
    async def test_date_is_eastern_and_body_is_templated(self, mocker, fake_smtp):
        scan = {
            "scan_parameters": {"symbols_scanned": 10},
            "summary": {
                "total_candidates": 1,
                "bullish_setups": 1,
                "bearish_setups": 0,
                "high_conviction": 0,
                "errors": 0,
            },
            "high_conviction_setups": [],
            "top_bullish": [{"symbol": "AAPL", "composite_score": 4.4, "latest_price": 10.0}],
            "top_bearish": [],
        }
        mocker.patch("volume_price_analysis.agent.morning_agent.run_scan", return_value=scan)
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_options_analysis",
            return_value={
                "symbol": "AAPL",
                "composite_signal": {"score": 4.4},
                "latest_price": 10.5,
            },
        )
        mock_generate = mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value=BriefingResult(text="## Executive Summary\nok"),
        )
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="k",
            email_from="a@b.com",
            email_password="p",
            email_to="c@d.com",
            max_deep_analysis=1,
        )
        # 01:30 UTC on the 5th is still the evening of the 4th in New York.
        now = datetime(2026, 9, 5, 1, 30, tzinfo=UTC)

        await run_morning_briefing(config, data_source=agent_source(["AAPL"]), now=now)

        assert mock_generate.call_args.kwargs["briefing_date"] == date(2026, 9, 4)
        # One derivation, three readers (PDE-150): the Pick list, the deep
        # analysis grafted from it, and the prompt projection fed from it.
        picks_to_model = mock_generate.call_args.kwargs["picks"]
        assert [(p.symbol, p.conviction) for p in picks_to_model] == [("AAPL", "MEDIUM")]
        assert mock_generate.call_args.kwargs["deep_analyses"][0]["conviction"] == "MEDIUM"
        prompt = build_briefing_prompt(
            mock_generate.call_args.kwargs["scan_results"], [], picks=picks_to_model
        )
        assert '"conviction": "MEDIUM"' in prompt

        assert _sent_message(fake_smtp)["Subject"] == "Morning Market Briefing - 2026-09-04"
        # The Eastern date reaches the rendered title; the layout itself is
        # pinned in tests/test_briefing.py.
        body = _sent_plain_body(fake_smtp)
        assert body.startswith("# Morning Market Briefing — Friday, September 4, 2026\n")

    @pytest.mark.asyncio
    async def test_fallback_briefing_gets_the_same_template(self, mocker, capsys):
        scan = {
            "summary": {
                "total_candidates": 0,
                "bullish_setups": 0,
                "bearish_setups": 0,
                "high_conviction": 0,
                "errors": 0,
            },
            "high_conviction_setups": [],
            "top_bullish": [],
            "top_bearish": [],
        }
        mocker.patch("volume_price_analysis.agent.morning_agent.run_scan", return_value=scan)
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            side_effect=RuntimeError("down"),
        )
        config = AgentConfig(ai_provider="gemini", ai_provider_api_key="k")

        result = await run_morning_briefing(
            config,
            dry_run=True,
            data_source=agent_source(),
            now=datetime(2026, 9, 4, 12, 30, tzinfo=UTC),
        )

        out = capsys.readouterr().out
        assert result.degraded
        # The fallback renderer ran (layout pinned in tests/test_briefing.py).
        assert "## Fallback Briefing (AI unavailable)" in out

    @pytest.mark.asyncio
    async def test_naive_now_is_rejected(self):
        config = AgentConfig(ai_provider="gemini", ai_provider_api_key="k")
        with pytest.raises(ValueError, match="aware"):
            await run_morning_briefing(
                config, dry_run=True, data_source=agent_source(), now=datetime(2026, 9, 4)
            )

    @pytest.mark.asyncio
    async def test_regime_check_uses_the_briefing_date(self, mocker):
        """The regime's causal cutoff is the injected briefing date, not the wall clock."""
        scan = {
            "summary": {
                "total_candidates": 0,
                "bullish_setups": 0,
                "bearish_setups": 0,
                "high_conviction": 0,
                "errors": 0,
            },
            "high_conviction_setups": [],
            "top_bullish": [],
            "top_bearish": [],
        }
        mocker.patch("volume_price_analysis.agent.morning_agent.run_scan", return_value=scan)
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value=BriefingResult(text="ok"),
        )
        regime = mocker.patch(
            "volume_price_analysis.agent.morning_agent.compute_market_regime",
            return_value={"regime": "unknown", "reason": "x"},
        )
        config = AgentConfig(ai_provider="gemini", ai_provider_api_key="k")

        await run_morning_briefing(
            config,
            dry_run=True,
            data_source=agent_source(spy=_STUB_FRAME),
            now=datetime(2026, 9, 5, 1, 30, tzinfo=UTC),
        )

        assert regime.call_args.kwargs["today"] == date(2026, 9, 4)


@pytest.fixture(autouse=True)
def retry_sleep(mocker):
    return mocker.patch("volume_price_analysis.agent.retry._sleep")


@pytest.mark.parametrize("failures, status", [(2, None), (3, None), (1, 400), (1, 401)])
async def test_ai_failures_before_fallback(mocker, capsys, retry_sleep, failures, status):
    mocker.patch(
        "volume_price_analysis.agent.morning_agent.run_scan",
        return_value={
            "summary": {
                "total_candidates": 0,
                "bullish_setups": 0,
                "bearish_setups": 0,
                "high_conviction": 0,
                "errors": 0,
            },
            "high_conviction_setups": [],
            "top_bullish": [],
            "top_bearish": [],
        },
    )
    error = TimeoutError()
    if status is not None:
        import anthropic
        import httpx

        error = anthropic.APIStatusError(
            "test error",
            response=httpx.Response(status, request=httpx.Request("POST", "https://example.test")),
            body=None,
        )
    provider = mocker.Mock(side_effect=[error] * failures + ["Recovered model prose"])
    mocker.patch.dict(PROVIDERS, {"anthropic": provider})
    result = await run_morning_briefing(
        AgentConfig(ai_provider="anthropic"), dry_run=True, data_source=agent_source()
    )
    body = capsys.readouterr().out
    assert provider.call_count == (1 if status is not None else 3)
    expected_delays = [] if status is not None else [mocker.call(2), mocker.call(4)]
    assert retry_sleep.call_args_list == expected_delays
    recovered = failures == 2
    assert result.degraded is (not recovered)
    assert ("## Fallback Briefing (AI unavailable)" in body) is (not recovered)
    if recovered:
        assert "Recovered model prose" in body
        assert result.reason is None
    else:
        assert "Recovered model prose" not in body
        assert "used fallback briefing" in result.reason


@pytest.mark.parametrize(
    "error",
    [
        smtplib.SMTPServerDisconnected("lost connection"),
        smtplib.SMTPDataError(451, b"try later"),
        OSError("connection failed"),
        TimeoutError(),
    ],
)
def test_email_recovers_from_transient_failure(mocker, retry_sleep, error):
    smtp = FakeSmtp()
    factory = mocker.Mock(side_effect=[error, smtp])
    creds = SmtpCreds.from_parts("a@b.com", "test", "c@d.com")
    send_email(build_error_message(creds, "boom"), creds, smtp_factory=factory)
    assert len(smtp.sent) == 1
    assert factory.call_count == 2
    retry_sleep.assert_called_once_with(2)


@pytest.mark.parametrize(
    "error",
    [
        smtplib.SMTPResponseException(421, b"closing connection"),
        smtplib.SMTPServerDisconnected("lost connection"),
        OSError("connection dropped"),
    ],
)
def test_email_cleanup_failure_after_send_is_success(mocker, retry_sleep, caplog, error):
    class CleanupFailureSmtp(FakeSmtp):
        def __exit__(self, *exc_info):
            raise error

    smtp = CleanupFailureSmtp()
    factory = mocker.Mock(return_value=smtp)
    creds = SmtpCreds.from_parts("a@b.com", "test", "c@d.com")
    with caplog.at_level(logging.WARNING):
        send_email(build_error_message(creds, "boom"), creds, smtp_factory=factory)
    assert len(smtp.sent) == 1
    factory.assert_called_once()
    retry_sleep.assert_not_called()
    assert "SMTP cleanup failed after successful send" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        smtplib.SMTPAuthenticationError(454, b"auth failed"),
        smtplib.SMTPDataError(550, b"rejected"),
        smtplib.SMTPNotSupportedError("no TLS"),
    ],
)
def test_email_permanent_failure_is_not_retried(mocker, retry_sleep, error):
    factory = mocker.Mock(side_effect=error)
    creds = SmtpCreds.from_parts("a@b.com", "test", "c@d.com")
    with pytest.raises(type(error)) as raised:
        send_email(build_error_message(creds, "boom"), creds, smtp_factory=factory)
    assert raised.value is error
    factory.assert_called_once()
    retry_sleep.assert_not_called()


@pytest.mark.parametrize("provider_name", ["anthropic", "gemini"])
@pytest.mark.parametrize(
    "status, retryable",
    [(408, True), (409, True), (429, True), (503, True), (400, False), (401, False), (403, False)],
)
def test_ai_provider_status_retry_policy(provider_name, status, retryable):
    import anthropic
    import httpx
    from google.genai.errors import APIError

    from volume_price_analysis.agent.ai_client import is_transient_ai_error

    if provider_name == "anthropic":
        error = anthropic.APIStatusError(
            "test error",
            response=httpx.Response(status, request=httpx.Request("POST", "https://example.test")),
            body=None,
        )
    else:
        error = APIError(status, {"error": {"message": "test error"}})
    assert is_transient_ai_error(error) is retryable


def test_ai_error_without_status_code_is_not_retried():
    from google.genai.errors import APIError

    from volume_price_analysis.agent.ai_client import is_transient_ai_error

    error = APIError(500, {"error": {"message": "test error"}})
    error.code = None  # type: ignore[assignment]
    assert is_transient_ai_error(error) is False


@pytest.mark.parametrize(
    "error",
    [
        smtplib.SMTPServerDisconnected("disconnected"),
        smtplib.SMTPDataError(451, b"try later"),
        smtplib.SMTPRecipientsRefused({"c@d.com": (450, b"busy")}),
        OSError("connection failed"),
        TimeoutError("timed out"),
        ssl.SSLError("TLS failed"),
        ConnectionError("connection lost"),
    ],
)
def test_email_send_exhaustion_reraises_last_error(mocker, retry_sleep, caplog, error):
    smtp = FakeSmtp(sendmail_error=error)
    factory = mocker.Mock(return_value=smtp)
    creds = SmtpCreds.from_parts("a@b.com", "test", "c@d.com")
    with pytest.raises(type(error)) as raised:
        send_email(build_error_message(creds, "boom"), creds, smtp_factory=factory)
    assert raised.value is error
    assert factory.call_count == 3
    assert smtp.sent == []
    assert retry_sleep.call_args_list == [mocker.call(2), mocker.call(4)]

    records = [r for r in caplog.records if r.message == "Failed to send email"]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].exc_info[1] is error


@pytest.mark.parametrize("provider_name", ["anthropic", "gemini"])
@pytest.mark.parametrize("text", ["", " \n\t", "Date: [Today's Date]\n"])
@pytest.mark.parametrize("truncated", [False, True])
async def test_empty_model_text_emails_fallback_without_retry(
    mocker, fake_smtp, retry_sleep, provider_name, text, truncated
):
    if provider_name == "anthropic":
        client = TestGenerateAnthropicAdapter._mock_client(
            mocker, text=text, stop_reason="max_tokens" if truncated else "end_turn"
        )
        request = client.messages.create
    else:
        client = TestGenerateGeminiAdapter._mock_client(
            mocker, text=text, finish_reason="MAX_TOKENS" if truncated else None
        )
        request = client.models.generate_content
    mocker.patch(
        "volume_price_analysis.agent.morning_agent.run_scan",
        return_value={
            "summary": {
                "total_candidates": 0,
                "bullish_setups": 0,
                "bearish_setups": 0,
                "high_conviction": 0,
                "errors": 0,
            },
            "high_conviction_setups": [],
            "top_bullish": [],
            "top_bearish": [],
        },
    )
    config = AgentConfig(
        ai_provider=provider_name,
        ai_provider_api_key="test-key",
        email_from="a@b.com",
        email_password="test",
        email_to="c@d.com",
    )

    result = await run_morning_briefing(config, data_source=agent_source())

    assert result.degraded
    assert "used fallback briefing" in result.reason
    assert result.email_sent
    assert "## Fallback Briefing (AI unavailable)" in _sent_plain_body(fake_smtp)
    request.assert_called_once()
    retry_sleep.assert_not_called()


def test_gemini_safety_block_is_non_transient_failure(mocker):
    client = TestGenerateGeminiAdapter._mock_client(mocker, text=None, finish_reason="SAFETY")

    with pytest.raises(ValueError, match="empty briefing") as raised:
        generate_briefing({}, [], generate_gemini, "model", "key")

    assert not is_transient_ai_error(raised.value)
    client.models.generate_content.assert_called_once()


async def test_earnings_failure_warns_and_briefing_continues(mocker, caplog, capsys):
    class FailingEarningsSource(InMemoryDataSource):
        def earnings_date(self, symbol):
            raise ValueError("unexpected earningsDate shape")

    mocker.patch(
        "volume_price_analysis.agent.morning_agent.run_scan",
        return_value={
            "summary": {
                "total_candidates": 1,
                "high_conviction": 0,
                "bullish_setups": 1,
                "bearish_setups": 0,
                "errors": 0,
            },
            "top_bullish": [{"symbol": "AAPL", "composite_score": 3.0}],
        },
    )
    mocker.patch(
        "volume_price_analysis.agent.morning_agent.run_options_analysis",
        return_value={"symbol": "AAPL", "composite_signal": {"score": 3.0}},
    )
    provider = _FakeProvider(text="Morning briefing prose")
    mocker.patch.dict(PROVIDERS, {"gemini": provider})

    result = await run_morning_briefing(
        AgentConfig(ai_provider="gemini"),
        dry_run=True,
        data_source=FailingEarningsSource(frames={"AAPL": _STUB_FRAME}),
    )

    assert not result.degraded
    assert "Morning briefing prose" in capsys.readouterr().out
    assert "EARNINGS" not in provider.user_content
    records = [r for r in caplog.records if r.message == "Earnings lookup failed for AAPL"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert isinstance(records[0].exc_info[1], ValueError)


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [5.0, -5.0])
@pytest.mark.parametrize("placement", ["high_only", "both", "regular_only"])
@pytest.mark.parametrize("analysis_status", ["capped", "failed", "success"])
async def test_tjx_earnings_demoted_in_prompt_and_briefing(
    mocker, capsys, score, placement, analysis_status
):
    candidate = {"symbol": "TJX", "composite_score": score, "adx": 35, "hv_percentile": 20}
    direction = "bullish" if score > 0 else "bearish"
    safe = {"symbol": "SAFE", "composite_score": 5.0}
    scan = {
        "summary": {
            "total_candidates": 2,
            "bullish_setups": 1,
            "bearish_setups": 0,
            "high_conviction": 2,
        },
        "high_conviction_setups": [safe] if placement == "regular_only" else [candidate, safe],
        "top_bullish": [],
        "top_bearish": [],
    }
    if placement != "high_only":
        scan[f"top_{direction}"] = [candidate]
    mocker.patch("volume_price_analysis.agent.morning_agent.run_scan", return_value=scan)
    generate = mocker.patch(
        "volume_price_analysis.agent.morning_agent.generate_briefing",
        return_value=SimpleNamespace(text="Briefing prose"),
    )
    mocker.patch(
        "volume_price_analysis.agent.morning_agent.run_options_analysis",
        return_value={"symbol": "TJX", "composite_signal": {"score": score}},
    )
    source = agent_source(
        ["TJX"] if analysis_status == "success" else [],
        earnings={"TJX": datetime(2026, 8, 6)},
    )
    await run_morning_briefing(
        AgentConfig(max_deep_analysis=0 if analysis_status == "capped" else 2),
        dry_run=True,
        data_source=source,
        now=datetime(2026, 8, 3, 12, tzinfo=UTC),
    )

    args = generate.call_args.kwargs
    assert args["scan_results"]["high_conviction_setups"] == [safe]
    assert args["scan_results"]["summary"]["high_conviction"] == 1
    assert args["scan_results"][f"top_{direction}"][0]["symbol"] == "TJX"
    picks = {pick.symbol: pick for pick in args["picks"]}
    assert picks["TJX"].conviction == "MEDIUM"
    assert picks["SAFE"].conviction == "HIGH"
    assert "2026-08-06" in picks["TJX"].earnings_warning
    assert "never present them as high conviction" in args["earnings_preamble"]
    prompt = build_briefing_prompt(
        args["scan_results"], args["deep_analyses"], args["earnings_preamble"], picks=args["picks"]
    )
    projected_scan = json.loads(prompt.split("```json\n", 1)[1].split("```", 1)[0])
    assert [c["symbol"] for c in projected_scan["high_conviction_setups"]] == ["SAFE"]
    assert projected_scan[f"top_{direction}"][0]["conviction"] == "MEDIUM"
    body = capsys.readouterr().out
    assert f"| TJX | {direction} | MEDIUM |" in body
    assert "2026-08-06" in body
    if analysis_status == "success":
        assert args["deep_analyses"][0]["conviction"] == "MEDIUM"
    assert scan["high_conviction_setups"] == (
        [safe] if placement == "regular_only" else [candidate, safe]
    )
    assert "earnings_warning" not in candidate


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("earnings", "expected"),
    [
        (datetime(2026, 8, 3), "MEDIUM"),  # date-only, same day
        (datetime(2026, 8, 17, 23, tzinfo=UTC), "MEDIUM"),  # final holding day
        (datetime(2026, 8, 18), "HIGH"),  # date-only, just outside
        (datetime(2026, 8, 2, tzinfo=UTC), "HIGH"),  # past
        (None, "HIGH"),
        (ValueError("lookup unavailable"), "HIGH"),
    ],
)
async def test_earnings_holding_window_and_missing_dates(mocker, capsys, earnings, expected):
    class EarningsSource(InMemoryDataSource):
        def earnings_date(self, symbol):
            if isinstance(earnings, Exception):
                raise earnings
            return super().earnings_date(symbol)

    candidate = {"symbol": "TJX", "composite_score": 5.0, "adx": 35, "hv_percentile": 20}
    mocker.patch(
        "volume_price_analysis.agent.morning_agent.run_scan",
        return_value={
            "summary": {
                "total_candidates": 1,
                "bullish_setups": 1,
                "bearish_setups": 0,
                "high_conviction": 1,
            },
            "high_conviction_setups": [candidate],
            "top_bullish": [candidate],
            "top_bearish": [],
        },
    )
    generate = mocker.patch(
        "volume_price_analysis.agent.morning_agent.generate_briefing",
        side_effect=ValueError("AI unavailable"),
    )
    source = EarningsSource(earnings={"TJX": None if isinstance(earnings, Exception) else earnings})
    await run_morning_briefing(
        AgentConfig(max_deep_analysis=0),
        dry_run=True,
        data_source=source,
        now=datetime(2026, 8, 3, 12, tzinfo=UTC),
    )
    args = generate.call_args.kwargs
    assert args["picks"][0].conviction == expected
    assert bool(args["scan_results"]["high_conviction_setups"]) == (expected == "HIGH")
    body = capsys.readouterr().out
    assert f"| TJX | bullish | {expected} |" in body
    assert f"**High conviction:** {1 if expected == 'HIGH' else 0}" in body


def test_prompt_carries_pick_entry_stop_and_target_without_mutating_candidates():
    from copy import deepcopy

    from volume_price_analysis.agent.picks import build_picks

    candidate = {"symbol": "TEST", "latest_price": 100.0, "composite_score": -4.0, "atr": 3.0}
    scan = {"top_bearish": [candidate]}
    original = deepcopy(scan)
    deep = [{"symbol": "TEST", "latest_price": 999.0}]
    prompt = build_briefing_prompt(scan, deep, picks=build_picks(scan, deep))
    projected = json.loads(prompt.split("```json\n", 1)[1].split("```", 1)[0])
    row = projected["top_bearish"][0]
    assert (row["entry"], row["stop"], row["target"]) == (100.0, 106.0, 88.0)
    assert scan == original
