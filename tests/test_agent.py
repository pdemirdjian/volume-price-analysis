"""Tests for the morning briefing agent."""

from unittest.mock import MagicMock

import pytest

from volume_price_analysis.agent.claude_client import generate_briefing
from volume_price_analysis.agent.config import AgentConfig
from volume_price_analysis.agent.email_sender import send_briefing_email
from volume_price_analysis.agent.morning_agent import (
    _fallback_briefing,
    _get_top_symbols,
    run_morning_briefing,
)


class TestAgentConfig:
    """Test configuration loading and validation."""

    def test_from_env_loads_required_vars(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.setenv("EMAIL_FROM", "test@example.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "test-password")
        monkeypatch.setenv("EMAIL_TO", "recipient@example.com")

        config = AgentConfig.from_env()
        assert config.anthropic_api_key == "sk-test-key"
        assert config.email_from == "test@example.com"
        assert config.email_password == "test-password"
        assert config.email_to == "recipient@example.com"

    def test_from_env_uses_defaults(self, monkeypatch):
        # Clear any existing env vars
        for key in [
            "ANTHROPIC_API_KEY",
            "EMAIL_FROM",
            "EMAIL_PASSWORD",
            "EMAIL_TO",
            "EMAIL_SMTP_HOST",
            "SCAN_UNIVERSE",
            "MAX_DEEP_ANALYSIS",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = AgentConfig.from_env()
        assert config.email_smtp_host == "smtp.gmail.com"
        assert config.email_smtp_port == 587
        assert config.scan_universe == "full_market"
        assert config.max_deep_analysis == 5
        assert config.claude_model == "claude-sonnet-4-5-20250929"

    def test_from_env_overrides_defaults(self, monkeypatch):
        monkeypatch.setenv("SCAN_UNIVERSE", "tech")
        monkeypatch.setenv("MAX_DEEP_ANALYSIS", "10")
        monkeypatch.setenv("EMAIL_SMTP_PORT", "465")

        config = AgentConfig.from_env()
        assert config.scan_universe == "tech"
        assert config.max_deep_analysis == 10
        assert config.email_smtp_port == 465

    def test_validate_missing_required(self):
        config = AgentConfig()
        errors = config.validate()
        assert len(errors) == 4
        assert any("ANTHROPIC_API_KEY" in e for e in errors)
        assert any("EMAIL_FROM" in e for e in errors)
        assert any("EMAIL_PASSWORD" in e for e in errors)
        assert any("EMAIL_TO" in e for e in errors)

    def test_validate_all_present(self):
        config = AgentConfig(
            anthropic_api_key="sk-test",
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


class TestFallbackBriefing:
    """Test fallback briefing when Claude API fails."""

    def test_includes_summary_stats(self):
        scan_results = {
            "summary": {
                "total_candidates": 15,
                "bullish_setups": 10,
                "bearish_setups": 5,
                "high_conviction": 3,
            },
        }
        briefing = _fallback_briefing(scan_results, [])
        assert "15" in briefing
        assert "10" in briefing
        assert "5" in briefing
        assert "3" in briefing

    def test_includes_deep_analyses(self):
        scan_results = {
            "summary": {
                "total_candidates": 1,
                "bullish_setups": 1,
                "bearish_setups": 0,
                "high_conviction": 0,
            }
        }
        deep = [
            {
                "symbol": "AAPL",
                "latest_price": 150.0,
                "composite_signal": {"score": 5.5, "recommendation": "strong_bullish"},
            },
        ]
        briefing = _fallback_briefing(scan_results, deep)
        assert "AAPL" in briefing
        assert "150.00" in briefing
        assert "5.5" in briefing


class TestGenerateBriefing:
    """Test Claude API integration (mocked)."""

    def test_calls_anthropic_api(self, mocker):
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="# Morning Briefing\nTest content")]
        mock_message.usage.input_tokens = 100
        mock_message.usage.output_tokens = 200
        mock_client.messages.create.return_value = mock_message

        mocker.patch(
            "volume_price_analysis.agent.claude_client.anthropic.Anthropic",
            return_value=mock_client,
        )

        result = generate_briefing(
            scan_results={"summary": {"total_candidates": 5}},
            deep_analyses=[],
            api_key="sk-test",
        )

        assert "Morning Briefing" in result
        mock_client.messages.create.assert_called_once()

    def test_includes_scan_data_in_prompt(self, mocker):
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="briefing")]
        mock_message.usage.input_tokens = 100
        mock_message.usage.output_tokens = 200
        mock_client.messages.create.return_value = mock_message

        mocker.patch(
            "volume_price_analysis.agent.claude_client.anthropic.Anthropic",
            return_value=mock_client,
        )

        generate_briefing(
            scan_results={"key": "value"},
            deep_analyses=[{"symbol": "AAPL"}],
            api_key="sk-test",
        )

        call_args = mock_client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "AAPL" in user_msg
        assert "key" in user_msg


class TestSendBriefingEmail:
    """Test email sending (mocked SMTP)."""

    def test_sends_multipart_email(self, mocker):
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        send_briefing_email(
            subject="Test Briefing",
            body_markdown="# Hello\n\nThis is a **test**.",
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
        )

        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("sender@test.com", "test-pass")
        mock_smtp_instance.sendmail.assert_called_once()

        # Verify email content
        sent_args = mock_smtp_instance.sendmail.call_args
        assert sent_args[0][0] == "sender@test.com"
        assert sent_args[0][1] == "recipient@test.com"
        # Message should contain both plain text and HTML parts
        msg_str = sent_args[0][2]
        assert "text/plain" in msg_str
        assert "text/html" in msg_str


class TestRunMorningBriefing:
    """Test the full orchestrator (all external calls mocked)."""

    @pytest.mark.asyncio
    async def test_dry_run_prints_to_stdout(self, mocker, capsys):
        # Mock scan
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

        # Mock fetch + analysis
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.fetch_stock_data",
            return_value=MagicMock(),
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_options_analysis",
            return_value={
                "symbol": "AAPL",
                "composite_signal": {"score": 4.2},
            },
        )

        # Mock Claude
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value="# Test Briefing\nLooks good!",
        )

        config = AgentConfig(
            anthropic_api_key="sk-test",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )

        await run_morning_briefing(config, dry_run=True)

        captured = capsys.readouterr()
        assert "Test Briefing" in captured.out

    @pytest.mark.asyncio
    async def test_no_ai_mode_skips_claude(self, mocker):
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
        mock_raw_email = mocker.patch(
            "volume_price_analysis.agent.morning_agent.send_raw_data_email",
        )

        config = AgentConfig(
            anthropic_api_key="sk-test",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )

        await run_morning_briefing(config, dry_run=False, no_ai=True)

        mock_generate.assert_not_called()
        mock_raw_email.assert_called_once()

    @pytest.mark.asyncio
    async def test_claude_failure_uses_fallback(self, mocker):
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
            "volume_price_analysis.agent.morning_agent.fetch_stock_data",
            return_value=MagicMock(),
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_options_analysis",
            return_value={"symbol": "AAPL", "composite_signal": {"score": 3.0}},
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            side_effect=Exception("API Error"),
        )
        mock_send = mocker.patch(
            "volume_price_analysis.agent.morning_agent.send_briefing_email",
        )

        config = AgentConfig(
            anthropic_api_key="sk-test",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )

        await run_morning_briefing(config, dry_run=False, no_ai=False)

        # Should still send an email with fallback content
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args
        assert "Fallback" in call_kwargs.kwargs["body_markdown"] or "Fallback" in (
            call_kwargs[1].get("body_markdown", "")
            if len(call_kwargs) > 1
            else call_kwargs[0][1]
            if len(call_kwargs[0]) > 1
            else ""
        )
