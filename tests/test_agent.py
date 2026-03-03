"""Tests for the morning briefing agent."""

from unittest.mock import MagicMock

import pytest

from volume_price_analysis.agent.ai_client import _TRUNCATION_WARNING, generate_briefing
from volume_price_analysis.agent.config import AgentConfig
from volume_price_analysis.agent.email_sender import send_briefing_email, send_error_email
from volume_price_analysis.agent.morning_agent import (
    _fallback_briefing,
    _get_top_symbols,
    run_morning_briefing,
)


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


class TestFallbackBriefing:
    """Test fallback briefing when AI API fails."""

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


class TestGenerateBriefingAnthropic:
    """Test Anthropic API integration (mocked)."""

    def test_calls_anthropic_api(self, mocker):
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="# Morning Briefing\nTest content")]
        mock_message.usage.input_tokens = 100
        mock_message.usage.output_tokens = 200
        mock_client.messages.create.return_value = mock_message

        mocker.patch("anthropic.Anthropic", return_value=mock_client)

        result = generate_briefing(
            scan_results={"summary": {"total_candidates": 5}},
            deep_analyses=[],
            provider="anthropic",
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

        mocker.patch("anthropic.Anthropic", return_value=mock_client)

        generate_briefing(
            scan_results={"key": "value"},
            deep_analyses=[{"symbol": "AAPL"}],
            provider="anthropic",
            api_key="sk-test",
        )

        call_args = mock_client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "AAPL" in user_msg
        assert "key" in user_msg

    def test_appends_warning_on_truncation(self, mocker):
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="# Truncated briefing")]
        mock_message.usage.input_tokens = 100
        mock_message.usage.output_tokens = 16384
        mock_message.stop_reason = "max_tokens"
        mock_client.messages.create.return_value = mock_message

        mocker.patch("anthropic.Anthropic", return_value=mock_client)

        result = generate_briefing(
            scan_results={},
            deep_analyses=[],
            provider="anthropic",
            api_key="sk-test",
        )

        assert _TRUNCATION_WARNING in result

    def test_no_warning_on_normal_completion(self, mocker):
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="# Full briefing")]
        mock_message.usage.input_tokens = 100
        mock_message.usage.output_tokens = 500
        mock_message.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_message

        mocker.patch("anthropic.Anthropic", return_value=mock_client)

        result = generate_briefing(
            scan_results={},
            deep_analyses=[],
            provider="anthropic",
            api_key="sk-test",
        )

        assert _TRUNCATION_WARNING not in result


class TestGenerateBriefingGemini:
    """Test Gemini API integration (mocked)."""

    def test_calls_gemini_api(self, mocker):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "# Morning Briefing\nGemini content"
        mock_client.models.generate_content.return_value = mock_response

        mocker.patch("google.genai.Client", return_value=mock_client)

        result = generate_briefing(
            scan_results={"summary": {"total_candidates": 5}},
            deep_analyses=[],
            provider="gemini",
            api_key="test-key",
        )

        assert "Morning Briefing" in result
        mock_client.models.generate_content.assert_called_once()

    def test_uses_default_model(self, mocker):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "briefing"
        mock_client.models.generate_content.return_value = mock_response

        mocker.patch("google.genai.Client", return_value=mock_client)

        generate_briefing(
            scan_results={},
            deep_analyses=[],
            provider="gemini",
            api_key="test-key",
        )

        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-2.5-flash"

    def test_appends_warning_on_truncation(self, mocker):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "# Truncated briefing"
        mock_finish_reason = MagicMock()
        mock_finish_reason.name = "MAX_TOKENS"
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = mock_finish_reason
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        mocker.patch("google.genai.Client", return_value=mock_client)

        result = generate_briefing(
            scan_results={},
            deep_analyses=[],
            provider="gemini",
            api_key="test-key",
        )

        assert _TRUNCATION_WARNING in result

    def test_no_warning_on_normal_completion(self, mocker):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "# Full briefing"
        mock_finish_reason = MagicMock()
        mock_finish_reason.name = "STOP"
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = mock_finish_reason
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        mocker.patch("google.genai.Client", return_value=mock_client)

        result = generate_briefing(
            scan_results={},
            deep_analyses=[],
            provider="gemini",
            api_key="test-key",
        )

        assert _TRUNCATION_WARNING not in result

    def test_truncation_detection_with_string_finish_reason(self, mocker):
        """Handles SDK versions where finish_reason is a plain string."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "# Truncated briefing"
        mock_candidate = MagicMock(spec=[])  # no attributes by default
        mock_candidate.finish_reason = "MAX_TOKENS"
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response

        mocker.patch("google.genai.Client", return_value=mock_client)

        result = generate_briefing(
            scan_results={},
            deep_analyses=[],
            provider="gemini",
            api_key="test-key",
        )

        assert _TRUNCATION_WARNING in result


class TestGenerateBriefingInvalidProvider:
    """Test error handling for invalid provider."""

    def test_raises_on_invalid_provider(self):
        with pytest.raises(ValueError, match="Unknown AI provider"):
            generate_briefing(
                scan_results={},
                deep_analyses=[],
                provider="openai",
                api_key="key",
            )


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

        sent_args = mock_smtp_instance.sendmail.call_args
        assert sent_args[0][0] == "sender@test.com"
        assert sent_args[0][1] == ["recipient@test.com"]
        msg_str = sent_args[0][2]
        assert "text/plain" in msg_str
        assert "text/html" in msg_str


    def test_sends_to_multiple_recipients(self, mocker):
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        send_briefing_email(
            subject="Test Briefing",
            body_markdown="# Hello",
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="alice@test.com,bob@test.com,carol@test.com",
        )

        sent_args = mock_smtp_instance.sendmail.call_args
        # sendmail must receive a LIST of addresses, not a comma-separated string
        assert sent_args[0][1] == ["alice@test.com", "bob@test.com", "carol@test.com"]

    def test_send_error_email_multiple_recipients(self, mocker):
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        send_error_email(
            error_message="Something failed",
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="alice@test.com,bob@test.com",
        )

        sent_args = mock_smtp_instance.sendmail.call_args
        assert sent_args[0][1] == ["alice@test.com", "bob@test.com"]


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
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value="# Test Briefing\nLooks good!",
        )

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )

        await run_morning_briefing(config, dry_run=True)

        captured = capsys.readouterr()
        assert "Test Briefing" in captured.out

    @pytest.mark.asyncio
    async def test_no_ai_mode_skips_generation(self, mocker):
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
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )

        await run_morning_briefing(config, dry_run=False, no_ai=True)

        mock_generate.assert_not_called()
        mock_raw_email.assert_called_once()

    @pytest.mark.asyncio
    async def test_ai_failure_uses_fallback(self, mocker):
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
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )

        await run_morning_briefing(config, dry_run=False, no_ai=False)

        mock_send.assert_called_once()
        body = mock_send.call_args.kwargs.get("body_markdown", "")
        assert "Fallback" in body
