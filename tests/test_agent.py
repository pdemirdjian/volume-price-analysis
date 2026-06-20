"""Tests for the morning briefing agent."""

import asyncio
import signal
import smtplib
from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock

import pytest

from volume_price_analysis.agent.ai_client import (
    _TRUNCATION_WARNING,
    SYSTEM_PROMPT,
    _check_briefing_grounding,
    generate_briefing,
)
from volume_price_analysis.agent.config import AgentConfig
from volume_price_analysis.agent.email_sender import (
    _parse_recipients,
    send_briefing_email,
    send_error_email,
    send_raw_data_email,
)
from volume_price_analysis.agent.morning_agent import (
    _fallback_briefing,
    _get_top_symbols,
    main,
    run_morning_briefing,
)
from volume_price_analysis.agent.scheduler import (
    ET,
    _next_run,
    _run_loop,
    run_scheduler,
)
from volume_price_analysis.agent.scheduler import (
    main as scheduler_main,
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


class TestErrorEmailSanitization:
    """Test error email sanitization logic."""

    def test_error_email_truncates_long_message(self, mocker):
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        long_message = "A" * 1000

        send_error_email(
            error_message=long_message,
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
        )

        sent_args = mock_smtp_instance.sendmail.call_args
        msg_str = sent_args[0][2]
        # The 1000-char message should be truncated to 500 chars
        assert "A" * 501 not in msg_str
        assert "A" * 500 in msg_str

    def test_error_email_redacts_urls(self, mocker):
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        send_error_email(
            error_message="Failed at https://api.example.com/key=abc123 during request",
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
        )

        sent_args = mock_smtp_instance.sendmail.call_args
        msg_str = sent_args[0][2]
        assert "https://api.example.com/key=abc123" not in msg_str
        assert "[URL redacted]" in msg_str

    def test_error_email_redacts_secrets(self, mocker):
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        send_error_email(
            error_message="Error: key=sk-12345 and password: mysecret were exposed",
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
        )

        sent_args = mock_smtp_instance.sendmail.call_args
        msg_str = sent_args[0][2]
        assert "sk-12345" not in msg_str
        assert "mysecret" not in msg_str
        assert "[REDACTED]" in msg_str

    def test_error_email_empty_message(self, mocker):
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        send_error_email(
            error_message="",
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
        )

        sent_args = mock_smtp_instance.sendmail.call_args
        msg_str = sent_args[0][2]
        assert "Unknown error" in msg_str


class TestHtmlSanitization:
    """Test that nh3 HTML sanitization strips dangerous tags in briefing emails."""

    def test_briefing_email_sanitizes_html(self, mocker):
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        # Markdown that contains an XSS script tag
        malicious_markdown = "# Hello\n\n<script>alert('xss')</script>\n\nSafe content here."

        send_briefing_email(
            subject="Test Briefing",
            body_markdown=malicious_markdown,
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
        )

        sent_args = mock_smtp_instance.sendmail.call_args
        msg_str = sent_args[0][2]

        # Extract the HTML part from the multipart message (after "Content-Type: text/html")
        html_boundary = msg_str.split("Content-Type: text/html")
        assert len(html_boundary) > 1, "Expected an HTML part in the multipart email"
        html_part = html_boundary[1]

        # The script tag must be stripped from the HTML part by nh3
        assert "<script>" not in html_part
        assert "alert('xss')" not in html_part
        # But safe content should remain in the HTML part
        assert "Safe content here" in html_part


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

    def test_filters_empty_recipients_from_trailing_comma(self, mocker):
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
            to_addr="alice@test.com,bob@test.com,",
        )

        sent_args = mock_smtp_instance.sendmail.call_args
        assert sent_args[0][1] == ["alice@test.com", "bob@test.com"]

    def test_raises_on_empty_recipients(self):
        with pytest.raises(ValueError, match="No valid recipient"):
            _parse_recipients("")

        with pytest.raises(ValueError, match="No valid recipient"):
            _parse_recipients(",,")

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


class TestSendBriefingEmailSmtpFailure:
    """Test that send_briefing_email re-raises SMTPException (lines 90-92)."""

    def test_raises_smtp_exception(self, mocker):
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.sendmail.side_effect = smtplib.SMTPException("Connection refused")
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        with pytest.raises(smtplib.SMTPException, match="Connection refused"):
            send_briefing_email(
                subject="Test Briefing",
                body_markdown="# Hello",
                from_addr="sender@test.com",
                password="test-pass",
                to_addr="recipient@test.com",
            )

    def test_logs_smtp_exception(self, mocker, caplog):
        import logging

        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.sendmail.side_effect = smtplib.SMTPException("Auth failed")
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        with caplog.at_level(logging.ERROR, logger="volume_price_analysis.agent.email_sender"):
            with pytest.raises(smtplib.SMTPException):
                send_briefing_email(
                    subject="Test",
                    body_markdown="content",
                    from_addr="a@b.com",
                    password="pass",
                    to_addr="c@d.com",
                )

        assert "Failed to send briefing email" in caplog.text


class TestSendErrorEmailFailure:
    """Test that send_error_email swallows exceptions (lines 132-133)."""

    def test_swallows_smtp_exception(self, mocker):
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.sendmail.side_effect = smtplib.SMTPException("Network error")
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        # Should NOT raise - the exception is caught and logged
        send_error_email(
            error_message="Something broke",
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
        )

    def test_logs_failure_to_send_error_email(self, mocker, caplog):
        import logging

        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.sendmail.side_effect = Exception("Unexpected failure")
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        with caplog.at_level(logging.ERROR, logger="volume_price_analysis.agent.email_sender"):
            send_error_email(
                error_message="Something broke",
                from_addr="sender@test.com",
                password="test-pass",
                to_addr="recipient@test.com",
            )

        assert "Failed to send error notification email" in caplog.text


class TestSendRawDataEmail:
    """Test send_raw_data_email function (lines 147-159)."""

    def test_sends_scan_results_as_json(self, mocker):
        mock_send = mocker.patch(
            "volume_price_analysis.agent.email_sender.send_briefing_email",
        )

        scan_results = {"summary": {"total_candidates": 5, "bullish": 3}}

        send_raw_data_email(
            scan_results=scan_results,
            deep_analyses=[],
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
            date_str="2026-03-02",
        )

        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args.kwargs
        assert call_kwargs["subject"] == "Morning Market Data (Raw) - 2026-03-02"
        assert call_kwargs["from_addr"] == "sender@test.com"
        assert call_kwargs["password"] == "test-pass"
        assert call_kwargs["to_addr"] == "recipient@test.com"
        body = call_kwargs["body_markdown"]
        assert "Morning Market Scan Results" in body
        assert "total_candidates" in body
        assert "5" in body

    def test_includes_deep_analyses(self, mocker):
        mock_send = mocker.patch(
            "volume_price_analysis.agent.email_sender.send_briefing_email",
        )

        scan_results = {"summary": {"total_candidates": 1}}
        deep_analyses = [
            {"symbol": "AAPL", "score": 4.5},
            {"symbol": "MSFT", "score": 3.8},
        ]

        send_raw_data_email(
            scan_results=scan_results,
            deep_analyses=deep_analyses,
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
            date_str="2026-03-02",
        )

        mock_send.assert_called_once()
        body = mock_send.call_args.kwargs["body_markdown"]
        assert "Deep Analysis Results" in body
        assert "## AAPL" in body
        assert "## MSFT" in body
        assert "4.5" in body
        assert "3.8" in body

    def test_no_deep_analysis_section_when_empty(self, mocker):
        mock_send = mocker.patch(
            "volume_price_analysis.agent.email_sender.send_briefing_email",
        )

        send_raw_data_email(
            scan_results={"summary": {}},
            deep_analyses=[],
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
        )

        body = mock_send.call_args.kwargs["body_markdown"]
        assert "Deep Analysis Results" not in body

    def test_handles_unknown_symbol(self, mocker):
        mock_send = mocker.patch(
            "volume_price_analysis.agent.email_sender.send_briefing_email",
        )

        send_raw_data_email(
            scan_results={},
            deep_analyses=[{"score": 2.0}],  # no "symbol" key
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
        )

        body = mock_send.call_args.kwargs["body_markdown"]
        assert "## Unknown" in body

    def test_passes_smtp_params(self, mocker):
        mock_send = mocker.patch(
            "volume_price_analysis.agent.email_sender.send_briefing_email",
        )

        send_raw_data_email(
            scan_results={},
            deep_analyses=[],
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
            smtp_host="custom.smtp.com",
            smtp_port=465,
        )

        call_kwargs = mock_send.call_args.kwargs
        assert call_kwargs["smtp_host"] == "custom.smtp.com"
        assert call_kwargs["smtp_port"] == 465


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


class TestNextRun:
    """Test _next_run scheduling logic."""

    def test_skips_holiday_that_lands_on_weekend(self):
        """Line 76: holiday on Friday -> skip to Saturday -> advance past weekend."""
        # Good Friday 2025-04-18 is a Friday NYSE holiday.
        # Set now to Thursday 2025-04-17 at 9:00 (after 08:30 target).
        tz = ET
        target = time(8, 30)
        now = datetime(2025, 4, 17, 9, 0, tzinfo=tz)

        result = _next_run(target, tz, now=now, skip_holidays=True)

        # Friday 2025-04-18 is a holiday -> skip to Saturday -> skip weekend -> Monday
        assert result.date() == datetime(2025, 4, 21, tzinfo=tz).date()
        assert result.weekday() == 0  # Monday


class TestRunLoop:
    """Test _run_loop scheduling loop."""

    @pytest.mark.asyncio
    async def test_config_validation_failure_exits(self, mocker):
        """Lines 90-93: config validation errors cause SystemExit(1)."""
        mocker.patch(
            "volume_price_analysis.agent.scheduler.AgentConfig.from_env",
            return_value=AgentConfig(ai_provider="gemini"),  # missing required fields
        )
        stop_event = asyncio.Event()

        with pytest.raises(SystemExit):
            await _run_loop(time(8, 30), ET, stop_event)

    @pytest.mark.asyncio
    async def test_holiday_log_with_skip_holidays_true(self, mocker):
        """Lines 103-110: log when skip_holidays=True and next run is on a holiday date."""
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.AgentConfig.from_env",
            return_value=config,
        )

        # Return a date that IS a holiday (Good Friday 2025) to trigger the defensive branch
        holiday_dt = datetime(2025, 4, 18, 8, 30, tzinfo=ET)
        mocker.patch(
            "volume_price_analysis.agent.scheduler._next_run",
            return_value=holiday_dt,
        )

        mock_logger = mocker.patch("volume_price_analysis.agent.scheduler.logger")

        stop_event = asyncio.Event()

        # Schedule stop_event.set() to fire during the await inside the loop body
        async def set_stop_soon():
            await asyncio.sleep(0)
            stop_event.set()

        asyncio.create_task(set_stop_soon())

        await _run_loop(time(8, 30), ET, stop_event, skip_holidays=True)

        # Check that the "skipping" log message was emitted
        log_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("skipping" in c.lower() for c in log_calls)

    @pytest.mark.asyncio
    async def test_holiday_log_with_skip_holidays_false(self, mocker):
        """Lines 111-116: log 'still running' when skip_holidays=False on a holiday."""
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.AgentConfig.from_env",
            return_value=config,
        )

        # Return a holiday date
        holiday_dt = datetime(2025, 4, 18, 8, 30, tzinfo=ET)
        mocker.patch(
            "volume_price_analysis.agent.scheduler._next_run",
            return_value=holiday_dt,
        )

        mock_logger = mocker.patch("volume_price_analysis.agent.scheduler.logger")

        stop_event = asyncio.Event()

        async def set_stop_soon():
            await asyncio.sleep(0)
            stop_event.set()

        asyncio.create_task(set_stop_soon())

        await _run_loop(time(8, 30), ET, stop_event, skip_holidays=False)

        log_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("still running" in c.lower() for c in log_calls)

    @pytest.mark.asyncio
    async def test_stop_event_breaks_loop(self, mocker):
        """Line 127: stop_event set during wait_for causes break."""
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.AgentConfig.from_env",
            return_value=config,
        )

        stop_event = asyncio.Event()

        # Schedule stop to fire during the await inside the loop
        async def set_stop_soon():
            await asyncio.sleep(0)
            stop_event.set()

        asyncio.create_task(set_stop_soon())

        # Should complete without hanging
        await _run_loop(time(8, 30), ET, stop_event)

    @pytest.mark.asyncio
    async def test_successful_briefing_run(self, mocker):
        """Line 135: successful briefing log message."""
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.AgentConfig.from_env",
            return_value=config,
        )

        # Make _next_run return a time in the past so delay=0 and timer expires immediately
        past_dt = datetime(2020, 1, 6, 8, 30, tzinfo=ET)  # Monday
        mocker.patch(
            "volume_price_analysis.agent.scheduler._next_run",
            return_value=past_dt,
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.datetime",
            wraps=datetime,
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.datetime.now",
            return_value=datetime(2020, 1, 6, 9, 0, tzinfo=ET),
        )

        stop_event = asyncio.Event()

        # After briefing runs, set stop to exit loop on next iteration
        async def briefing_side_effect(cfg):
            stop_event.set()
            return True

        mock_run_briefing = AsyncMock(side_effect=briefing_side_effect)
        mocker.patch(
            "volume_price_analysis.agent.scheduler.run_morning_briefing",
            mock_run_briefing,
        )

        mock_logger = mocker.patch("volume_price_analysis.agent.scheduler.logger")

        await _run_loop(time(8, 30), ET, stop_event)

        mock_run_briefing.assert_called_once_with(config)
        log_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("completed successfully" in c.lower() for c in log_calls)

    @pytest.mark.asyncio
    async def test_briefing_failure_sends_error_email(self, mocker):
        """Lines 136-148: briefing failure triggers error email."""
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.AgentConfig.from_env",
            return_value=config,
        )

        past_dt = datetime(2020, 1, 6, 8, 30, tzinfo=ET)
        mocker.patch(
            "volume_price_analysis.agent.scheduler._next_run",
            return_value=past_dt,
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.datetime",
            wraps=datetime,
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.datetime.now",
            return_value=datetime(2020, 1, 6, 9, 0, tzinfo=ET),
        )

        stop_event = asyncio.Event()

        async def briefing_side_effect(cfg):
            stop_event.set()
            raise RuntimeError("Briefing exploded")

        mock_run_briefing = AsyncMock(side_effect=briefing_side_effect)
        mocker.patch(
            "volume_price_analysis.agent.scheduler.run_morning_briefing",
            mock_run_briefing,
        )

        mock_send_error = mocker.patch(
            "volume_price_analysis.agent.scheduler.send_error_email",
        )

        await _run_loop(time(8, 30), ET, stop_event)

        mock_send_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_email_failure_is_logged(self, mocker):
        """Lines 149-150: exception when sending error email is logged."""
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.AgentConfig.from_env",
            return_value=config,
        )

        past_dt = datetime(2020, 1, 6, 8, 30, tzinfo=ET)
        mocker.patch(
            "volume_price_analysis.agent.scheduler._next_run",
            return_value=past_dt,
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.datetime",
            wraps=datetime,
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.datetime.now",
            return_value=datetime(2020, 1, 6, 9, 0, tzinfo=ET),
        )

        stop_event = asyncio.Event()

        async def briefing_side_effect(cfg):
            stop_event.set()
            raise RuntimeError("Briefing exploded")

        mock_run_briefing = AsyncMock(side_effect=briefing_side_effect)
        mocker.patch(
            "volume_price_analysis.agent.scheduler.run_morning_briefing",
            mock_run_briefing,
        )

        mocker.patch(
            "volume_price_analysis.agent.scheduler.send_error_email",
            side_effect=Exception("SMTP down"),
        )

        mock_logger = mocker.patch("volume_price_analysis.agent.scheduler.logger")

        await _run_loop(time(8, 30), ET, stop_event)

        # logger.exception should have been called for the error email failure
        log_calls = [str(c) for c in mock_logger.exception.call_args_list]
        assert any("error email" in c.lower() for c in log_calls)


class TestRunScheduler:
    """Test run_scheduler signal handling setup."""

    @pytest.mark.asyncio
    async def test_unix_signal_handler(self, mocker):
        """Lines 172-173: Unix signal handler sets stop_event."""
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.AgentConfig.from_env",
            return_value=config,
        )

        # Track the signal handler that gets registered
        registered_handlers = {}
        loop = asyncio.get_running_loop()

        def capture_handler(sig, handler):
            registered_handlers[sig] = handler

        mocker.patch.object(loop, "add_signal_handler", side_effect=capture_handler)

        # Mock _run_loop to not actually run
        mock_run_loop = AsyncMock()
        mocker.patch("volume_price_analysis.agent.scheduler._run_loop", mock_run_loop)

        # Force non-win32 path
        mock_sys = mocker.patch("volume_price_analysis.agent.scheduler.sys")
        mock_sys.platform = "linux"

        await run_scheduler(time(8, 30), ET)

        # Verify signal handlers were registered for SIGTERM and SIGINT
        assert signal.SIGTERM in registered_handlers
        assert signal.SIGINT in registered_handlers

        # Call the handler to exercise lines 172-173
        handler = registered_handlers[signal.SIGTERM]
        handler()

    @pytest.mark.asyncio
    async def test_win32_signal_handler(self, mocker):
        """Lines 165-166: Windows signal handler uses call_soon_threadsafe."""
        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )
        mocker.patch(
            "volume_price_analysis.agent.scheduler.AgentConfig.from_env",
            return_value=config,
        )

        # Mock _run_loop to not actually run
        mock_run_loop = AsyncMock()
        mocker.patch("volume_price_analysis.agent.scheduler._run_loop", mock_run_loop)

        # Force win32 path
        mock_sys = mocker.patch("volume_price_analysis.agent.scheduler.sys")
        mock_sys.platform = "win32"

        # Track signal.signal calls
        registered_handlers = {}

        def capture_signal(sig, handler):
            registered_handlers[sig] = handler

        mocker.patch(
            "volume_price_analysis.agent.scheduler.signal.signal",
            side_effect=capture_signal,
        )

        await run_scheduler(time(8, 30), ET)

        # On win32 path, signal.signal should have been called for SIGINT
        assert signal.SIGINT in registered_handlers

        # Call the handler to exercise lines 165-166
        handler = registered_handlers[signal.SIGINT]
        handler(signal.SIGINT, None)


class TestSchedulerMain:
    """Test the scheduler main() CLI entry point."""

    def test_default_time(self, mocker):
        """Lines 192-220: main() parses args and calls asyncio.run."""
        mocker.patch("sys.argv", ["morning-scheduler"])
        mock_run = mocker.patch("volume_price_analysis.agent.scheduler.asyncio.run")

        scheduler_main()

        mock_run.assert_called_once()
        # It should be a coroutine from run_scheduler
        call_args = mock_run.call_args[0][0]
        assert asyncio.iscoroutine(call_args)
        call_args.close()  # Clean up unawaited coroutine

    def test_custom_time(self, mocker):
        """Lines 207-215: main() parses custom --time argument."""
        mocker.patch("sys.argv", ["morning-scheduler", "--time", "10:45"])
        mock_run = mocker.patch("volume_price_analysis.agent.scheduler.asyncio.run")

        scheduler_main()

        mock_run.assert_called_once()
        coro = mock_run.call_args[0][0]
        coro.close()

    def test_skip_holidays_flag(self, mocker):
        """Lines 198-202: main() parses --skip-holidays flag."""
        mocker.patch("sys.argv", ["morning-scheduler", "--skip-holidays"])
        mock_run = mocker.patch("volume_price_analysis.agent.scheduler.asyncio.run")

        scheduler_main()

        mock_run.assert_called_once()
        coro = mock_run.call_args[0][0]
        coro.close()

    def test_invalid_time_format_exits(self, mocker):
        """Lines 216-218: invalid --time causes sys.exit(1)."""
        mocker.patch("sys.argv", ["morning-scheduler", "--time", "not-a-time"])
        mock_exit = mocker.patch(
            "volume_price_analysis.agent.scheduler.sys.exit", side_effect=SystemExit(1)
        )

        with pytest.raises(SystemExit):
            scheduler_main()

        mock_exit.assert_called_once_with(1)

    def test_invalid_time_bad_hour(self, mocker):
        """Lines 211-212: hour out of range causes exit."""
        mocker.patch("sys.argv", ["morning-scheduler", "--time", "25:00"])
        mock_exit = mocker.patch(
            "volume_price_analysis.agent.scheduler.sys.exit", side_effect=SystemExit(1)
        )

        with pytest.raises(SystemExit):
            scheduler_main()

        mock_exit.assert_called_once_with(1)

    def test_invalid_time_bad_minute(self, mocker):
        """Lines 213-214: minute out of range causes exit."""
        mocker.patch("sys.argv", ["morning-scheduler", "--time", "08:99"])
        mock_exit = mocker.patch(
            "volume_price_analysis.agent.scheduler.sys.exit", side_effect=SystemExit(1)
        )

        with pytest.raises(SystemExit):
            scheduler_main()

        mock_exit.assert_called_once_with(1)

    def test_invalid_time_missing_colon(self, mocker):
        """Lines 208-209: time without colon separator causes exit."""
        mocker.patch("sys.argv", ["morning-scheduler", "--time", "0830"])
        mock_exit = mocker.patch(
            "volume_price_analysis.agent.scheduler.sys.exit", side_effect=SystemExit(1)
        )

        with pytest.raises(SystemExit):
            scheduler_main()

        mock_exit.assert_called_once_with(1)


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
            "volume_price_analysis.agent.morning_agent.fetch_stock_data",
            side_effect=Exception("Yahoo Finance unavailable"),
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value="# Briefing with no deep data",
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
        await run_morning_briefing(config, dry_run=True)

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
            "volume_price_analysis.agent.morning_agent.fetch_stock_data",
            return_value=MagicMock(),
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.run_options_analysis",
            side_effect=ValueError("Bad data"),
        )
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.generate_briefing",
            return_value="# Briefing after analysis error",
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
            "volume_price_analysis.agent.morning_agent.fetch_stock_data",
            return_value=MagicMock(),
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

        await run_morning_briefing(config, dry_run=True, no_ai=True)

        captured = capsys.readouterr()
        # scan_results JSON printed (line 127)
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

        await run_morning_briefing(config, dry_run=True, no_ai=True)

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
        )

        main()

        mock_run.assert_called_once()

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
        )

        main()

        mock_run.assert_called_once()

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
        )

        main()

        mock_run.assert_called_once()

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
        )

        main()

        mock_run.assert_called_once()

    def test_main_critical_failure_sends_error_email(self, mocker):
        """Critical exception triggers send_error_email and sys.exit(1)."""
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
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.asyncio.run",
            side_effect=RuntimeError("Critical failure"),
        )
        mock_send_error = mocker.patch(
            "volume_price_analysis.agent.morning_agent.send_error_email",
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        mock_send_error.assert_called_once()
        assert "Critical failure" in mock_send_error.call_args.kwargs["error_message"]

    def test_main_critical_failure_dry_run_no_error_email(self, mocker):
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
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.asyncio.run",
            side_effect=RuntimeError("Critical failure"),
        )
        mock_send_error = mocker.patch(
            "volume_price_analysis.agent.morning_agent.send_error_email",
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        mock_send_error.assert_not_called()

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
        """When run_morning_briefing returns False (degraded), main() exits with code 2."""
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
        mocker.patch(
            "volume_price_analysis.agent.morning_agent.asyncio.run",
            return_value=False,
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


class TestSubjectHeaderInjection:
    """Test that SMTP header injection via subject is prevented."""

    def test_newlines_stripped_from_subject(self, mocker):
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch("volume_price_analysis.agent.email_sender.smtplib.SMTP", mock_smtp)

        send_briefing_email(
            subject="Test\r\nBcc: evil@attacker.com\r\nSubject: Injected",
            body_markdown="# Hello",
            from_addr="sender@test.com",
            password="test-pass",
            to_addr="recipient@test.com",
        )

        sent_args = mock_smtp_instance.sendmail.call_args
        msg_str = sent_args[0][2]
        # Newlines removed: no injected Bcc header on its own line
        assert "\nBcc:" not in msg_str
        assert "\r\nBcc:" not in msg_str


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
    """Test that run_morning_briefing returns False when fallback is used."""

    @pytest.mark.asyncio
    async def test_ai_failure_returns_false(self, mocker):
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
        mocker.patch("volume_price_analysis.agent.morning_agent.send_briefing_email")

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
            max_deep_analysis=1,
        )

        result = await run_morning_briefing(config, dry_run=False, no_ai=False)
        assert result is False

    @pytest.mark.asyncio
    async def test_successful_briefing_returns_true(self, mocker):
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
            return_value="# Briefing",
        )
        mocker.patch("volume_price_analysis.agent.morning_agent.send_briefing_email")

        config = AgentConfig(
            ai_provider="gemini",
            ai_provider_api_key="test-key",
            email_from="a@b.com",
            email_password="pass",
            email_to="c@d.com",
        )

        result = await run_morning_briefing(config, dry_run=False, no_ai=False)
        assert result is True


class TestBriefingGrounding:
    """Anti-hallucination grounding check for generated briefings (HOM-38)."""

    @staticmethod
    def _scan():
        return {
            "high_conviction_setups": [{"symbol": "NVDA"}],
            "top_bullish": [{"symbol": "AAPL"}],
            "top_bearish": [{"symbol": "TSLA"}],
            "errors": [{"symbol": "BADX", "error": "no data"}],
        }

    def test_returns_empty_when_all_tickers_known(self):
        briefing = "Top pick: AAPL looks bullish. NVDA strong, TSLA bearish."
        assert _check_briefing_grounding(briefing, self._scan(), []) == []

    def test_flags_hallucinated_ticker(self):
        briefing = "We also like GOOGL here."
        assert "GOOGL" in _check_briefing_grounding(briefing, self._scan(), [])

    def test_ignores_indicator_acronyms(self):
        briefing = (
            "AAPL shows strong RSI and ADX with low IV. Consider 14-day calls. "
            "Watch ATR for stops. Volume confirmed via OBV and MFI; check VWAP."
        )
        assert _check_briefing_grounding(briefing, self._scan(), []) == []

    def test_ignores_options_and_general_acronyms(self):
        briefing = "AAPL 14-day calls (DTE). Slightly OTM. AI sentiment positive. US open."
        assert _check_briefing_grounding(briefing, self._scan(), []) == []

    def test_ignores_macro_and_index_acronyms(self):
        briefing = (
            "Market context: CPI and GDP in focus; FOMC midweek. "
            "VIX low, SPX near resistance. AAPL still the pick."
        )
        assert _check_briefing_grounding(briefing, self._scan(), []) == []

    def test_recognizes_deep_analysis_symbols(self):
        briefing = "MSFT is the standout today."
        assert _check_briefing_grounding(briefing, self._scan(), [{"symbol": "MSFT"}]) == []

    def test_recognizes_errored_symbols(self):
        briefing = "BADX could not be analyzed today."
        assert _check_briefing_grounding(briefing, self._scan(), []) == []

    def test_recognizes_cashtag(self):
        briefing = "$AAPL is bullish."
        assert _check_briefing_grounding(briefing, self._scan(), []) == []

    def test_flags_hallucinated_cashtag(self):
        briefing = "$GOOGL looks great."
        assert "GOOGL" in _check_briefing_grounding(briefing, self._scan(), [])

    def test_ignores_single_capital_letters_in_prose(self):
        briefing = "I think AAPL is a buy. U.S. markets are open."
        assert _check_briefing_grounding(briefing, self._scan(), []) == []

    def test_deduplicates_and_sorts_unknowns(self):
        briefing = "Buy GOOGL and AMZN. GOOGL again, then AMZN."
        assert _check_briefing_grounding(briefing, self._scan(), []) == ["AMZN", "GOOGL"]

    def test_handles_missing_scan_keys(self):
        # errors / high_conviction keys absent should not raise
        briefing = "AAPL looks good."
        scan = {"top_bullish": [{"symbol": "AAPL"}]}
        assert _check_briefing_grounding(briefing, scan, []) == []

    def test_logs_warning_on_mismatch(self, caplog):
        import logging

        briefing = "We like GOOGL and AMZN."
        with caplog.at_level(logging.WARNING, logger="volume_price_analysis.agent.ai_client"):
            _check_briefing_grounding(briefing, self._scan(), [])
        assert "unrecognized" in caplog.text.lower()
        assert "GOOGL" in caplog.text
        assert "AMZN" in caplog.text

    def test_no_warning_when_grounded(self, caplog):
        import logging

        briefing = "AAPL and NVDA are top picks."
        with caplog.at_level(logging.WARNING, logger="volume_price_analysis.agent.ai_client"):
            result = _check_briefing_grounding(briefing, self._scan(), [])
        assert result == []
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_system_prompt_has_grounding_clause(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "only" in lowered
        assert "invent" in lowered or "hallucin" in lowered

    def test_generate_briefing_runs_grounding_check(self, mocker, caplog):
        import logging

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Buy GOOGL now."  # GOOGL is not in the scan data
        mock_response.candidates = None
        mock_client.models.generate_content.return_value = mock_response
        mocker.patch("google.genai.Client", return_value=mock_client)

        with caplog.at_level(logging.WARNING, logger="volume_price_analysis.agent.ai_client"):
            result = generate_briefing(
                scan_results={"top_bullish": [{"symbol": "AAPL"}]},
                deep_analyses=[],
                provider="gemini",
                api_key="test-key",
            )

        assert "GOOGL" in result  # briefing text is never modified
        assert "GOOGL" in caplog.text  # but the mismatch is flagged
