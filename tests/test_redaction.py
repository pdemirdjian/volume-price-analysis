"""Tests for the shared error-message redaction rule."""

from volume_price_analysis.redaction import redact_error_message


class TestRedactErrorMessage:
    def test_redacts_urls(self):
        message = "Failed: https://query2.finance.yahoo.com/v8/chart/AAPL?crumb=abc123 returned 404"
        assert redact_error_message(message) == "Failed: [URL redacted] returned 404"

    def test_redacts_key_like_values(self):
        message = "auth failed api_key=sk-12345 token: abcdef"
        redacted = redact_error_message(message)
        assert "sk-12345" not in redacted
        assert "abcdef" not in redacted
        assert redacted.count("[REDACTED]") == 2

    def test_leaves_plain_messages_unchanged(self):
        assert redact_error_message("No data found") == "No data found"
