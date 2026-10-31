"""Redaction of provider error messages before they leave the process.

Upstream (yfinance) exception text can embed request URLs, query parameters and
key-like values. Scan output and the agent's error email share this one rule.
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+")
_SECRET_RE = re.compile(r"(?i)(key|token|password|secret|credential)[=:]\s*\S+")


def redact_error_message(message: str) -> str:
    """Replace URLs and key/token-like values in ``message`` with placeholders."""
    message = _URL_RE.sub("[URL redacted]", message)
    return _SECRET_RE.sub(r"\1=[REDACTED]", message)
