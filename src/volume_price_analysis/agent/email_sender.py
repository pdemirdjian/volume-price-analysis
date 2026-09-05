"""Email delivery for morning briefings.

The module is split into pure message builders (``build_*_message``) and a
single transport function (:func:`send_email`). The three ``send_*_email``
functions are thin build-and-send wrappers kept for existing call sites.
"""

from __future__ import annotations

import json
import logging
import re
import smtplib
import ssl
from collections.abc import Callable, Collection
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown  # type: ignore[import-untyped]
import nh3

from .config import AgentConfig

logger = logging.getLogger(__name__)

SmtpFactory = Callable[..., smtplib.SMTP]


def _parse_recipients(to_addr: str) -> list[str]:
    """Split a comma-separated address string into individual recipients."""
    recipients = [addr.strip() for addr in to_addr.split(",") if addr.strip()]
    if not recipients:
        raise ValueError("No valid recipient email addresses provided.")
    return recipients


@dataclass(frozen=True)
class SmtpCreds:
    """Everything needed to address and deliver one message."""

    from_addr: str
    password: str
    to_addrs: list[str]
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587

    @classmethod
    def from_config(cls, config: AgentConfig) -> SmtpCreds:
        """Build credentials from an :class:`AgentConfig`."""
        return cls(
            from_addr=config.email_from,
            password=config.email_password,
            to_addrs=_parse_recipients(config.email_to),
            smtp_host=config.email_smtp_host,
            smtp_port=config.email_smtp_port,
        )

    @classmethod
    def from_parts(
        cls,
        from_addr: str,
        password: str,
        to_addr: str,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
    ) -> SmtpCreds:
        """Build credentials from a comma-separated recipient string."""
        return cls(
            from_addr=from_addr,
            password=password,
            to_addrs=_parse_recipients(to_addr),
            smtp_host=smtp_host,
            smtp_port=smtp_port,
        )


_TRADINGVIEW_BASE = "https://www.tradingview.com/chart/?symbol="


def _linkify_tickers(html: str, symbols: Collection[str] | None = None) -> str:
    """Wrap known ticker symbols in TradingView chart links.

    Only exact matches from `symbols` (the actual scan/analysis candidates)
    are linked — heuristic all-caps matching mislinks indicator acronyms
    (RSI, ADX), option strikes (430C), and 6+ letter words (EXTREME → EXTRE).
    With no symbols, the HTML is returned unchanged.
    """
    if not symbols:
        return html

    # Longest-first so overlapping symbols (e.g. "GOOG" vs "GOOGL") match fully.
    # Boundaries reject adjacent alphanumerics: "430C" must not link a "C" candidate.
    alternation = "|".join(re.escape(s) for s in sorted(symbols, key=len, reverse=True))
    ticker_re = re.compile(rf"(?<![A-Za-z0-9<>])({alternation})(?![A-Za-z0-9>])")

    def replace(m: re.Match[str]) -> str:
        ticker = m.group(1)
        url = f"{_TRADINGVIEW_BASE}{ticker}"
        return f'<a href="{url}" rel="noopener noreferrer" target="_blank">{ticker}</a>'

    # Only process text nodes — skip tag content and anything inside <a>…</a>.
    parts = re.split(r"(<[^>]+>)", html)
    result = []
    in_anchor = 0  # nesting depth: >0 means inside an <a> tag
    for part in parts:
        if part.startswith("<"):
            tag_lower = part.lower()
            if tag_lower.startswith("<a ") or tag_lower == "<a>":
                in_anchor += 1
            elif tag_lower.startswith("</a"):
                in_anchor = max(0, in_anchor - 1)
            result.append(part)
        elif in_anchor:
            result.append(part)
        else:
            result.append(ticker_re.sub(replace, part))
    return "".join(result)


def _new_message(subject: str, creds: SmtpCreds) -> MIMEMultipart:
    """Create an addressed multipart/alternative shell with a safe subject."""
    msg = MIMEMultipart("alternative")
    # Sanitize subject to prevent SMTP header injection
    msg["Subject"] = subject.replace("\r", "").replace("\n", "")
    msg["From"] = creds.from_addr
    msg["To"] = ", ".join(creds.to_addrs)
    return msg


def build_briefing_message(
    creds: SmtpCreds,
    subject: str,
    body_markdown: str,
    ticker_symbols: Collection[str] | None = None,
) -> MIMEMultipart:
    """Build the briefing email with both plain text and HTML parts.

    Args:
        creds: Sender/recipient addressing (transport fields are unused here).
        subject: Email subject line.
        body_markdown: Briefing content in markdown format.
        ticker_symbols: Symbols to wrap in TradingView chart links in the
            HTML part. None or empty means no linkification.
    """
    msg = _new_message(subject, creds)

    # Plain text part
    msg.attach(MIMEText(body_markdown, "plain"))

    # HTML part (convert markdown to HTML, sanitize, then linkify tickers)
    html_body = nh3.clean(markdown.markdown(body_markdown, extensions=["tables", "fenced_code"]))
    html_body = _linkify_tickers(html_body, ticker_symbols)
    # Wrap in minimal styling for email clients
    html_full = f"""\
<html>
<head>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 8px; }}
h2 {{ color: #16213e; }}
h3 {{ color: #0f3460; }}
code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
pre {{ background: #f5f5f5; padding: 12px; border-radius: 6px; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #16213e; color: white; }}
strong {{ color: #0f3460; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
    msg.attach(MIMEText(html_full, "html"))
    return msg


def build_error_message(creds: SmtpCreds, error_message: str) -> MIMEMultipart:
    """Build the plain-text error notification sent when a briefing fails."""
    # Truncate and sanitize error message to avoid leaking sensitive details
    safe_message = error_message[:500] if error_message else "Unknown error"
    # Strip potential secrets: URLs with query params, key-like strings
    safe_message = re.sub(r"https?://\S+", "[URL redacted]", safe_message)
    safe_message = re.sub(
        r"(?i)(key|token|password|secret|credential)[=:]\s*\S+", r"\1=[REDACTED]", safe_message
    )
    body = (
        f"The morning briefing agent encountered a critical error:\n\n"
        f"{safe_message}\n\n"
        f"Check server logs for full details."
    )

    msg = _new_message("Morning Briefing - ERROR", creds)
    msg.attach(MIMEText(body, "plain"))
    return msg


def build_raw_data_message(
    creds: SmtpCreds,
    scan_results: dict,
    deep_analyses: list[dict],
    date_str: str = "",
    preamble: str = "",
) -> MIMEMultipart:
    """Build the raw scan/analysis data email (for --no-ai mode).

    ``preamble`` is an optional markdown block (e.g. the market-regime verdict)
    placed above the raw JSON dumps.
    """
    body = f"{preamble}\n\n" if preamble else ""
    body += "# Morning Market Scan Results\n\n"
    body += f"```json\n{json.dumps(scan_results, indent=2, default=str)}\n```\n\n"

    if deep_analyses:
        body += "# Deep Analysis Results\n\n"
        for analysis in deep_analyses:
            symbol = analysis.get("symbol", "Unknown")
            body += f"## {symbol}\n"
            body += f"```json\n{json.dumps(analysis, indent=2, default=str)}\n```\n\n"

    return build_briefing_message(
        creds,
        subject=f"Morning Market Data (Raw) - {date_str}",
        body_markdown=body,
    )


def send_email(
    message: MIMEMultipart,
    creds: SmtpCreds,
    *,
    smtp_factory: SmtpFactory = smtplib.SMTP,
) -> None:
    """Deliver a built message over SMTP.

    ``smtp_factory`` is the injection point for tests; it must return a
    context-manager SMTP client.
    """
    logger.info("Sending email to %s via %s:%d", creds.to_addrs, creds.smtp_host, creds.smtp_port)
    try:
        with smtp_factory(creds.smtp_host, creds.smtp_port) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(creds.from_addr, creds.password)
            server.sendmail(creds.from_addr, creds.to_addrs, message.as_string())
        logger.info("Email sent successfully")
    except smtplib.SMTPException:
        logger.exception("Failed to send email")
        raise


def send_briefing_email(
    subject: str,
    body_markdown: str,
    from_addr: str,
    password: str,
    to_addr: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
    ticker_symbols: Collection[str] | None = None,
) -> None:
    """Build and send a briefing email. See :func:`build_briefing_message`."""
    creds = SmtpCreds.from_parts(from_addr, password, to_addr, smtp_host, smtp_port)
    send_email(build_briefing_message(creds, subject, body_markdown, ticker_symbols), creds)


def send_error_email(
    error_message: str,
    from_addr: str,
    password: str,
    to_addr: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> None:
    """Send an error notification email when the briefing fails critically.

    Never raises: a failure here must not mask the original error.
    """
    try:
        creds = SmtpCreds.from_parts(from_addr, password, to_addr, smtp_host, smtp_port)
        send_email(build_error_message(creds, error_message), creds)
    except Exception:
        logger.exception("Failed to send error notification email")


def send_raw_data_email(
    scan_results: dict,
    deep_analyses: list[dict],
    from_addr: str,
    password: str,
    to_addr: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
    date_str: str = "",
    preamble: str = "",
) -> None:
    """Build and send the raw data email. See :func:`build_raw_data_message`."""
    creds = SmtpCreds.from_parts(from_addr, password, to_addr, smtp_host, smtp_port)
    send_email(
        build_raw_data_message(creds, scan_results, deep_analyses, date_str, preamble), creds
    )
