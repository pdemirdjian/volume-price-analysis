"""Email delivery for morning briefings."""

import json
import logging
import re
import smtplib
import ssl
from collections.abc import Collection
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown  # type: ignore[import-untyped]
import nh3

logger = logging.getLogger(__name__)


def _parse_recipients(to_addr: str) -> list[str]:
    """Split a comma-separated address string into individual recipients."""
    recipients = [addr.strip() for addr in to_addr.split(",") if addr.strip()]
    if not recipients:
        raise ValueError("No valid recipient email addresses provided.")
    return recipients


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
    """
    Send a briefing email with both plain text and HTML parts.

    Args:
        subject: Email subject line.
        body_markdown: Briefing content in markdown format.
        from_addr: Sender email address.
        password: Sender email password (app-specific password for Gmail).
        to_addr: Comma-separated recipient email address(es).
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
        ticker_symbols: Symbols to wrap in TradingView chart links in the
            HTML part. None or empty means no linkification.
    """
    recipients = _parse_recipients(to_addr)

    # Sanitize subject to prevent SMTP header injection
    subject = subject.replace("\r", "").replace("\n", "")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(recipients)

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

    logger.info("Sending briefing email to %s via %s:%d", recipients, smtp_host, smtp_port)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(from_addr, password)
            server.sendmail(from_addr, recipients, msg.as_string())
        logger.info("Email sent successfully")
    except smtplib.SMTPException:
        logger.exception("Failed to send briefing email")
        raise


def send_error_email(
    error_message: str,
    from_addr: str,
    password: str,
    to_addr: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> None:
    """Send an error notification email when the briefing fails critically."""
    subject = "Morning Briefing - ERROR"
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

    try:
        recipients = _parse_recipients(to_addr)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(from_addr, password)
            server.sendmail(from_addr, recipients, msg.as_string())
        logger.info("Error notification email sent")
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
    """Send raw scan/analysis data as email (for --no-ai mode).

    ``preamble`` is an optional markdown block (e.g. the market-regime verdict)
    placed above the raw JSON dumps.
    """
    subject = f"Morning Market Data (Raw) - {date_str}"

    body = f"{preamble}\n\n" if preamble else ""
    body += "# Morning Market Scan Results\n\n"
    body += f"```json\n{json.dumps(scan_results, indent=2, default=str)}\n```\n\n"

    if deep_analyses:
        body += "# Deep Analysis Results\n\n"
        for analysis in deep_analyses:
            symbol = analysis.get("symbol", "Unknown")
            body += f"## {symbol}\n"
            body += f"```json\n{json.dumps(analysis, indent=2, default=str)}\n```\n\n"

    send_briefing_email(
        subject=subject,
        body_markdown=body,
        from_addr=from_addr,
        password=password,
        to_addr=to_addr,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
    )
