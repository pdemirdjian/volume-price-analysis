"""Email delivery for morning briefings."""

import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def _parse_recipients(to_addr: str) -> list[str]:
    """Split a comma-separated address string into individual recipients."""
    recipients = [addr.strip() for addr in to_addr.split(",") if addr.strip()]
    if not recipients:
        raise ValueError("No valid recipient email addresses provided.")
    return recipients


def send_briefing_email(
    subject: str,
    body_markdown: str,
    from_addr: str,
    password: str,
    to_addr: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> None:
    """
    Send a briefing email with both plain text and HTML parts.

    Args:
        subject: Email subject line.
        body_markdown: Briefing content in markdown format.
        from_addr: Sender email address.
        password: Sender email password (app-specific password for Gmail).
        to_addr: Recipient email address.
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    # Plain text part
    msg.attach(MIMEText(body_markdown, "plain"))

    # HTML part (convert markdown to HTML)
    html_body = markdown.markdown(
        body_markdown,
        extensions=["tables", "fenced_code"],
    )
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

    recipients = _parse_recipients(to_addr)

    logger.info("Sending briefing email to %s via %s:%d", recipients, smtp_host, smtp_port)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
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
    body = f"The morning briefing agent encountered a critical error:\n\n{error_message}"

    recipients = _parse_recipients(to_addr)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
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
) -> None:
    """Send raw scan/analysis data as email (for --no-ai mode)."""
    subject = f"Morning Market Data (Raw) - {date_str}"

    body = "# Morning Market Scan Results\n\n"
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
