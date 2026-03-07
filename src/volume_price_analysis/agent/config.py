"""Configuration for the morning briefing agent via environment variables."""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration loaded from environment variables."""

    # AI provider ("gemini" or "anthropic")
    ai_provider: str = "gemini"
    ai_provider_api_key: str = ""
    ai_model: str = ""  # Empty = use provider default

    # Email
    email_from: str = ""
    email_password: str = ""
    email_to: str = ""
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587

    # Scan settings
    scan_universe: str = "full_market"
    max_deep_analysis: int = 5

    def __repr__(self) -> str:
        return (
            f"AgentConfig(ai_provider={self.ai_provider!r}, "
            f"ai_provider_api_key='***', ai_model={self.ai_model!r}, "
            f"email_from={self.email_from!r}, email_password='***', "
            f"email_to={self.email_to!r}, email_smtp_host={self.email_smtp_host!r}, "
            f"email_smtp_port={self.email_smtp_port!r}, "
            f"scan_universe={self.scan_universe!r}, "
            f"max_deep_analysis={self.max_deep_analysis!r})"
        )

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Load configuration from environment variables."""
        smtp_port = 587
        raw = os.environ.get("EMAIL_SMTP_PORT", "587")
        try:
            smtp_port = int(raw)
        except ValueError:
            logger.warning("EMAIL_SMTP_PORT=%r is not a valid integer, using default 587", raw)

        max_deep = 5
        raw = os.environ.get("MAX_DEEP_ANALYSIS", "5")
        try:
            max_deep = int(raw)
        except ValueError:
            logger.warning("MAX_DEEP_ANALYSIS=%r is not a valid integer, using default 5", raw)

        return cls(
            ai_provider=os.environ.get("AI_PROVIDER", "gemini").lower(),
            ai_provider_api_key=os.environ.get("AI_PROVIDER_API_KEY", ""),
            ai_model=os.environ.get("AI_MODEL", ""),
            email_from=os.environ.get("EMAIL_FROM", ""),
            email_password=os.environ.get("EMAIL_PASSWORD", ""),
            email_to=os.environ.get("EMAIL_TO", ""),
            email_smtp_host=os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com"),
            email_smtp_port=smtp_port,
            scan_universe=os.environ.get("SCAN_UNIVERSE", "full_market"),
            max_deep_analysis=max_deep,
        )

    def validate(self) -> list[str]:
        """Validate required configuration. Returns list of error messages."""
        errors = []

        # Validate AI provider
        if self.ai_provider not in ("gemini", "anthropic"):
            errors.append(
                f"AI_PROVIDER={self.ai_provider!r} is invalid. Use 'gemini' or 'anthropic'."
            )
        elif not self.ai_provider_api_key:
            errors.append("AI_PROVIDER_API_KEY is required")

        # Validate email
        if not self.email_from:
            errors.append("EMAIL_FROM is required")
        elif "@" not in self.email_from or "." not in self.email_from.split("@")[-1]:
            errors.append(f"EMAIL_FROM={self.email_from!r} is not a valid email address")
        if not self.email_password:
            errors.append("EMAIL_PASSWORD is required")
        if not self.email_to:
            errors.append("EMAIL_TO is required")
        else:
            for addr in (a.strip() for a in self.email_to.split(",") if a.strip()):
                if "@" not in addr or "." not in addr.split("@")[-1]:
                    errors.append(f"EMAIL_TO address {addr!r} is not a valid email address")
        return errors
