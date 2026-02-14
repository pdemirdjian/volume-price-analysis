"""Configuration for the morning briefing agent via environment variables."""

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration loaded from environment variables."""

    # Required
    anthropic_api_key: str = ""
    email_from: str = ""
    email_password: str = ""
    email_to: str = ""

    # Optional with defaults
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    scan_universe: str = "full_market"
    max_deep_analysis: int = 5
    claude_model: str = "claude-sonnet-4-5-20250929"
    briefing_cron: str = "30 8 * * 1-5"

    # Derived
    _errors: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Load configuration from environment variables."""
        config = cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            email_from=os.environ.get("EMAIL_FROM", ""),
            email_password=os.environ.get("EMAIL_PASSWORD", ""),
            email_to=os.environ.get("EMAIL_TO", ""),
            email_smtp_host=os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com"),
            email_smtp_port=int(os.environ.get("EMAIL_SMTP_PORT", "587")),
            scan_universe=os.environ.get("SCAN_UNIVERSE", "full_market"),
            max_deep_analysis=int(os.environ.get("MAX_DEEP_ANALYSIS", "5")),
            claude_model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
            briefing_cron=os.environ.get("BRIEFING_CRON", "30 8 * * 1-5"),
        )
        return config

    def validate(self) -> list[str]:
        """Validate required configuration. Returns list of error messages."""
        errors = []
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")
        if not self.email_from:
            errors.append("EMAIL_FROM is required")
        if not self.email_password:
            errors.append("EMAIL_PASSWORD is required")
        if not self.email_to:
            errors.append("EMAIL_TO is required")
        return errors
