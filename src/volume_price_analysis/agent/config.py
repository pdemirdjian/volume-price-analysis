"""Configuration for the morning briefing agent via environment variables."""

import logging
import os
from dataclasses import dataclass

from ..analysis import get_universes
from ..data_fetcher import VALID_PERIODS
from ..tools import TOOLS_BY_NAME, validate_arguments

logger = logging.getLogger(__name__)

# Upper bound for MAX_DEEP_ANALYSIS: each candidate costs a data fetch, an
# options analysis, and an earnings lookup, so an unbounded value would hammer
# Yahoo Finance and spawn an equally unbounded thread pool.
MAX_DEEP_ANALYSIS_CAP = 20


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
    scan_period: str = "3mo"
    scan_holding_period: int = 14
    scan_min_score: float = 2.0
    scan_min_adx: float = 20
    scan_max_iv_percentile: float = 70
    scan_min_avg_daily_volume: float = 500_000
    scan_max_results: int = 15

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
        else:
            if max_deep < 1:
                logger.warning("MAX_DEEP_ANALYSIS=%d must be at least 1, using default 5", max_deep)
                max_deep = 5
            elif max_deep > MAX_DEEP_ANALYSIS_CAP:
                logger.warning(
                    "MAX_DEEP_ANALYSIS=%d exceeds cap, clamping to %d",
                    max_deep,
                    MAX_DEEP_ANALYSIS_CAP,
                )
                max_deep = MAX_DEEP_ANALYSIS_CAP

        def scan_number(name: str, default: int | float, kind: type[int] | type[float]):
            key = f"SCAN_{name.upper()}"
            raw = os.environ.get(key, str(default))
            try:
                return kind(raw)
            except ValueError:
                raise ValueError(f"{key}={raw!r} must be a valid {kind.__name__}") from None

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
            scan_period=os.environ.get("SCAN_PERIOD", "3mo"),
            scan_holding_period=scan_number("holding_period", 14, int),
            scan_min_score=scan_number("min_score", 2.0, float),
            scan_min_adx=scan_number("min_adx", 20, float),
            scan_max_iv_percentile=scan_number("max_iv_percentile", 70, float),
            scan_min_avg_daily_volume=scan_number("min_avg_daily_volume", 500_000, float),
            scan_max_results=scan_number("max_results", 15, int),
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

        # Validate scan settings (from_env clamps, but configs can be built directly)
        if self.max_deep_analysis < 1:
            errors.append(f"max_deep_analysis={self.max_deep_analysis} must be at least 1")
        elif self.max_deep_analysis > MAX_DEEP_ANALYSIS_CAP:
            errors.append(
                f"max_deep_analysis={self.max_deep_analysis} exceeds cap {MAX_DEEP_ANALYSIS_CAP}"
            )
        if self.scan_universe not in get_universes():
            errors.append(f"SCAN_UNIVERSE={self.scan_universe!r} is unknown")
        if self.scan_period not in VALID_PERIODS:
            errors.append(f"SCAN_PERIOD={self.scan_period!r} is invalid")
        schema = TOOLS_BY_NAME["scan_candidates"].input_schema
        for name in (
            "holding_period",
            "min_score",
            "min_adx",
            "max_iv_percentile",
            "min_avg_daily_volume",
            "max_results",
        ):
            try:
                validate_arguments({name: getattr(self, f"scan_{name}")}, schema)
            except ValueError as exc:
                errors.append(f"SCAN_{name.upper()}: {exc}")
        return errors
