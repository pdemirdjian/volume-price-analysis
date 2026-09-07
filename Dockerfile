# ── Stage 1: build ──
FROM python:3.14.7-slim@sha256:83c1cebb322d099ac9e3a3a532ba74b0146d702838b25e4c75c02fa81ffeb910 AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.21@sha256:ff07b86af50d4d9391d9daf4ff89ce427bc544f9aae87057e69a1cc0aa369946 /uv /usr/local/bin/uv

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen --no-install-project

# Copy source and install the project (non-editable)
COPY src/ src/
RUN uv sync --no-dev --frozen --no-editable

# ── Stage 2: runtime ──
FROM python:3.14.7-slim@sha256:83c1cebb322d099ac9e3a3a532ba74b0146d702838b25e4c75c02fa81ffeb910

WORKDIR /app
# --chown here rather than a later `chown -R`: a recursive chown in a RUN
# layer would re-copy the whole venv into that layer.
COPY --from=builder --chown=10001:10001 /app/.venv .venv
# PYTHONUNBUFFERED: the scheduler is a long-running daemon whose only
# observability is `docker logs`; block-buffered stdout would delay log lines
# indefinitely. PYTHONDONTWRITEBYTECODE: nothing should write into the venv.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Patch OS-level vulnerabilities from the base image (see docs/adr/0001), remove
# system pip (unused — uv manages deps; CVE-2026-1703), and create the non-root
# user, in one layer.
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* \
    && /usr/local/bin/python -m pip uninstall -y pip \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /home/appuser/.cache/py-yfinance \
    && chown appuser:appuser /app \
    && chown -R appuser:appuser /home/appuser/.cache
USER 10001

# Liveness: the scheduler writes a heartbeat file on every wake (<= 15 min
# apart) and around each briefing; the probe fails once it is >45 min stale.
# retries=4 x interval=300s guarantees at least one scheduler wake falls inside
# any failing streak, so a transient stale reading cannot flip the status.
# See src/volume_price_analysis/agent/healthcheck.py.
HEALTHCHECK --interval=300s --timeout=10s --start-period=120s --retries=4 \
  CMD ["morning-healthcheck"]

CMD ["morning-scheduler"]
