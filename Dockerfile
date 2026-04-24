# ── Stage 1: build ──
FROM python:3.14-slim@sha256:3989a23fd2c28a34c7be819e488b958a10601d421ac25bea1e7a5d757365e2d5 AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /usr/local/bin/uv

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen --no-install-project

# Copy source and install the project (non-editable)
COPY src/ src/
RUN uv sync --no-dev --frozen --no-editable

# ── Stage 2: runtime ──
FROM python:3.14-slim@sha256:3989a23fd2c28a34c7be819e488b958a10601d421ac25bea1e7a5d757365e2d5

WORKDIR /app
COPY --from=builder /app/.venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Patch OS-level vulnerabilities from the base image
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Remove system pip (unused — uv manages deps) to fix CVE-2026-1703 and reduce attack surface
RUN /usr/local/bin/python -m pip uninstall -y pip

# Run as non-root user
RUN useradd --create-home appuser \
    && mkdir -p /home/appuser/.cache/py-yfinance \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

HEALTHCHECK --interval=300s --timeout=5s --retries=3 \
  CMD python -c "open('/proc/1/cmdline').read()" || exit 1

CMD ["morning-scheduler"]
