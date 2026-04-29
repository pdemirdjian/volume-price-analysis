# ── Stage 1: build ──
FROM python:3.14.4-slim@sha256:5c9b9aeee369d854acd666375ebb2a9a45a8ac9d264973bed69dbb28cf48f648 AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.8@sha256:3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347 /uv /usr/local/bin/uv

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen --no-install-project

# Copy source and install the project (non-editable)
COPY src/ src/
RUN uv sync --no-dev --frozen --no-editable

# ── Stage 2: runtime ──
FROM python:3.14.4-slim@sha256:5c9b9aeee369d854acd666375ebb2a9a45a8ac9d264973bed69dbb28cf48f648

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
