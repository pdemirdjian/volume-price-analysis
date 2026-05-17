# ── Stage 1: build ──
FROM python:3.14.5-slim@sha256:7a500125bc50693f2214e842a621440a1b1b9cbb2188f74ab045d29ed2ea5856 AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.13@sha256:841c8e6fe30a8b07b4478d12d0c608cba6de66102d29d65d1cc423af86051563 /uv /usr/local/bin/uv

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen --no-install-project

# Copy source and install the project (non-editable)
COPY src/ src/
RUN uv sync --no-dev --frozen --no-editable

# ── Stage 2: runtime ──
FROM python:3.14.5-slim@sha256:7a500125bc50693f2214e842a621440a1b1b9cbb2188f74ab045d29ed2ea5856

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
