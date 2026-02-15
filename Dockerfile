FROM python:3.14-slim

# Install supercronic (lightweight cron for containers)
# - Logs to stdout/stderr (works with docker logs)
# - No syslog dependency
# - Graceful signal handling for container stops
# - Supports env vars from Docker
# renovate: datasource=github-releases depName=aptible/supercronic
ARG SUPERCRONIC_VERSION=v0.2.33
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -fsSL "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-amd64" -o /usr/local/bin/supercronic && \
    chmod +x /usr/local/bin/supercronic && \
    apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen

# Copy application code
COPY src/ src/
COPY docker/crontab /app/crontab

# Default: run supercronic with the crontab
CMD ["supercronic", "/app/crontab"]
