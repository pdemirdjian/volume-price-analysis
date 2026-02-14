FROM python:3.14-slim

# Install supercronic (lightweight cron for containers)
# - Logs to stdout/stderr (works with docker logs)
# - No syslog dependency
# - Graceful signal handling for container stops
# - Supports env vars from Docker
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    SUPERCRONIC_VERSION="v0.2.33" && \
    SUPERCRONIC_BASE_URL="https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}" && \
    curl -fsSL "${SUPERCRONIC_BASE_URL}/supercronic-linux-amd64" -o /usr/local/bin/supercronic && \
    curl -fsSL "${SUPERCRONIC_BASE_URL}/supercronic-linux-amd64.sha256" -o /tmp/supercronic.sha256 && \
    (cd /usr/local/bin && sha256sum -c /tmp/supercronic.sha256) && \
    rm /tmp/supercronic.sha256 && \
    chmod +x /usr/local/bin/supercronic && \
    apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Copy application code
COPY src/ src/
COPY docker/crontab /app/crontab

# Default: run supercronic with the crontab
CMD ["supercronic", "/app/crontab"]
