FROM python:3.14-slim

# Install supercronic (lightweight cron for containers)
# - Logs to stdout/stderr (works with docker logs)
# - No syslog dependency
# - Graceful signal handling for container stops
# - Supports env vars from Docker
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    SUPERCRONIC_URL="https://github.com/aptible/supercronic/releases/download/v0.2.33/supercronic-linux-amd64" && \
    curl -fsSL "$SUPERCRONIC_URL" -o /usr/local/bin/supercronic && \
    chmod +x /usr/local/bin/supercronic && \
    apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Copy application code
COPY src/ src/
COPY docker/crontab /app/crontab

# Default: run supercronic with the crontab
CMD ["supercronic", "/app/crontab"]
