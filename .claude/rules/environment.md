# Environment Variables

AI and email variables are required for the morning briefing agent only.
`DATA_CACHE_TTL_SECONDS` applies to both the MCP server and briefing agent. See `src/volume_price_analysis/agent/config.py` `AgentConfig.from_env()` for full list.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `AI_PROVIDER` | No | `gemini` | `gemini` or `anthropic` |
| `AI_PROVIDER_API_KEY` | Yes | — | API key for chosen provider |
| `AI_MODEL` | No | provider default | Override model |
| `EMAIL_FROM` | Yes | — | Gmail address for sending |
| `EMAIL_PASSWORD` | Yes | — | Gmail app password |
| `EMAIL_TO` | Yes | — | Comma-separated recipients |
| `EMAIL_SMTP_HOST` | No | `smtp.gmail.com` | SMTP server |
| `EMAIL_SMTP_PORT` | No | `587` | SMTP port |
| `DATA_CACHE_TTL_SECONDS` | No | `900` | History cache TTL in seconds; `0` disables caching; invalid values fall back to `900` |
| `SCAN_UNIVERSE` | No | `full_market` | Symbol universe |
| `MAX_DEEP_ANALYSIS` | No | `5` | Max candidates for deep analysis (1–20; out-of-range values fall back/clamp) |
| `SCHEDULER_HEARTBEAT_FILE` | No | `<tempdir>/morning-scheduler.heartbeat` | Liveness file the scheduler writes (monotonic stamp); read by `morning-healthcheck` |

Briefing scan variables are documented with defaults and bounds in the
[README environment table](../../README.md#configuration). They are `SCAN_PERIOD`,
`SCAN_HOLDING_PERIOD`, `SCAN_MIN_SCORE`, `SCAN_MIN_ADX`, `SCAN_MAX_IV_PERCENTILE`,
`SCAN_MIN_AVG_DAILY_VOLUME`, and `SCAN_MAX_RESULTS`. Invalid numeric scan values
raise an error; validation uses the MCP scan bounds and registered universes.
`LOG_LEVEL` defaults to `INFO` and controls MCP server logging to stderr.

Store in `.env` (gitignored) or pass via Docker `--env-file`.
