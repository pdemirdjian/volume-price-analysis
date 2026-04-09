# Environment Variables

Required for the morning briefing agent only. Not needed for the MCP server. See `agent/config.py` `AgentConfig.from_env()` for full list.

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
| `SCAN_UNIVERSE` | No | `full_market` | Symbol universe |
| `MAX_DEEP_ANALYSIS` | No | `5` | Max candidates for deep analysis |

Store in `.env` (gitignored) or pass via Docker `--env-file`.
