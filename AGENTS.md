# AI Context & Developer Guide

This file provides context and guidance for AI assistants (Claude Code, Gemini,
etc.) and developers working on this repository.

## Project Overview

This is an MCP (Model Context Protocol) server that provides volume-price
technical analysis tools for stock market data. It fetches data via Yahoo
Finance (yfinance) and exposes analysis capabilities to AI assistants.

## Tech Stack

- **Language**: Python 3.14+
- **Core Libraries**:
  - `mcp`: Model Context Protocol SDK (1.26.0)
  - `yfinance`: Stock data fetching (1.2.0)
  - `pandas` (3.0.1) & `numpy` (2.4.2)
  - `anthropic` & `google-genai`: AI providers for morning briefing
  - `markdown`: Email HTML rendering
  - `holidays`: NYSE market holiday detection
  - `pytickersymbols`: Stock symbol universe (~200 symbols)
- **Package Management**: `uv` (recommended) or `pip`
- **Deployment**: Docker container with Python asyncio scheduler, CI/CD via
  GitHub Actions to ghcr.io

## Architecture

```text
src/volume_price_analysis/
├── server.py        # MCP server - tool definitions & handlers
├── indicators.py    # Pure calculation functions (23 indicators)
├── data_fetcher.py  # Yahoo Finance data retrieval
├── analysis.py      # Reusable scan/analysis logic (used by server + agent)
└── agent/           # Morning briefing agent
    ├── ai_client.py       # AI provider abstraction (Gemini/Anthropic)
    ├── config.py          # Environment-based configuration
    ├── email_sender.py    # SMTP email delivery
    ├── morning_agent.py   # Main orchestrator (entry point)
    └── scheduler.py       # Asyncio scheduler (replaces supercronic)
```

### MCP Data Flow

1. **MCP Client** calls a tool (e.g., `comprehensive_analysis`)
2. **server.py** `handle_call_tool()` receives the request
3. **data_fetcher.py** `fetch_stock_data()` retrieves OHLCV data from Yahoo
   Finance
4. **indicators.py** functions compute the requested indicators
5. **server.py** formats results as JSON and returns via MCP protocol

### Morning Briefing Flow

1. **scheduler.py** triggers `morning_agent.py` at 8:30 AM ET by default (weekdays)
   - Use `--time HH:MM` to change the trigger time
   - Use `--skip-holidays` to skip NYSE market holidays
2. **analysis.py** `run_scan()` scans ~200 symbols for top candidates
3. **analysis.py** `run_options_analysis()` deep-analyzes top N candidates
4. **ai_client.py** sends data to Gemini or Claude API for natural-language
   briefing
5. **email_sender.py** delivers the briefing via Gmail SMTP

## Key Features & Tools

The server exposes the following MCP tools:

- **`get_stock_data`**: Fetch historical stock data for any symbol.
- **`calculate_obv`**: Calculate On-Balance Volume (OBV).
- **`calculate_vwap`**: Calculate Volume Weighted Average Price (VWAP).
- **`calculate_volume_profile`**: Analyze volume distribution across price
  levels.
- **`calculate_mfi`**: Calculate Money Flow Index (MFI).
- **`analyze_volume_trends`**: Analyze volume trends and detect price-volume
  divergences.
- **`comprehensive_analysis`**: Perform a full analysis including all major
  indicators and a summary.
- **`options_analysis`**: Specialized analysis optimized for short-term options
  trading (14-30 day holding period).
- **`scan_candidates`**: Scan the market to find the best options trading
  candidates based on composite scores.

## CLI Entry Points

Defined in `pyproject.toml [project.scripts]`:

- **`volume-price-analysis`** → `server:main` — Run the MCP server
- **`morning-briefing`** → `agent.morning_agent:main` — Run a single morning
  briefing
- **`morning-scheduler`** → `agent.scheduler:main` — Run the asyncio scheduler
  (used in Docker)

## Development Commands

```bash
# Install dependencies (use uv, it's configured in pyproject.toml)
uv sync --all-extras --dev

# Run tests
uv run pytest

# Run single test file
uv run pytest tests/test_indicators.py

# Run single test
uv run pytest tests/test_indicators.py::test_calculate_obv

# Run with coverage (also the default via pyproject.toml addopts)
uv run pytest --cov=src/volume_price_analysis --cov-report=term-missing

# Format code
uv run ruff format src/ tests/

# Lint (with auto-fix)
uv run ruff check --fix src/ tests/

# Type check
uv run mypy src/

# Run the MCP server directly
uv run python -m volume_price_analysis.server
```

### Docker

```bash
# Build
docker build -t volume-price-analysis .

# Run (requires env vars — see Environment Variables below)
docker run --env-file .env volume-price-analysis
```

## Environment Variables

Required for the morning briefing agent (not needed for MCP server). See
`agent/config.py` `AgentConfig.from_env()` for all options.

| Variable | Required | Purpose |
|----------|----------|---------|
| `AI_PROVIDER` | No | `"gemini"` (default) or `"anthropic"` |
| `AI_PROVIDER_API_KEY` | Yes | API key for the chosen AI provider |
| `AI_MODEL` | No | Override default model (empty = provider default) |
| `EMAIL_FROM` | Yes | Gmail address for sending briefings |
| `EMAIL_PASSWORD` | Yes | Gmail app password |
| `EMAIL_TO` | Yes | Comma-separated recipient addresses |
| `EMAIL_SMTP_HOST` | No | SMTP server (default: `smtp.gmail.com`) |
| `EMAIL_SMTP_PORT` | No | SMTP port (default: `587`) |
| `SCAN_UNIVERSE` | No | Symbol universe (default: `full_market`) |
| `MAX_DEEP_ANALYSIS` | No | Max candidates for deep analysis (default: `5`) |

Store these in `.env` (gitignored) or pass via Docker `--env-file`.

## Code Style

Configured in `pyproject.toml [tool.ruff]`:

- **Line length**: 100
- **Quote style**: double quotes
- **Indent style**: spaces
- **Lint rules**: pycodestyle, pyflakes, isort, bugbear, comprehensions,
  pyupgrade, pep8-naming, flake8-async

### Adding New Indicators

1. Add calculation function to `indicators.py` (takes DataFrame, returns
   Series/dict)
2. Add corresponding test to `tests/test_indicators.py`
3. If exposing as MCP tool, add `Tool()` definition in `server.py`
   `handle_list_tools()`
4. Add handler case in `handle_call_tool()`
