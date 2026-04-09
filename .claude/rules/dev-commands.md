# Development Commands

```bash
uv sync --all-extras --dev          # install dependencies
uv run pytest                       # run all tests
uv run pytest tests/test_indicators.py                        # single file
uv run pytest tests/test_indicators.py::test_calculate_obv   # single test
uv run pytest --cov=src/volume_price_analysis --cov-report=term-missing  # coverage
uv run ruff format src/ tests/      # format
uv run ruff check --fix src/ tests/ # lint + auto-fix
uv run mypy src/                    # type check
uv run python -m volume_price_analysis.server  # run MCP server directly
```

## Docker

```bash
docker build -t volume-price-analysis .
docker run --env-file .env volume-price-analysis
```

## CLI Entry Points (pyproject.toml)

- `volume-price-analysis` → `server:main` — MCP server
- `morning-briefing` → `morning_agent:main` — single briefing run
- `morning-scheduler` → `scheduler:main` — asyncio scheduler (used in Docker)
