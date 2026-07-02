# Development Commands

```bash
uv sync --all-extras --dev          # install dependencies
uv run pytest                       # run all tests
uv run pytest tests/test_indicators.py                        # single file
uv run pytest tests/test_indicators.py::TestOBV::test_obv_basic_calculation  # single test
uv run pytest -n 0                  # serial run (disable pytest-xdist parallelism, e.g. for pdb)
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

- `volume-price-analysis` → `volume_price_analysis.server:main` — MCP server
- `morning-briefing` → `volume_price_analysis.agent.morning_agent:main` — single briefing run
- `morning-scheduler` → `volume_price_analysis.agent.scheduler:main` — asyncio scheduler (used in Docker)
