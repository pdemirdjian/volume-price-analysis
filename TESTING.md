# Testing Guide

Comprehensive testing documentation for the Volume-Price Analysis MCP Server.

## Test Suite Overview

Tests live in `tests/`, one `test_*.py` module per area (indicators, MCP tool
registry and server, scan/analysis, data fetching and caching, backtest, morning
briefing agent, scheduler, healthcheck). Shared fixtures and the integration-test
opt-in live in `tests/conftest.py`. List what exists rather than trusting a
hand-maintained count:

```bash
uv run pytest --collect-only -q
```

Default `pytest` options come from `[tool.pytest.ini_options]` in `pyproject.toml`:
tests run in parallel (`-n=auto`, pytest-xdist), with coverage on and the **80%**
gate (`--cov-fail-under=80`) enforced, and non-loopback network blocked
(pytest-socket).

## Quick Start

```bash
# Install with test dependencies
uv sync --all-extras --dev

# Run all tests (coverage report and 80% gate included)
uv run pytest
```

## Running Tests

### All Tests

```bash
uv run pytest
```

### Specific Test File

```bash
uv run pytest tests/test_indicators.py
uv run pytest tests/test_data_fetcher.py
uv run pytest tests/test_server.py
```

### Specific Test Class

```bash
uv run pytest tests/test_indicators.py::TestOBV
uv run pytest tests/test_server.py::TestCallToolVWAP
```

### Specific Test Function

```bash
uv run pytest tests/test_indicators.py::TestOBV::test_obv_uptrend
uv run pytest tests/test_server.py::TestListTools::test_list_tools_returns_all_tools
```

### With Verbose Output

```bash
uv run pytest -v
```

### With Print Statements

```bash
uv run pytest -s
```

### Serial Run (debugging, pdb)

```bash
uv run pytest -n 0
```

### Stop on First Failure

```bash
uv run pytest -x
```

### Run Last Failed Tests

```bash
uv run pytest --lf
```

## Coverage Reports

### Terminal Coverage

```bash
uv run pytest --cov=src/volume_price_analysis --cov-report=term-missing
```

This shows:

- Overall statement-and-branch coverage percentage
- Line-by-line coverage
- Missing lines and branch destinations highlighted

### HTML Coverage Report

```bash
uv run pytest --cov=src/volume_price_analysis --cov-report=html
```

Open `htmlcov/index.html` in your browser for interactive coverage exploration.

### XML Coverage Report

```bash
uv run pytest --cov-report=xml
```

This creates `coverage.xml` with line and branch data.

### Coverage by Module

```bash
# Only indicators module
uv run pytest tests/test_indicators.py --cov-reset --cov=volume_price_analysis.indicators

# Only data fetcher
uv run pytest tests/test_data_fetcher.py --cov-reset --cov=volume_price_analysis.data_fetcher

# Only server
uv run pytest tests/test_server.py --cov-reset --cov=volume_price_analysis.server
```

## Test Fixtures

The test suite uses several fixtures defined in `conftest.py`:

### `sample_stock_data`

General-purpose stock data with realistic patterns

### `uptrend_data`

Clear uptrend with increasing volume

### `downtrend_data`

Clear downtrend with increasing volume

### `flat_price_data`

Flat price with varying volume

## Mocking

Tests avoid network calls by injecting an `InMemoryDataSource` through the
`DataSource` seam instead of patching module-level fetch functions:

```python
from volume_price_analysis.data_fetcher import InMemoryDataSource
from volume_price_analysis.server import dispatch


async def test_get_stock_data_basic():
    source = InMemoryDataSource(frames={"AAPL": pd.DataFrame(...)})

    result = await dispatch("get_stock_data", {"symbol": "AAPL"}, data_source=source)
```

Benefits:

- ⚡ Fast (no network delays)
- 🔒 Reliable (no external dependencies)
- 🎯 Focused (tests our code, not yfinance)

## Integration Tests

Tests that call real upstream services (Yahoo Finance) are marked `integration`
and skipped by default with the reason `integration test: needs --run-integration`.
The default run keeps blocking non-loopback hosts; `--run-integration` lifts the
pytest-socket guard for `integration` tests only.

```python
@pytest.mark.integration
class TestIntegration:
    def test_fetch_real_stock_data(self):
        result = YFinanceDataSource(cache_ttl=0).fetch("AAPL", period="5d")
        assert len(result) > 0
```

Run them locally (needs network; `--no-cov` because the coverage floor only
makes sense for the full suite):

```bash
uv run pytest -m integration --run-integration --no-cov -n 0
```

The [integration workflow](.github/workflows/integration.yml) runs the same
command nightly and on manual dispatch, so a yfinance API break shows up as a
failed scheduled run.

## Continuous Testing

### Watch Mode (requires pytest-watch)

```bash
pip install pytest-watch
ptw
```

Tests automatically re-run when files change.

### Pre-commit Hooks

Hooks are managed by the [pre-commit](https://pre-commit.com) framework and configured in
`.pre-commit-config.yaml` (ruff, ruff-format, mypy, and baseline file checks). Install them
once per clone:

```bash
uv run --with pre-commit pre-commit install
```

Run every hook against the whole tree (what CI's `pre-commit` job does):

```bash
uv run --with pre-commit pre-commit run --all-files
```

The hooks don't run the test suite; run `uv run pytest` before pushing.

## Writing New Tests

### Test Structure

```python
import pytest
from volume_price_analysis.indicators import calculate_obv


class TestNewIndicator:
    """Tests for new indicator."""

    def test_basic_calculation(self, sample_stock_data):
        """Test basic calculation."""
        result = calculate_obv(sample_stock_data)

        assert len(result) == len(sample_stock_data)
        assert result.iloc[0] == 0

    def test_edge_case(self):
        """Test edge case handling."""
        # Create specific test data
        data = pd.DataFrame(...)

        # Test behavior
        result = calculate_obv(data)
        assert ...
```

### Async Tests

```python
import pytest


class TestAsyncFunction:
    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test async function."""
        result = await some_async_function()
        assert result is not None
```

### Parametrized Tests

```python
@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("AAPL", True),
        ("BRK-B", True),
        ("AAPL!@#", False),
    ],
)
def test_symbol_formats(symbol, expected):
    assert validate_symbol_format(symbol) is expected
```

## Troubleshooting

### ImportError

```bash
# Ensure package is installed
uv sync --all-extras --dev
```

### ModuleNotFoundError

```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Async Warnings

Configure pytest in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### Fixture Not Found

Check that `conftest.py` exists in the `tests/` directory.

## Best Practices

1. **Test Names** - Use descriptive names: `test_obv_increases_with_uptrend`
2. **One Assertion Per Test** - Or closely related assertions
3. **Arrange-Act-Assert** - Clear test structure
4. **Mock External Calls** - Don't rely on network/external APIs
5. **Test Edge Cases** - Empty data, None values, zeros
6. **Use Fixtures** - Reuse common test data
7. **Keep Tests Fast** - Mock slow operations
8. **Test Behavior** - Not implementation details

## Coverage Goals

The enforced minimum is **80% statement-and-branch coverage**
(`--cov-fail-under=80` in `pyproject.toml`). See the latest CI summary or PR
coverage comment for the current total.

Excluded from coverage (see `[tool.coverage.report]` in `pyproject.toml`):

- Main entry points (`if __name__ == "__main__":`)
- Protocol stub bodies (`...`)

## CI/CD Integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) is the source of truth.
In short: it runs ruff (format and lint), mypy, and the full test suite with the
same 80% coverage gate on Ubuntu, macOS, and Windows, and posts a coverage
summary on pull requests. Real-network `integration` tests run separately in
[`.github/workflows/integration.yml`](.github/workflows/integration.yml)
(nightly and on manual dispatch).

To reproduce the CI checks locally:

```bash
uv run ruff format --check src/ tests/ example_usage.py run_analysis_local.py
uv run ruff check src/ tests/ example_usage.py run_analysis_local.py
uv run mypy src/
uv run pytest
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

Happy testing! 🧪
