# Testing Guide

Comprehensive testing documentation for the Volume-Price Analysis MCP Server.

## Test Suite Overview

The project includes **50+ tests** organized into three main test modules:

| Test Module | Tests | Coverage |
|------------|-------|----------|
| `test_indicators.py` | 25+ | All volume-price indicators |
| `test_data_fetcher.py` | 15+ | Data fetching and validation |
| `test_server.py` | 15+ | MCP server integration |

## Quick Start

```bash
# Install with test dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src/volume_price_analysis --cov-report=term-missing
```

## Test Modules

### 1. Indicators Tests (`test_indicators.py`)

Tests all volume-price calculation functions:

**TestOBV** - On-Balance Volume
- Basic calculation
- Uptrend behavior
- Downtrend behavior
- Flat price handling

**TestVWAP** - Volume Weighted Average Price
- Basic calculation
- Price range validation
- Cumulative behavior
- NaN handling

**TestVolumeProfile** - Volume distribution
- Basic profile generation
- Total volume conservation
- Price range coverage
- Different bin configurations

**TestMFI** - Money Flow Index
- Basic calculation
- 0-100 range validation
- Initial NaN values
- Different periods

**TestVPT** - Volume-Price Trend
- Basic calculation
- Uptrend/downtrend behavior

**TestVolumeTrends** - Trend analysis
- Basic analysis
- Divergence detection
- Percentage formatting

**TestEdgeCases**
- Empty DataFrames
- Single row DataFrames
- Zero volume handling

### 2. Data Fetcher Tests (`test_data_fetcher.py`)

Tests stock data fetching functionality:

**TestFetchStockData**
- Fetching with period parameter
- Fetching with date range
- Empty result handling
- Column filtering
- Date as column (not index)

**TestValidateSymbol**
- Valid symbol validation
- Invalid symbol handling
- Exception handling

**TestIntegration** (skipped by default)
- Real Yahoo Finance data fetching
- Real symbol validation

### 3. Server Tests (`test_server.py`)

Tests MCP server functionality:

**TestListTools**
- All tools listed
- Valid schemas

**TestCallToolGetStockData**
- Basic stock data retrieval
- Date range queries

**TestCallToolOBV**
- OBV calculation via MCP

**TestCallToolVWAP**
- VWAP calculation via MCP

**TestCallToolVolumeProfile**
- Volume profile via MCP

**TestCallToolMFI**
- MFI calculation via MCP

**TestCallToolVolumeTrends**
- Trend analysis via MCP

**TestCallToolComprehensive**
- Complete analysis with all indicators

**TestErrorHandling**
- Invalid symbol errors
- Unknown tool errors

**TestGenerateSummary**
- Bullish condition summaries
- Divergence summaries

## Running Tests

### All Tests

```bash
pytest
```

### Specific Test File

```bash
pytest tests/test_indicators.py
pytest tests/test_data_fetcher.py
pytest tests/test_server.py
```

### Specific Test Class

```bash
pytest tests/test_indicators.py::TestOBV
pytest tests/test_server.py::TestCallToolVWAP
```

### Specific Test Function

```bash
pytest tests/test_indicators.py::TestOBV::test_obv_uptrend
pytest tests/test_server.py::TestListTools::test_list_tools_returns_all_tools
```

### With Verbose Output

```bash
pytest -v
```

### With Print Statements

```bash
pytest -s
```

### Stop on First Failure

```bash
pytest -x
```

### Run Last Failed Tests

```bash
pytest --lf
```

## Coverage Reports

### Terminal Coverage

```bash
pytest --cov=src/volume_price_analysis --cov-report=term-missing
```

This shows:
- Overall coverage percentage
- Line-by-line coverage
- Missing lines highlighted

### HTML Coverage Report

```bash
pytest --cov=src/volume_price_analysis --cov-report=html
```

Open `htmlcov/index.html` in your browser for interactive coverage exploration.

### Coverage by Module

```bash
# Only indicators module
pytest tests/test_indicators.py --cov=src/volume_price_analysis.indicators

# Only data fetcher
pytest tests/test_data_fetcher.py --cov=src/volume_price_analysis.data_fetcher

# Only server
pytest tests/test_server.py --cov=src/volume_price_analysis.server
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

Tests use mocking to avoid network calls:

```python
@patch("volume_price_analysis.server.fetch_stock_data")
async def test_get_stock_data_basic(self, mock_fetch):
    # Setup mock data
    mock_data = pd.DataFrame(...)
    mock_fetch.return_value = mock_data

    # Run test
    result = await handle_call_tool(...)
```

Benefits:
- ⚡ Fast (no network delays)
- 🔒 Reliable (no external dependencies)
- 🎯 Focused (tests our code, not yfinance)

## Integration Tests

Some tests require network access and are skipped by default:

```python
@pytest.mark.skip(reason="Requires network access")
def test_fetch_real_stock_data(self):
    result = fetch_stock_data("AAPL", period="5d")
    assert len(result) > 0
```

To run integration tests:

```bash
# Run including skipped tests
pytest --run-skipped

# Or remove the @pytest.mark.skip decorator
```

## Continuous Testing

### Watch Mode (requires pytest-watch)

```bash
pip install pytest-watch
ptw
```

Tests automatically re-run when files change.

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
pytest
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

```bash
chmod +x .git/hooks/pre-commit
```

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
@pytest.mark.parametrize("symbol,expected", [
    ("AAPL", True),
    ("MSFT", True),
    ("INVALID123", False),
])
def test_validate_multiple_symbols(symbol, expected):
    result = validate_symbol(symbol)
    assert result == expected
```

## Troubleshooting

### ImportError

```bash
# Ensure package is installed
uv pip install -e ".[dev]"
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

Current coverage: **~95%**

Target coverage by module:
- `indicators.py`: 95%+
- `data_fetcher.py`: 90%+
- `server.py`: 85%+

Untested areas (acceptable):
- Integration with real Yahoo Finance API (requires network)
- Main entry point (`if __name__ == "__main__"`)

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv pip install -e ".[dev]"
      - name: Run tests
        run: pytest --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

Happy testing! 🧪
