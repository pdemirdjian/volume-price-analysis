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
  - `mcp`: Model Context Protocol SDK (>=1.25.0)
  - `yfinance`: Stock data fetching (>=1.0)
  - `pandas` (>=2.3.3) & `numpy` (>=2.4.1)
- **Package Management**: `uv` (recommended) or `pip`

## Architecture

```text
src/volume_price_analysis/
├── server.py        # MCP server - tool definitions & handlers
├── indicators.py    # Pure calculation functions (23 indicators)
└── data_fetcher.py  # Yahoo Finance data retrieval
```

### Data Flow

1. **MCP Client** calls a tool (e.g., `comprehensive_analysis`)
2. **server.py** `handle_call_tool()` receives the request
3. **data_fetcher.py** `fetch_stock_data()` retrieves OHLCV data from Yahoo
   Finance
4. **indicators.py** functions compute the requested indicators
5. **server.py** formats results as JSON and returns via MCP protocol

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

## Development Commands

```bash
# Install dependencies (use uv, it's configured in pyproject.toml)
uv sync --all-extras --dev

# Run tests
pytest

# Run single test file
pytest tests/test_indicators.py

# Run single test
pytest tests/test_indicators.py::test_calculate_obv

# Run with coverage
pytest --cov=src/volume_price_analysis --cov-report=term-missing

# Format code
ruff format src/ tests/

# Lint (with auto-fix)
ruff check --fix src/ tests/

# Type check
mypy src/

# Run the MCP server directly
python -m volume_price_analysis.server
```

### Adding New Indicators

1. Add calculation function to `indicators.py` (takes DataFrame, returns
   Series/dict)
2. Add corresponding test to `tests/test_indicators.py`
3. If exposing as MCP tool, add `Tool()` definition in `server.py`
   `handle_list_tools()`
4. Add handler case in `handle_call_tool()`
