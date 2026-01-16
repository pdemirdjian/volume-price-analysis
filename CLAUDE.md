# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server that provides volume-price technical analysis tools for stock market data. It fetches data via Yahoo Finance (yfinance) and exposes analysis capabilities to AI assistants like Claude Code.

## Commands

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

## Architecture

```
src/volume_price_analysis/
├── server.py        # MCP server - tool definitions & handlers
├── indicators.py    # Pure calculation functions (24+ indicators)
└── data_fetcher.py  # Yahoo Finance data retrieval
```

### Data Flow

1. **MCP Client** calls a tool (e.g., `comprehensive_analysis`)
2. **server.py** `handle_call_tool()` receives the request
3. **data_fetcher.py** `fetch_stock_data()` retrieves OHLCV data from Yahoo Finance
4. **indicators.py** functions compute the requested indicators
5. **server.py** formats results as JSON and returns via MCP protocol

### Key Design Patterns

- **indicators.py**: Stateless pure functions that take a pandas DataFrame and return calculated values. All functions expect columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

- **server.py**: Defines 9 MCP tools via `@server.list_tools()` decorator. Tool execution is handled by `handle_call_tool()` which dispatches based on tool name.

- **Tool categories**:
  - Basic: `get_stock_data`, `calculate_obv`, `calculate_vwap`, `calculate_volume_profile`, `calculate_mfi`, `analyze_volume_trends`
  - Comprehensive: `comprehensive_analysis` (all indicators + summary)
  - Options-focused: `options_analysis` (ADX, RSI divergence, IV percentile, expected move)
  - Scanning: `scan_candidates` (market-wide screening with pre-built universes)

### Adding New Indicators

1. Add calculation function to `indicators.py` (takes DataFrame, returns Series/dict)
2. Add corresponding test to `tests/test_indicators.py`
3. If exposing as MCP tool, add `Tool()` definition in `server.py` `handle_list_tools()`
4. Add handler case in `handle_call_tool()`
