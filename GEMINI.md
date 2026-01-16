# Volume-Price Analysis MCP Server

## Project Overview
This project is a Model Context Protocol (MCP) server that provides comprehensive volume-price analysis tools for stock market data. It allows users to analyze stocks using various technical indicators like On-Balance Volume (OBV), Volume Weighted Average Price (VWAP), Volume Profile, Money Flow Index (MFI), and more. It is designed to be used with AI assistants like Claude Code or Gemini.

## Key Features & Tools
The server exposes several tools for financial analysis:

*   **`get_stock_data`**: Fetch historical stock data for any symbol.
*   **`calculate_obv`**: Calculate On-Balance Volume (OBV) to track money flow.
*   **`calculate_vwap`**: Calculate Volume Weighted Average Price (VWAP) as a trading benchmark.
*   **`calculate_volume_profile`**: Analyze volume distribution across price levels (includes POC, VAH, VAL).
*   **`calculate_mfi`**: Calculate Money Flow Index (volume-weighted RSI).
*   **`analyze_volume_trends`**: Analyze volume trends and detect price-volume divergences.
*   **`comprehensive_analysis`**: Perform a full analysis including all major indicators and a summary.
*   **`options_analysis`**: Specialized analysis optimized for short-term options trading (2-week holding period).

## Tech Stack
*   **Language**: Python 3.11+
*   **Core Libraries**:
    *   `mcp`: Model Context Protocol SDK
    *   `yfinance`: Stock data fetching
    *   `pandas` & `numpy`: Data manipulation and numerical analysis
*   **Package Management**: `uv` (recommended) or `pip`

## Development

### Setup
Recommended setup using `uv`:
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Running the Server
To run the server locally (using stdio):
```bash
python -m volume_price_analysis.server
```

### Testing
The project uses `pytest` for testing.
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/volume_price_analysis --cov-report=term-missing
```

### Code Quality
The project enforces code quality using `ruff` and `mypy`.
```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## Directory Structure
*   `src/volume_price_analysis/`: Source code.
    *   `server.py`: Main MCP server entry point and tool definitions.
    *   `indicators.py`: Implementation of financial indicators.
    *   `data_fetcher.py`: Logic for fetching data from Yahoo Finance.
*   `tests/`: Test suite (pytest).
*   `pyproject.toml`: Project configuration and dependencies.
*   `QUICKSTART.md` & `README.md`: Documentation.
*   `UV_SETUP.md`: Instructions for using `uv`.
