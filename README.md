# Volume-Price Analysis MCP Server

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

A Model Context Protocol (MCP) server that provides comprehensive volume-price
analysis tools for stock market data. Analyze stocks using indicators like OBV,
VWAP, Volume Profile, Money Flow Index, and more.

## Features

### Available Analysis Tools

1. **get_stock_data** - Fetch historical stock data for any symbol
2. **calculate_obv** - On-Balance Volume indicator
3. **calculate_vwap** - Volume Weighted Average Price
4. **calculate_volume_profile** - Distribution of volume at price levels
5. **calculate_mfi** - Money Flow Index (volume-weighted RSI)
6. **calculate_ad_line** - Accumulation/Distribution Line
7. **calculate_cmf** - Chaikin Money Flow
8. **analyze_volume_trends** - Volume trend analysis with divergence detection
9. **comprehensive_analysis** - Complete analysis with all indicators
10. **options_analysis** - Specialized analysis for short-term options trading
11. **scan_candidates** - Scan ~200 symbols for top options trading candidates

### Key Indicators Explained

#### Volume Indicators

- **OBV (On-Balance Volume)**: Cumulative volume indicator that tracks money
  flow. Rising OBV suggests accumulation, falling OBV suggests distribution.

- **A/D Line (Accumulation/Distribution)**: More sophisticated than OBV -
  considers where price closes within the high-low range. Better detects
  institutional buying/selling pressure.

- **VPT (Volume-Price Trend)**: Similar to OBV but uses percentage price
  changes, more sensitive to magnitude of moves.

- **MFI (Money Flow Index)**: Volume-weighted version of RSI. Values >80
  indicate overbought conditions, <20 indicate oversold.

- **CMF (Chaikin Money Flow)**: Measures buying/selling pressure over a period.
  Ranges from -1 to +1. Positive = buying pressure, negative = selling pressure.

- **RVOL (Relative Volume)**: Compares current volume to average. RVOL > 1.5
  indicates significantly higher activity (potential catalyst).

- **Volume Breakout Detection**: Identifies when volume exceeds historical
  thresholds, signaling potential trend changes.

- **Volume Trends**: Analyzes current volume vs. historical average and detects
  price-volume divergences (potential reversal signals).

#### Price Indicators

- **VWAP (Volume Weighted Average Price)**: The average price weighted by
  volume. Institutional traders use this as a benchmark. Price above VWAP =
  bullish, below = bearish.

- **VWMA (Volume-Weighted Moving Average)**: Moving average weighted by volume,
  more responsive to institutional activity than simple MA.

- **Price ROC (Rate of Change)**: Momentum indicator with volume confirmation to
  validate trend strength.

#### Volatility Indicators (Critical for Options)

- **Historical Volatility (HV)**: Measures actual price volatility over time.
  Essential for comparing against Implied Volatility to find mispriced options.

- **ATR (Average True Range)**: Measures market volatility across entire price
  range. Critical for position sizing and stop-loss placement.

- **Bollinger Bands**: Identifies overbought/oversold conditions and volatility
  squeezes. %B indicator and bandwidth help time entries.

#### Volume Profile

- **POC (Point of Control)**: Price level with highest volume - strongest
  support/resistance.

- **VAH/VAL (Value Area High/Low)**: Upper and lower bounds of 70% of volume.
  Critical for options strike selection.

- **Enhanced Volume Profile**: Shows volume distribution across price levels for
  identifying key support/resistance zones.

## Installation

### Prerequisites

- Python 3.14 or higher
- [UV](https://github.com/astral-sh/uv) (recommended) or pip

### Recommended: Install with UV (Fast!)

UV is 10-100x faster than pip. [See UV_SETUP.md](UV_SETUP.md) for detailed
instructions.

```bash
# Install UV (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Install UV (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Navigate to project
cd volume-price-analysis

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install the package with dev tools
uv sync --all-extras --dev
```

### Alternative: Traditional pip Installation

```bash
cd volume-price-analysis

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install the package
pip install -e .
```

This installs:

- mcp (Model Context Protocol SDK)
- yfinance (Yahoo Finance data fetcher)
- pandas (Data manipulation)
- numpy (Numerical computations)

## Configuration

### For Claude Code

Add this server to your Claude Code MCP configuration:

**macOS/Linux**: `~/.config/claude-code/mcp_settings.json`

**Windows**: `%APPDATA%\claude-code\mcp_settings.json`

```json
{
  "mcpServers": {
    "volume-price-analysis": {
      "command": "python",
      "args": ["-m", "volume_price_analysis.server"],
      "env": {}
    }
  }
}
```

If you installed in a virtual environment, use the full path to Python:

```json
{
  "mcpServers": {
    "volume-price-analysis": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "volume_price_analysis.server"],
      "env": {}
    }
  }
}
```

### For Other MCP Clients

The server uses stdio for communication. Run it with:

```bash
python -m volume_price_analysis.server
```

## Usage Examples

Once configured, you can use the tools through Claude Code or any MCP client:

### Example 1: Get Stock Data

```text
Fetch the last 3 months of data for Apple stock
```

This will use the `get_stock_data` tool with parameters:

- symbol: "AAPL"
- period: "3mo"

### Example 2: Calculate VWAP

```text
Calculate VWAP for Tesla over the last 6 months
```

Uses `calculate_vwap` with:

- symbol: "TSLA"
- period: "6mo"

### Example 3: Volume Profile

```text
Show me the volume profile for MSFT from 2024-01-01 to 2024-12-31
```

Uses `calculate_volume_profile` with:

- symbol: "MSFT"
- start_date: "2024-01-01"
- end_date: "2024-12-31"

### Example 4: Comprehensive Analysis

```text
Perform a comprehensive volume-price analysis on NVDA for the last year
```

Uses `comprehensive_analysis` which returns:

- All major indicators (OBV, VWAP, MFI, VPT)
- Volume profile with Point of Control
- Volume trend analysis
- Price-volume divergence detection
- Human-readable summary

### Example 5: Money Flow Index

```text
Calculate the Money Flow Index for SPY over the last month
```

Uses `calculate_mfi` with:

- symbol: "SPY"
- period: "1mo"

## Tool Parameters

### Common Parameters

All tools accept these parameters:

- **symbol** (required): Stock ticker symbol (e.g., "AAPL", "MSFT", "TSLA")
- **period** (optional): Time period - "1d", "5d", "1mo", "3mo", "6mo", "1y",
  "2y", "5y", "10y", "ytd", "max"
- **start_date** (optional): Start date in YYYY-MM-DD format
- **end_date** (optional): End date in YYYY-MM-DD format

**Note**: Use either `period` OR (`start_date` AND `end_date`), not both.

### Tool-Specific Parameters

- **calculate_volume_profile**:
  - `num_bins` (default: 20): Number of price levels to analyze

- **calculate_mfi**:
  - `mfi_period` (default: 14): Lookback period for MFI calculation

- **analyze_volume_trends**:
  - `window` (default: 20): Rolling window for trend analysis

## Output Format

All tools return structured JSON data. Example output from
`comprehensive_analysis`:

```json
{
  "symbol": "AAPL",
  "period": "2024-01-01 to 2024-12-31",
  "latest_price": 185.23,
  "indicators": {
    "obv": {
      "value": 123456789,
      "trend": "increasing"
    },
    "vwap": {
      "value": 182.45,
      "price_vs_vwap": "+1.52%",
      "position": "above"
    },
    "mfi": {
      "value": 65.3,
      "condition": "Neutral"
    }
  },
  "volume_profile": {
    "point_of_control": 180.50,
    "poc_significance": "High volume node - potential support/resistance"
  },
  "volume_trends": {
    "current_volume": 50234000,
    "average_volume": 48123000,
    "volume_vs_average": "+4.39%",
    "divergence_detected": false
  },
  "summary": [
    "Price is trading above VWAP, indicating bullish sentiment",
    "OBV is increasing, suggesting accumulation"
  ]
}
```

## Development

### Running Tests

Comprehensive test suite with 600+ tests covering all functionality:

```bash
# Install with dev dependencies
uv sync --all-extras --dev

# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src/volume_price_analysis --cov-report=term-missing --cov-report=html

# Run specific test file
uv run pytest tests/test_indicators.py

# Run with verbose output
uv run pytest -v
```

Test coverage includes:

- ✅ All volume-price indicators (OBV, VWAP, MFI, Volume Profile, VPT)
- ✅ Data fetching and validation
- ✅ MCP server tool registration and execution
- ✅ Morning briefing agent (config, AI providers, email, orchestrator)
- ✅ Analysis module (scan, options analysis, universes)
- ✅ Edge cases and error handling

### Code Quality

```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/

# Type checking
mypy src/
```

### Evidence Harness (strictly-causal backtest)

`volume_price_analysis.backtest` is an **internal evidence tool** (no MCP
surface) for measuring the *forward predictive power* of the composite score, so
scoring changes can be evaluated with before/after numbers instead of intuition.

It is **strictly causal**: the signal at bar `t` is computed by slicing history
to `data[:t+1]` and calling the existing scorer, so future bars are physically
absent and cannot leak. The forward return (`Close[t+N]/Close[t] - 1`) is the
*outcome* only and is never fed back into the signal; a bar is evaluated only
when its forward bar `t+N` exists. The future-invariance contract is locked by
`tests/test_backtest.py::test_causal_score_is_invariant_to_future_bars`.

```bash
# Single symbol, 14-day forward window over 2y of history
python -m volume_price_analysis.backtest AAPL --period 2y --horizon 14

# Pool several symbols for statistical power; subsample bars to reduce overlap
python -m volume_price_analysis.backtest AAPL MSFT NVDA SPY --horizon 14 --step 3
```

It reports, over all evaluated bars: directional hit rate, mean directional
return (`mean(sign(score) * forward_return)`), Spearman/Pearson IC, and a
breakdown by score bucket, by `|score|` gate threshold, and for the production
high-conviction gate (`|score|>=4 & ADX>=28 & IV<=50`). To compare a scoring
change, run the harness on the same symbols/period before and after and diff the
reports.

> Caveat: overlapping forward windows are autocorrelated, so single-symbol IC is
> directional, not conclusive — pool across symbols (and/or use `--step`) for a
> trustworthy read. Data is ~15-20 min delayed; this measures predictive
> information, not live tradeability.

### Environment Management

See [UV_SETUP.md](UV_SETUP.md) for detailed UV usage, or use traditional venv:

```bash
# Create environment
python -m venv .venv

# Activate
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install runtime dependencies (dev tools require uv — see UV_SETUP.md)
pip install -e .
```

## Morning Briefing Agent

An automated agent that runs daily at 8:30 AM ET to scan the market, analyze top
candidates, and email you an AI-generated briefing.

### Setup

1. Get an API key from [Google AI Studio](https://aistudio.google.com/apikey)
   (free) or [Anthropic Console](https://console.anthropic.com/) (paid)

2. Create a `.env` file:

   ```env
   AI_PROVIDER=gemini
   AI_PROVIDER_API_KEY=your-api-key
   EMAIL_FROM=your-email@gmail.com
   EMAIL_PASSWORD=your-gmail-app-password
   EMAIL_TO=recipient@example.com
   ```

3. Build and run with Docker:

   ```bash
   docker build -t volume-price-analysis .
   docker run --env-file .env volume-price-analysis
   ```

### Manual Run

```bash
# Dry run (prints to stdout, no email)
uv run python -m volume_price_analysis.agent.morning_agent --dry-run

# Skip AI, email raw data
uv run python -m volume_price_analysis.agent.morning_agent --no-ai
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `AI_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `AI_PROVIDER_API_KEY` | (required) | API key for chosen provider |
| `AI_MODEL` | provider default | Override model (e.g., `gemini-2.5-flash`) |
| `EMAIL_FROM` | (required) | Sender Gmail address |
| `EMAIL_PASSWORD` | (required) | Gmail App Password |
| `EMAIL_TO` | (required) | Recipient email address |
| `SCAN_UNIVERSE` | `full_market` | Symbol universe to scan |
| `MAX_DEEP_ANALYSIS` | `5` | Top N candidates for deep analysis |
| `TZ` | `America/New_York` | Container timezone |

## Data Source

This server uses [yfinance](https://github.com/ranaroussi/yfinance) to fetch
stock data from Yahoo Finance. Data is delayed and should not be used for live
trading decisions.

## Limitations

- Data is delayed (typically 15-20 minutes for US stocks)
- Yahoo Finance API rate limits may apply
- Not suitable for high-frequency or real-time trading
- Historical data availability varies by symbol

## Platform Support

- Supported on **macOS**, **Windows**, and **Linux**
- Docker deployment uses Linux containers (works on all platforms via Docker
  Desktop)
- On Windows, the scheduler handles `SIGINT` (Ctrl+C) only; `SIGTERM` is not
  available

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions welcome! Potential enhancements:

- Support for multiple data sources (Alpha Vantage, Polygon.io)
- Real-time data streams
- Chart generation
- Technical pattern detection

## Support

For issues or questions:

1. Check that yfinance can fetch data for your symbol
2. Verify your date ranges are valid
3. Ensure Python version is 3.14+
4. Check MCP configuration is correct

## Acknowledgments

Built using:

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
