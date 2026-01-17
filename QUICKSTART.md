# Quick Start Guide

Get the Volume-Price Analysis MCP server running in 5 minutes.

## Step 1: Install

### Option A: With UV (Recommended - 10x faster!)

```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or visit https://github.com/astral-sh/uv for Windows

# Navigate to the project directory
cd volume-price-analysis

# Create virtual environment and install
uv venv
source .venv/bin/activate  # macOS/Linux (.venv\Scripts\activate on Windows)
uv pip install -e ".[dev]"
```

### Option B: With pip (Traditional)

```bash
# Navigate to the project directory
cd volume-price-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux (.venv\Scripts\activate on Windows)

# Install the package
pip install -e .
```

## Step 2: Test It Works

### Run Example Script

```bash
python example_usage.py
```

You should see analysis output for Apple stock (AAPL). If this works,
everything is installed correctly!

### Run Test Suite (Optional but Recommended)

```bash
# Run all tests
pytest

# Or with coverage
pytest --cov=src/volume_price_analysis --cov-report=term-missing
```

If tests pass, you're ready to go! 🚀

## Step 3: Configure with Claude Code

Edit your MCP settings file:

**macOS/Linux**: `~/.config/claude-code/mcp_settings.json`

**Windows**: `%APPDATA%\claude-code\mcp_settings.json`

Add this configuration:

```json
{
  "mcpServers": {
    "volume-price-analysis": {
      "command": "python",
      "args": ["-m", "volume_price_analysis.server"]
    }
  }
}
```

**Note**: If using a virtual environment, use the full path to Python:

```bash
# Find your Python path
which python  # macOS/Linux
where python  # Windows
```

Then update the command to use that full path.

## Step 4: Restart Claude Code

Restart Claude Code to load the new MCP server.

## Step 5: Try It Out

Ask Claude Code:

```text
Analyze Apple stock (AAPL) volume and price trends for the last 3 months
```

or

```text
Show me a comprehensive volume-price analysis for Tesla
```

or

```text
Calculate the VWAP and volume profile for Microsoft over the past year
```

## What You Get

The server provides 7 tools:

1. **get_stock_data** - Raw stock data
2. **calculate_obv** - On-Balance Volume
3. **calculate_vwap** - Volume Weighted Average Price
4. **calculate_volume_profile** - Volume distribution by price
5. **calculate_mfi** - Money Flow Index
6. **analyze_volume_trends** - Volume trend analysis
7. **comprehensive_analysis** - Everything at once!

## Common Issues

### "No module named mcp"

Install dependencies:

```bash
pip install -e .
```

### "yfinance" errors

Make sure you have internet connection. Yahoo Finance may occasionally have
rate limits.

### Server not appearing in Claude Code

1. Check the MCP settings file path is correct
2. Verify the Python path in the configuration
3. Restart Claude Code
4. Check Claude Code logs for errors

### Permission denied

Make the example script executable:

```bash
chmod +x example_usage.py
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Modify `example_usage.py` to analyze different stocks
- Try different time periods: "1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"
- Experiment with specific date ranges using start_date and end_date

## Support

If something isn't working:

1. Verify Python version: `python --version` (need 3.10+)
2. Test the example script first
3. Check MCP server logs in Claude Code
4. Make sure the stock symbol is valid

Happy analyzing! 📈
