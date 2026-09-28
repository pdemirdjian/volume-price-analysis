---
name: add-indicator
description: Add a new technical indicator to the MCP server with tests and tool wiring. Use when asked to add a new indicator, calculation, or analysis function.
---

# Add New Indicator

Follow these steps exactly to add a new indicator to the project.

## Arguments

- `$ARGUMENTS` — Name of the indicator (e.g., "RSI", "Bollinger Bands", "ADL")

## Step 1: Add calculation function to `indicators.py`

Read `src/volume_price_analysis/indicators.py` to understand existing patterns.

Every indicator function follows this signature:

```python
def calculate_<name>(df: pd.DataFrame, ...) -> pd.Series | dict:
```

Key conventions:
- Takes a DataFrame with columns: Open, High, Low, Close, Volume
- Additional parameters (periods, thresholds) have sensible defaults
- Returns a Series for single-value indicators or a dict for multi-value results
- Pure functions — no side effects, no data fetching

Add the new function following the same pattern.

## Step 2: Add tests to `tests/test_indicators.py`

Read `tests/test_indicators.py` and `tests/conftest.py` to see existing test patterns.

Tests use fixtures from conftest.py: `sample_stock_data`, `uptrend_data`, `downtrend_data`, `flat_price_data`.

Write tests that cover:
- Basic calculation returns expected shape/type
- Known trend behavior (e.g., uptrend should produce expected signal)
- Edge cases if applicable

## Step 3: Add MCP Tool definition in `tools.py`

Read `src/volume_price_analysis/tools.py` and find `TOOLS`.

Append a new `ToolSpec(name, description, input_schema, run)` entry following the existing pattern:
- Name: `calculate_<name>` (matching the function name)
- Description: Clear explanation of what the indicator measures
- Input schema: JSON Schema matching the function parameters
- Run: `_run_calculate_<name>` defined in Step 4

## Step 4: Add tool run function and registry test case

In the same `tools.py`, add an `async def _run_calculate_<name>(ctx: ToolContext)` that:
1. Extracts parameters from `ctx.args` and the symbol using `ctx.require_symbol()`
2. Calls `ctx.fetch()` for the symbol's data
3. Calls the indicator function
4. Returns the result as a dict

Add a case for the new tool to `MINIMAL_ARGS` in `tests/test_tool_registry.py`.

## Step 5: Verify

Run:
```bash
uv run pytest tests/test_indicators.py tests/test_tool_registry.py -v
uv run ruff check src/ tests/
uv run mypy src/
```

All must pass before considering the task complete.
