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

## Step 3: Register the tool in `tools.py`

Read `src/volume_price_analysis/tools.py` and the existing `ToolSpec` entries in `TOOLS`.

Add an `async def _run_calculate_<name>(ctx: ToolContext) -> dict` handler that reads parameters from `ctx.args`, gets the DataFrame via `ctx.fetch()`, calls the indicator function, and returns a result dict. Then add a `ToolSpec` for it to `TOOLS`, following the existing entries:
- Name: `calculate_<name>` (matching the function name)
- Description: Clear explanation of what the indicator measures
- Input schema: JSON Schema matching the function parameters

`server.py` needs no change: the list-tools response and the dispatcher both derive from `TOOLS`.

## Step 4: Cover it in the registry sweep

Add an entry for the new tool name to `MINIMAL_ARGS` in `tests/test_tool_registry.py`; a guard test fails if the registry and `MINIMAL_ARGS` diverge.

## Step 5: Verify

Run:
```bash
uv run pytest tests/test_indicators.py -v
uv run ruff check src/ tests/
uv run mypy src/
```

All must pass before considering the task complete.
