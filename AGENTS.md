# volume-price-analysis

MCP server providing volume-price technical analysis tools for stock market data. Python 3.14+, uv, deployed as Docker container to ghcr.io.

## Architecture

```
src/volume_price_analysis/
├── server.py        # MCP tool definitions & handlers
├── indicators.py    # Pure calculation functions (23 indicators)
├── data_fetcher.py  # DataSource protocol; YFinanceDataSource (prod) + InMemoryDataSource (tests)
├── analysis.py      # Reusable scan/analysis logic
└── agent/           # Morning briefing agent (scheduler, AI client, email)
```

## Adding New Indicators

1. Add calculation function to `indicators.py` (takes DataFrame, returns Series/dict)
2. Add test to `tests/test_indicators.py`
3. Add `Tool()` definition in `server.py` `handle_list_tools()`
4. Add handler case in `handle_call_tool()`

Use skills `indicator-validator` and `scan-reviewer` when modifying indicators or scan logic.

## Code Style

- Line length: 100, double quotes, spaces (ruff-enforced)
- Coverage threshold: 80% (`--cov-fail-under=80`)
- Lint rules: pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade, pep8-naming, flake8-async
- Always run `uv run ruff check --fix` and `uv run mypy src/` before committing

## Constraints

- Never hardcode API keys or credentials — use env vars, store in `.env` (gitignored)
- PR titles must follow conventional commits (`feat:`, `fix:`, `chore:`, etc.)
- Don't break MCP tool signatures — clients depend on them

## Agent skills

### Issue tracker

Issues are tracked in Linear (team `pdemirdjian`, issue keys `PDE-*`), accessed via the Linear MCP tools. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical state labels exist verbatim in Linear; categories map `bug`→`Bug`, `enhancement`→`Feature`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` at the repo root plus `docs/adr/`. See `docs/agents/domain.md`.
