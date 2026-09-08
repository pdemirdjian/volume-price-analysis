# volume-price-analysis

MCP server providing volume-price technical analysis tools for stock market data. Python 3.14+, uv, deployed as Docker container to ghcr.io.

## Architecture

```
src/volume_price_analysis/
├── server.py        # MCP adapter: list-tools + one dispatcher, nothing else
├── tools.py         # Tool registry: one ToolSpec (name/schema/run) per MCP tool
├── indicators.py    # Pure calculation functions (23 indicators)
├── data_fetcher.py  # DataSource protocol; YFinanceDataSource (prod) + InMemoryDataSource (tests)
├── analysis.py      # Reusable scan/analysis logic
└── agent/           # Morning briefing agent (scheduler, AI client, email)
```

## Adding New Indicators

1. Add calculation function to `indicators.py` (takes DataFrame, returns Series/dict)
2. Add test to `tests/test_indicators.py`
3. Append a `ToolSpec(name, description, input_schema, run)` to `TOOLS` in `tools.py` — the
   list-tools response and the dispatcher both derive from it; `server.py` needs no change
4. Add a case to `tests/test_tool_registry.py`'s `MINIMAL_ARGS` so the registry sweep covers it

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

Single-context: the domain glossary is [`CONTEXT.md`](CONTEXT.md) at the repo root, plus `docs/adr/`. Use its terms and avoid the synonyms it marks avoided. See `docs/agents/domain.md`.
