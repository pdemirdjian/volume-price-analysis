# CONTEXT

Domain glossary for volume-price-analysis. One entry per concept the code already
uses, named the way the code names it. When writing issues, tests, comments, or
docs, use these terms and avoid the synonyms marked _Avoided_.

Single-context repo: this file plus `docs/adr/`. See `docs/agents/domain.md`.

## symbol

The identifier of a tradable instrument, e.g. `"AAPL"`. It is the key everything
else hangs off: universes are lists of symbols (`analysis.get_universes`), the
data source fetches per symbol (`data_fetcher.DataSource.fetch`, and the
module-level `data_fetcher.fetch_stock_data`), and every candidate, pick, and MCP
tool argument carries a `symbol` field.

_Avoided:_ **ticker** and **stock** as the domain term — say symbol in new code,
tests, issues, and docs. Both words survive in places that are not free to
change: the MCP tool name `get_stock_data` and the tool descriptions in
`server.py` and `analysis.py` ("Stock ticker symbol") are part of the client
contract, and `data_fetcher.fetch_stock_data` plus the `for stock in
pts.get_stocks_by_index(...)` loop in `analysis._build_sp500_symbols` are legacy
identifiers (the latter iterating a third-party payload). Existing names stay;
new ones say symbol.

## scan

The universe-wide run that scores every symbol in a universe and returns ranked
candidates: `analysis.run_scan`. It fans out over symbols with bounded
concurrency (`MAX_CONCURRENT_SCANS`), buckets each symbol as scanned, skipped
(insufficient history, `MIN_SCAN_HISTORY`), or errored, sorts survivors by
absolute composite score, and returns a dict of `scan_parameters`, `summary`,
`high_conviction_setups`, `top_bullish`, `top_bearish`, and `errors`. The scan
result dict is the contract every downstream consumer reads.

## candidate

One symbol the scan scored and kept — a plain dict with at least `symbol`,
`composite_score`, `adx`, and the volatility percentile, produced by
`analysis.score_symbol` and collected in `analysis.run_scan`. Candidate dicts are
shared between the scan's result lists, so annotators copy rather than mutate
them (`agent/picks.annotate_conviction`, `agent/regime.annotate_regime_conflicts`).

_Avoided:_ **signal** and **setup** as names for the thing itself. `signal_quality`
is a field on a candidate and `high_conviction_setups` is a result-list key, but
the entity is a candidate.

## briefing

The morning document delivered by email: `agent/morning_agent.run_morning_briefing`,
which fetches the regime, runs the scan, deep-analyses the top symbols, annotates
conviction, asks the AI client for prose, and sends the result. Its outcome is a
`BriefingRunResult` (`degraded`, `reason`, `regime`, `symbols_analyzed`,
`email_sent`). The body is assembled by `build_briefing_body` and, when the AI
call fails, by `_fallback_briefing`.

## pick

A candidate rendered as a row of the briefing's pick table: the frozen
`agent/picks.Pick` dataclass (`symbol`, `direction`, `conviction`, `score`,
`price`, `regime_conflict`, `earnings_warning`). `build_picks` collects every
scan candidate once in briefing priority order; `render_picks_table` renders them
under the fixed `PICK_TABLE_HEADER` column order, which audits parse.

## conviction

The programmatic label on a pick: `HIGH`, `MEDIUM`, or `LOW`
(`agent/picks.Conviction` / `CONVICTIONS`), derived by `agent/picks.conviction_for`.
`HIGH` means the candidate clears the high-conviction gate (or the scan already
listed it under `high_conviction_setups`); `MEDIUM` means absolute composite score
at or above `STRONG_SCORE` or `signal_quality == "high"`; otherwise `LOW`.
`annotate_conviction` stamps the field onto every candidate before the AI prompt
is built, so the model echoes the label rather than inventing one.

## regime

The market-wide trend verdict computed from SPY daily data:
`agent/regime.compute_market_regime` compares the last close against the SMA of
the final `REGIME_SMA_PERIOD` (20) closes and returns `bullish`, `bearish`, or
`unknown` with a `reason`. `annotate_regime_conflicts` attaches the verdict under
`market_regime` and adds a `regime_conflict` note to any high-conviction candidate
whose direction opposes the verdict ("bullish setup against a bearish tape");
`format_regime_header` renders the header line with the conflict count.

_Avoided:_ **tape** as a term of its own — it appears only inside the conflict
note's wording.

## high-conviction gate

The three-threshold predicate on a candidate, `analysis.passes_high_conviction_gate`.
A candidate qualifies only when all three hold, as the constants have them today:
absolute `composite_score` >= `HIGH_CONVICTION_MIN_ABS_SCORE` (4.0), `adx` >=
`HIGH_CONVICTION_MIN_ADX` (28.0), and the HV percentile (`hv_percentile`, falling
back to `iv_percentile`, an HV-based proxy) <= `HIGH_CONVICTION_MAX_HV_PERCENTILE`
(50.0). Missing or non-numeric fields fail the gate rather than raise. The `adx`
read is the composite's adaptive-period ADX from
`indicators.composite_adx_period`: ADX(10) for `holding_period <= 14`, else
ADX(14); the scan reports the period it used as `scan_parameters.adx_period`. The
gate is shared by the scan (`analysis.run_scan`), the briefing picks
(`agent/picks.py`), and the backtest (`backtest.py`, whose `_HC_*` constants and
`_ADX_PERIOD` follow production).

## tool registry

The record list that drives both MCP list and call dispatch: `tools.TOOLS`, a
tuple of `tools.ToolSpec` (`name`, `description`, `input_schema`, `run`,
`default_period`). `server.handle_list_tools` maps it to `Tool()` objects and
`server.dispatch` looks a record up by name and awaits its `run` — there is no
per-tool branching anywhere in `server.py`, which is only the MCP adapter and
that one dispatcher. Adding a tool means appending one record.

_Avoided:_ **handler** for the per-tool callable — it is a registry record's
`run`. "Handler" is reserved for the MCP protocol wrappers (`_on_list_tools`,
`_on_call_tool`, `handle_call_tool`).

## tool context

What one tool invocation is allowed to touch: `tools.ToolContext`, carrying the
parsed `args`, the `data_source` to read market data through, and a `fetch()`
that lazily resolves the standard symbol/period/start/end arguments into an
OHLCV frame. `fetch()` is memoised (one invocation is one fetch) and hands out a
copy, so no tool can mutate the frame the data source owns. `scan_candidates` is
the one tool that never calls `fetch()` — it fetches per symbol inside
`analysis.run_scan`.
