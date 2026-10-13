# ADR-0002: Keep scan scoring (`score_symbol`) separate from the fetch step

**Status:** Accepted (2026-10-13)

## Context

A scan reaches each symbol through three layers in `analysis.py`:

1. `_analyze_symbol_async` runs under the scan's semaphore and buckets the
   outcome: `InsufficientDataError` is a skip, any other exception an error.
2. `analyze_single_symbol` fetches the symbol's history through the
   `DataSource`, applies the optional `as_of` cutoff, and calls `score_symbol`.
3. `score_symbol` is the pure core: an OHLCV DataFrame in, a candidate dict
   (or `None` when a filter rejects it) out, with no I/O.

Layer 2 is thin, so architecture reviews keep proposing to fold it into
`score_symbol` (PDE-144 raised it again as PDE-156). The cost people point at
was real but sat in the tests: one test existed only to assert the two layers
agree, and the server's `scan_candidates` tests patched `analyze_single_symbol`
with hand-built candidate dicts, so the real scoring path never ran under the
server.

## Decision

Keep the split. `score_symbol` stays a pure function of an already-fetched
frame; fetching stays in `analyze_single_symbol`. Neither function's name,
signature, or behaviour changes.

- **Mock-free scoring tests.** Scan scoring and filters are tested on synthetic,
  deterministic frames by calling `score_symbol` directly, with no data source
  and no patching (`tests/test_analysis.py::TestScoreSymbol`).
- **Backtest contract purity.** `tests/test_backtest_contract.py` pins the
  backtest harness to classify a bar as high-conviction exactly when the scan
  would, by scoring the same synthetic frame through `score_symbol` and the
  harness side by side. That only works while `score_symbol` takes a frame and
  does nothing else.
- **Parity with `run_options_analysis`,** which also takes a DataFrame and
  leaves fetching to its caller. One convention for "analysis core takes data"
  across the module.

Folding the two would make one function accept either a frame or a data
source, blurring the seam the tests and the backtest depend on.

## Consequences

- The fetch wrapper stays a few lines long and has no test of its own beyond
  what exercises it end to end. That is intended; do not add a test that only
  asserts it agrees with `score_symbol`.
- Server-level `scan_candidates` tests drive the real dispatcher with an
  injected `InMemoryDataSource` and patch nothing on the scan path, so they
  cover fetch, scoring, filters, and skip/error bucketing together.
- `run_scan` tests in `tests/test_analysis.py` may still patch
  `analyze_single_symbol`, but only to control concurrency and error
  aggregation, not to stand in for scoring.
- Revisit only if `score_symbol` needs I/O of its own (e.g. earnings dates in
  the gate), which would remove the reason it is separate.
