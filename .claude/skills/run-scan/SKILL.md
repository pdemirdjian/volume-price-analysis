---
name: run-scan
description: Run a local market scan for testing the morning briefing agent
disable-model-invocation: true
---

# Run Local Market Scan

Run the market scanning pipeline locally for testing and debugging.

## Prerequisites

Ensure `.env` is configured with required variables (see `.env.example`).

## Usage

### Quick scan (scan only, no AI briefing)

```bash
uv run python -c "
import asyncio
from volume_price_analysis.analysis import run_scan
results = asyncio.run(run_scan())
candidates = results.get('top_bullish', []) + results.get('top_bearish', [])
for r in candidates[:10]:
    print(f\"{r['symbol']:6s} score={r['composite_score']:.2f}\")
"
```

### Full briefing (scan + AI analysis + email)

```bash
uv run morning-briefing
```

### Scheduler (runs daily at configured time)

```bash
uv run morning-scheduler
# Options: --time HH:MM  --skip-holidays
```

## Debugging Tips

- If yfinance fails, check network access and rate limits
- Set `MAX_DEEP_ANALYSIS=1` in `.env` to speed up test runs
- Set `SCAN_UNIVERSE=test` if available, to use a smaller symbol set
