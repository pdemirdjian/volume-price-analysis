---
name: indicator-validator
description: Review indicator calculations for mathematical correctness, edge case handling, and test coverage. Use when indicators are added, modified, or when asked to validate/review indicators.
---

# Indicator Validator

Review the indicator codebase for correctness and completeness.

## How to run

1. Dispatch the `indicator-validator` subagent (Agent tool; defined in `.claude/agents/indicator-validator.md`), targeting:
   - `src/volume_price_analysis/indicators.py`
   - `tests/test_indicators.py`
2. The subagent should provide a detailed report organized by indicator function, citing line numbers
3. This is a **read-only review** — the subagent should NOT modify any files
