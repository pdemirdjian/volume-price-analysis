---
name: indicator-validator
description: Review indicator calculations for mathematical correctness, edge case handling, and test coverage. Use when indicators are added, modified, or when asked to validate/review indicators.
---

# Indicator Validator

Review the indicator codebase for correctness and completeness.

## How to run

1. Read the agent instructions from `.claude/agents/indicator-validator.md`
2. Dispatch a `general-purpose` subagent (Task tool) with those instructions, targeting:
   - `src/volume_price_analysis/indicators.py`
   - `tests/test_indicators.py`
3. The subagent should provide a detailed report organized by indicator function, citing line numbers
4. This is a **read-only review** — the subagent should NOT modify any files
