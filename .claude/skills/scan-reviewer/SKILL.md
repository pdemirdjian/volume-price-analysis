---
name: scan-reviewer
description: Review market scanning logic for scoring correctness, concurrency safety, and failure handling. Use when scan/analysis code is modified or when asked to review scanning logic.
---

# Scan Reviewer

Review the scanning and analysis codebase for correctness and robustness.

## How to run

1. Read the agent instructions from `.claude/agents/scan-reviewer.md`
2. Dispatch a `general-purpose` subagent (Task tool) with those instructions, targeting:
   - `src/volume_price_analysis/analysis.py`
   - `tests/test_analysis.py`
3. The subagent should provide a detailed report covering scoring, concurrency, failure handling, and test gaps, citing line numbers
4. This is a **read-only review** — the subagent should NOT modify any files
