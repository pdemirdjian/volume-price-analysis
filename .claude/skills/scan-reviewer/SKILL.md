---
name: scan-reviewer
description: Review market scanning logic for scoring correctness, concurrency safety, and failure handling. Use when scan/analysis code is modified or when asked to review scanning logic.
---

# Scan Reviewer

Review the scanning and analysis codebase for correctness and robustness.

## How to run

1. Dispatch the `scan-reviewer` subagent (Agent tool; defined in `.claude/agents/scan-reviewer.md`), targeting:
   - `src/volume_price_analysis/analysis.py`
   - `tests/test_analysis.py`
2. The subagent should provide a detailed report covering scoring, concurrency, failure handling, and test gaps, citing line numbers
3. This is a **read-only review** — the subagent should NOT modify any files
