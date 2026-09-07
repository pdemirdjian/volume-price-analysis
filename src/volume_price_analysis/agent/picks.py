"""Deterministic pick block for the morning briefing (PDE-69).

The briefing audit found conviction labels drifting between briefings
("High Conviction", "High Signal Quality", nothing at all) because the model
was left to invent them. This module is the single home of the conviction
vocabulary: every pick gets exactly one of ``HIGH`` / ``MEDIUM`` / ``LOW``,
derived from the scan's own gates, and the picks are rendered as a fixed
markdown table that audits can parse without reading prose.

Conviction rule (shares its thresholds with ``analysis.run_scan``):

- ``HIGH``: the scan's high-conviction gate held — |score| >= 4, ADX >= 28 and
  HV percentile <= 50 (``analysis.passes_high_conviction_gate``). Evaluated on
  the candidate itself, not on membership in ``high_conviction_setups``: the
  scan truncates that list to five, but the summary counts every qualifier.
- ``MEDIUM``: one strong leg but not the full gate — |score| >= 4 *or* the
  composite's ``signal_quality`` is ``"high"`` (ADX > 30).
- ``LOW``: everything else that passed the scan filters.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from ..analysis import HIGH_CONVICTION_MIN_ABS_SCORE, passes_high_conviction_gate

Conviction = Literal["HIGH", "MEDIUM", "LOW"]
CONVICTIONS: tuple[Conviction, ...] = ("HIGH", "MEDIUM", "LOW")

# |composite_score| at or above which a pick counts as strongly scored (the
# score leg of the high-conviction gate).
STRONG_SCORE = HIGH_CONVICTION_MIN_ABS_SCORE

_CANDIDATE_LISTS = ("high_conviction_setups", "top_bullish", "top_bearish")

# Column order of the rendered table. Fixed on purpose: audits parse it.
PICK_TABLE_HEADER = "| Symbol | Direction | Conviction | Score | Price | Flags |"
_PICK_TABLE_RULE = "|---|---|---|---|---|---|"


@dataclass(frozen=True)
class Pick:
    """One row of the pick block."""

    symbol: str
    direction: str
    conviction: Conviction
    score: float
    price: float | None = None
    regime_conflict: bool = False
    earnings_warning: str | None = None


def conviction_for(candidate: dict, *, high_conviction: bool = False) -> Conviction:
    """Classify one scan candidate.

    ``high_conviction`` force-qualifies a candidate the scan already listed
    under ``high_conviction_setups``; otherwise the gate is re-evaluated from
    the candidate's own fields, so qualifiers beyond the scan's five-entry cap
    still read HIGH.
    """
    if high_conviction or passes_high_conviction_gate(candidate):
        return "HIGH"
    if _score_of(candidate) >= STRONG_SCORE or candidate.get("signal_quality") == "high":
        return "MEDIUM"
    return "LOW"


def _score_of(candidate: dict) -> float:
    """Absolute composite score, or 0.0 when missing or non-numeric."""
    try:
        return abs(float(candidate.get("composite_score") or 0.0))
    except TypeError, ValueError:
        return 0.0


def _high_conviction_symbols(scan_results: dict) -> set[str]:
    return {
        c["symbol"]
        for c in scan_results.get("high_conviction_setups") or []
        if isinstance(c, dict) and c.get("symbol")
    }


def annotate_conviction(scan_results: dict) -> dict:
    """Return a copy of ``scan_results`` with ``conviction`` on every candidate.

    The model is told to echo this field verbatim, so the label it prints and
    the label in the pick table can never disagree. Input is never mutated;
    the scan shares candidate dicts between lists, so copies are annotated.
    """
    high = _high_conviction_symbols(scan_results)
    result = dict(scan_results)
    for key in _CANDIDATE_LISTS:
        candidates = scan_results.get(key)
        if not isinstance(candidates, list):
            continue
        result[key] = [
            {**c, "conviction": conviction_for(c, high_conviction=c.get("symbol") in high)}
            if isinstance(c, dict)
            else c
            for c in candidates
        ]
    return result


def build_picks(scan_results: dict, deep_analyses: Iterable[dict] = ()) -> list[Pick]:
    """Collect every scan candidate once, in briefing priority order.

    Order matches ``_get_top_symbols``: high-conviction setups first, then the
    bullish list, then the bearish list. A deep analysis for a symbol supplies
    its price and score (the consistency rule: deep-analysis values win) and
    any earnings warning attached to it.
    """
    deep_by_symbol = {
        a["symbol"]: a for a in deep_analyses if isinstance(a, dict) and a.get("symbol")
    }
    high = _high_conviction_symbols(scan_results)
    picks: list[Pick] = []
    seen: set[str] = set()
    for key in _CANDIDATE_LISTS:
        for candidate in scan_results.get(key) or []:
            if not isinstance(candidate, dict):
                continue
            symbol = candidate.get("symbol")
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            deep = deep_by_symbol.get(symbol, {})
            composite = deep.get("composite_signal")
            deep_score = composite.get("score") if isinstance(composite, dict) else None
            raw_score = candidate.get("composite_score") if deep_score is None else deep_score
            try:
                score = float(raw_score or 0.0)
            except TypeError, ValueError:
                score = 0.0
            price = deep.get("latest_price", candidate.get("latest_price"))
            picks.append(
                Pick(
                    symbol=symbol,
                    direction="bullish" if score >= 0 else "bearish",
                    conviction=conviction_for(candidate, high_conviction=symbol in high),
                    score=round(score, 2),
                    price=None if price is None else float(price),
                    regime_conflict=bool(candidate.get("regime_conflict")),
                    earnings_warning=deep.get("earnings_warning"),
                )
            )
    return picks


def _flags(pick: Pick) -> str:
    flags = []
    if pick.regime_conflict:
        flags.append("counter-regime")
    if pick.earnings_warning:
        flags.append(pick.earnings_warning)
    return "; ".join(flags) or "—"


def render_picks_table(picks: Iterable[Pick]) -> str:
    """Render the pick block as a fixed-column markdown table.

    Always emits the header so a briefing with no picks still carries the
    block (audits can then distinguish "no picks" from "block missing").
    """
    rows = [PICK_TABLE_HEADER, _PICK_TABLE_RULE]
    for p in picks:
        price = "—" if p.price is None else f"{p.price:.2f}"
        rows.append(
            f"| {p.symbol} | {p.direction} | {p.conviction} | {p.score:+.2f} | {price} "
            f"| {_flags(p)} |"
        )
    if len(rows) == 2:
        rows.append("| — | — | — | — | — | no candidates passed the scan filters |")
    return "\n".join(rows)
