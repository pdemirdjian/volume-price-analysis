"""Market-regime check for the morning briefing (PDE-66).

Computes SPY's prior-session close against its 20-day SMA and flags
high-conviction picks whose direction fights the prevailing tape. The
briefing audit (PDE-66) showed regime-fighting picks driving negative
expectancy, but the follow-up full-history re-grade showed a gate adds no
edge — so the verdict is presented as context only: picks keep their
high-conviction billing, summary counts, and deep-analysis priority, and
counter-regime picks merely carry a ``regime_conflict`` note.

Strictly causal: bars dated on or after the caller-supplied "today" are
excluded, so even an intraday run compares only the prior session's close;
no in-progress bar enters the check.
"""

import logging
from collections.abc import Mapping
from datetime import date
from typing import Any, cast

import pandas as pd

from ..payloads import ScanResult

logger = logging.getLogger(__name__)

REGIME_SMA_PERIOD = 20

# The two decided verdicts `compute_market_regime` can return ("unknown" is
# the third, undecided one). Shared with `picks` so both agree on what counts
# as a directional tape.
REGIME_DIRECTIONS = ("bullish", "bearish")

# Breadth qualifier thresholds (PDE-210), read against the scan's
# ``summary.breadth.pct_above_sma20`` -- the same 20-day basis as the verdict.
# Participation is the share of counted sp500 symbols on the verdict's side of
# their SMA20 (above for bullish, below for bearish): at least
# BREADTH_BROAD_PCT is "broad", at most BREADTH_NARROW_PCT is "narrow", in
# between is "mixed". Fewer than BREADTH_MIN_SYMBOLS counted symbols (e.g. a
# small custom scan) gets no qualifier. Breadth qualifies the verdict only; it
# never flips it or changes ``regime_conflict``.
BREADTH_BROAD_PCT = 60.0
BREADTH_NARROW_PCT = 40.0
BREADTH_MIN_SYMBOLS = 20


def compute_market_regime(spy_data: pd.DataFrame | None, today: date | None = None) -> dict:
    """Classify the market regime from SPY daily data.

    Compares the last close against the SMA of the final ``REGIME_SMA_PERIOD``
    closes (NaNs dropped first). When ``today`` is given, bars dated on or
    after it are excluded first, keeping the check strictly causal even when
    the fetch happens mid-session. Returns a dict with ``regime`` set to
    ``"bullish"`` (close at or above the SMA), ``"bearish"`` (below), or
    ``"unknown"`` with a ``reason`` when the check cannot be computed —
    insufficient history never raises.
    """
    if spy_data is None or spy_data.empty or "Close" not in spy_data.columns:
        return {"regime": "unknown", "reason": "SPY data unavailable"}

    if today is not None and "Date" in spy_data.columns:
        spy_data = spy_data[pd.to_datetime(spy_data["Date"]).dt.date < today]
        if spy_data.empty:
            return {"regime": "unknown", "reason": "no SPY sessions before today"}

    closes = spy_data["Close"].dropna()
    if len(closes) < REGIME_SMA_PERIOD:
        return {
            "regime": "unknown",
            "reason": (
                f"insufficient SPY history ({len(closes)} closes, need {REGIME_SMA_PERIOD})"
            ),
        }

    window = closes.iloc[-REGIME_SMA_PERIOD:]
    close = float(window.iloc[-1])
    sma = float(window.mean())

    as_of = None
    if "Date" in spy_data.columns:
        last_date = spy_data.loc[closes.index[-1], "Date"]
        if pd.notna(last_date):
            as_of = pd.Timestamp(last_date).strftime("%Y-%m-%d")

    return {
        "regime": "bullish" if close >= sma else "bearish",
        "spy_close": round(close, 2),
        "sma20": round(sma, 2),
        "close_vs_sma_pct": round((close - sma) / sma * 100, 2),
        "as_of": as_of,
        "basis": f"SPY prior close vs {REGIME_SMA_PERIOD}-day SMA",
    }


def annotate_regime_conflicts(scan_results: Mapping[str, Any], regime: dict) -> ScanResult:
    """Attach the regime verdict and flag counter-regime high-conviction picks.

    Returns a copy of ``scan_results`` (input never mutated) with the regime
    verdict attached under ``market_regime`` and a ``regime_conflict`` note
    added to each high-conviction candidate whose direction fights the tape.
    A flagged symbol's entries in ``top_bullish``/``top_bearish`` carry the
    same note, so the flag is visible wherever the pick appears (the scan
    shares candidate dicts between those lists, so copies are annotated —
    never the originals). Annotation only: picks keep their list membership,
    summary counts, and deep-analysis priority. An unknown regime annotates
    nothing.
    """
    result = dict(scan_results)
    result["market_regime"] = regime
    high_conviction = scan_results.get("high_conviction_setups") or []
    result["high_conviction_setups"] = high_conviction

    verdict = regime.get("regime")
    if verdict not in REGIME_DIRECTIONS or not high_conviction:
        return cast(ScanResult, result)

    annotated: list[dict] = []
    conflict_notes: dict[str, str] = {}
    for candidate in high_conviction:
        score = candidate.get("composite_score")
        if score is None:
            annotated.append(candidate)
            continue
        direction = "bullish" if score >= 0 else "bearish"
        if direction == verdict:
            annotated.append(candidate)
        else:
            note = f"{direction} setup against a {verdict} tape"
            annotated.append({**candidate, "regime_conflict": note})
            conflict_notes[candidate.get("symbol", "?")] = note

    if conflict_notes:
        logger.info(
            "Regime check (%s tape): flagged %d of %d high-conviction pick(s): %s",
            verdict,
            len(conflict_notes),
            len(high_conviction),
            ", ".join(conflict_notes),
        )

    result["high_conviction_setups"] = annotated
    for key in ("top_bullish", "top_bearish"):
        candidates = scan_results.get(key)
        if not isinstance(candidates, list):
            continue
        result[key] = [
            {**c, "regime_conflict": conflict_notes[c["symbol"]]}
            if isinstance(c, dict) and c.get("symbol") in conflict_notes
            else c
            for c in candidates
        ]
    return cast(ScanResult, result)


def breadth_qualifier(verdict: str, breadth: Mapping[str, Any] | None) -> str | None:
    """Qualify a decided verdict as ``broad``, ``mixed``, or ``narrow`` from scan breadth.

    None when the verdict is undecided, breadth is missing or undefined, or
    fewer than ``BREADTH_MIN_SYMBOLS`` symbols were counted.
    """
    if verdict not in REGIME_DIRECTIONS or not breadth:
        return None
    pct_above = breadth.get("pct_above_sma20")
    if pct_above is None or breadth.get("symbols_counted", 0) < BREADTH_MIN_SYMBOLS:
        return None
    participation = pct_above if verdict == "bullish" else 100.0 - pct_above
    if participation >= BREADTH_BROAD_PCT:
        return "broad"
    if participation <= BREADTH_NARROW_PCT:
        return "narrow"
    return "mixed"


def _breadth_sentence(breadth: Mapping[str, Any]) -> str:
    sma50 = breadth.get("pct_above_sma50")
    sma50_part = f" ({sma50:.0f}% above their 50-day)" if sma50 is not None else ""
    ratio = breadth.get("up_down_volume_ratio")
    ratio_part = f"{ratio:.2f}" if ratio is not None else "n/a"
    return (
        f" Breadth: {breadth['pct_above_sma20']:.0f}% of {breadth['symbols_counted']} "
        f"S&P 500 symbols above their 20-day SMA{sma50_part}, up/down volume {ratio_part}, "
        f"net new 20-day highs {breadth.get('net_new_highs', 0):+d}."
    )


def format_regime_header(
    regime: dict, conflict_count: int = 0, breadth: Mapping[str, Any] | None = None
) -> str:
    """Render the regime verdict as a markdown line for the top of the email.

    ``breadth`` is the scan's ``summary.breadth``; when it yields a qualifier
    (see ``breadth_qualifier``) the verdict reads e.g. ``BULLISH (narrow)`` and
    a breadth sentence follows the SPY sentence. The verdict itself is unchanged.
    """
    verdict = regime.get("regime", "unknown")
    if verdict not in REGIME_DIRECTIONS:
        reason = regime.get("reason", "no reason recorded")
        return f"**Market Regime: UNKNOWN** — regime check unavailable ({reason})."

    qualifier = breadth_qualifier(verdict, breadth)
    label = verdict.upper() + (f" ({qualifier})" if qualifier else "")
    relation = "above" if verdict == "bullish" else "below"
    pct = abs(regime.get("close_vs_sma_pct", 0.0))
    as_of = regime.get("as_of")
    as_of_part = f" as of {as_of}" if as_of else ""
    header = (
        f"**Market Regime: {label}** — SPY closed at "
        f"{regime.get('spy_close', 0.0):.2f}, {pct:.1f}% {relation} its "
        f"{REGIME_SMA_PERIOD}-day SMA ({regime.get('sma20', 0.0):.2f}){as_of_part}."
    )
    if qualifier and breadth is not None:
        header += _breadth_sentence(breadth)
    if conflict_count:
        picks = "pick" if conflict_count == 1 else "picks"
        header += f" {conflict_count} high-conviction {picks} flagged as counter-regime."
    return header
