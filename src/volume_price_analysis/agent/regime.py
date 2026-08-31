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
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

REGIME_SMA_PERIOD = 20

_REGIME_DIRECTIONS = ("bullish", "bearish")


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


def annotate_regime_conflicts(scan_results: dict, regime: dict) -> dict:
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
    if verdict not in _REGIME_DIRECTIONS or not high_conviction:
        return result

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
    return result


def format_regime_header(regime: dict, conflict_count: int = 0) -> str:
    """Render the regime verdict as a markdown line for the top of the email."""
    verdict = regime.get("regime", "unknown")
    if verdict not in _REGIME_DIRECTIONS:
        reason = regime.get("reason", "no reason recorded")
        return f"**Market Regime: UNKNOWN** — regime check unavailable ({reason})."

    relation = "above" if verdict == "bullish" else "below"
    pct = abs(regime.get("close_vs_sma_pct", 0.0))
    as_of = regime.get("as_of")
    as_of_part = f" as of {as_of}" if as_of else ""
    header = (
        f"**Market Regime: {verdict.upper()}** — SPY closed at "
        f"{regime.get('spy_close', 0.0):.2f}, {pct:.1f}% {relation} its "
        f"{REGIME_SMA_PERIOD}-day SMA ({regime.get('sma20', 0.0):.2f}){as_of_part}."
    )
    if conflict_count:
        picks = "pick" if conflict_count == 1 else "picks"
        header += f" {conflict_count} high-conviction {picks} flagged as counter-regime."
    return header
