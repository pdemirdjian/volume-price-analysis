"""Market-regime check for the morning briefing (PDE-66).

Computes SPY's prior-session close against its 20-day SMA and demotes
high-conviction picks whose direction fights the prevailing tape. The
briefing audit (PDE-66) showed regime-fighting picks driving negative
expectancy; the follow-up re-grade showed the gate itself adds no edge,
so demoted picks are kept visible as flagged context rather than dropped.

Strictly causal: bars dated on or after the caller-supplied "today" are
excluded, so even an intraday run compares only the prior session's close;
no in-progress bar enters the check.
"""

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

REGIME_SMA_PERIOD = 20

_GATED_DIRECTIONS = ("bullish", "bearish")


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


def apply_regime_gate(scan_results: dict, regime: dict) -> dict:
    """Demote high-conviction picks whose direction fights the regime.

    Returns a copy of ``scan_results`` (input never mutated) where
    counter-regime candidates are moved from ``high_conviction_setups`` into
    ``regime_demoted`` (each copied with a ``regime_conflict`` note), the
    ``summary`` count is updated, and the regime verdict is attached under
    ``market_regime``. An unknown regime gates nothing. Demoted candidates
    still appear in their ``top_bullish``/``top_bearish`` lists, so they stay
    scan candidates — but they do lose high-conviction billing everywhere it
    matters, including priority for deep-analysis slots.
    """
    high_conviction = scan_results.get("high_conviction_setups") or []
    result = dict(scan_results)
    result["market_regime"] = regime

    verdict = regime.get("regime")
    if verdict not in _GATED_DIRECTIONS:
        result.setdefault("high_conviction_setups", [])
        result["regime_demoted"] = []
        return result

    kept: list[dict] = []
    demoted: list[dict] = []
    for candidate in high_conviction:
        score = candidate.get("composite_score")
        if score is None:
            kept.append(candidate)
            continue
        direction = "bullish" if score >= 0 else "bearish"
        if direction == verdict:
            kept.append(candidate)
        else:
            demoted.append(
                {
                    **candidate,
                    "regime_conflict": f"{direction} setup against a {verdict} tape",
                }
            )

    if demoted:
        logger.info(
            "Regime gate (%s tape): demoted %d of %d high-conviction pick(s): %s",
            verdict,
            len(demoted),
            len(high_conviction),
            ", ".join(c.get("symbol", "?") for c in demoted),
        )

    result["high_conviction_setups"] = kept
    result["regime_demoted"] = demoted
    summary = scan_results.get("summary")
    if isinstance(summary, dict):
        # summary.high_conviction is the *uncapped* count from run_scan while
        # high_conviction_setups is capped at 5, so subtract demotions from
        # the reported count rather than recounting the capped list.
        reported = summary.get("high_conviction")
        if isinstance(reported, int):
            new_count = max(reported - len(demoted), len(kept))
        else:
            new_count = len(kept)
        result["summary"] = {**summary, "high_conviction": new_count}
    return result


def format_regime_header(regime: dict, demoted_count: int = 0) -> str:
    """Render the regime verdict as a markdown line for the top of the email."""
    verdict = regime.get("regime", "unknown")
    if verdict not in _GATED_DIRECTIONS:
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
    if demoted_count:
        picks = "pick" if demoted_count == 1 else "picks"
        header += (
            f" {demoted_count} counter-regime {picks} demoted from "
            f"high-conviction (see Risk Warnings)."
        )
    return header
