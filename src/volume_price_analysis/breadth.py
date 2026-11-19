"""Market breadth from the scan's own bars (PDE-210).

Each scan worker reduces its symbol's frame (already cut at ``as_of``) to one
small :class:`BreadthObservation` with :func:`observe_breadth`; ``run_scan``
reduces those once with :func:`summarize_breadth` into ``summary.breadth``. No
extra fetches and no frames outlive the worker.

Only sp500 members are counted, whatever universe was scanned, so ETFs
(including leveraged and inverse ones) never skew the percentages or volumes.
"""

import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from .payloads import BreadthSummary

BREADTH_UNIVERSE = "sp500"
BREADTH_SHORT_SMA = 20
BREADTH_LONG_SMA = 50
# Sessions in the "new N-day high/low" window, the last session included: a new high
# is a last-session High strictly above every High of the prior N - 1 sessions.
BREADTH_HIGH_LOW_WINDOW = 20


@dataclass(frozen=True, slots=True)
class BreadthObservation:
    """One symbol's last session, as breadth needs it. Scalars only.

    ``above_sma20``/``above_sma50`` are None when there are fewer closes than
    the SMA needs. ``up_close`` is True for a close above the prior close,
    False for below, and None when unchanged. ``session`` is the last bar's
    date (None when the frame has no ``Date`` column).
    """

    symbol: str
    session: date | None
    above_sma20: bool | None
    above_sma50: bool | None
    up_close: bool | None
    volume: float
    new_high: bool
    new_low: bool


def _above_sma(closes: pd.Series, period: int) -> bool | None:
    if len(closes) < period:
        return None
    return bool(closes.iloc[-1] > closes.iloc[-period:].mean())


def observe_breadth(symbol: str, data: pd.DataFrame) -> BreadthObservation | None:
    """Reduce a symbol's OHLCV frame to its last-session breadth observation.

    Reads only the bars it is handed: callers apply the ``as_of`` cutoff first.
    Returns None with fewer than two bars or a missing last close or volume.
    A new high (low) is a last-session High (Low) strictly above (below) every
    High (Low) of the prior ``BREADTH_HIGH_LOW_WINDOW - 1`` sessions, so ties and
    flat series are neither; with fewer than ``BREADTH_HIGH_LOW_WINDOW`` bars
    both are False.
    """
    if len(data) < 2:
        return None
    last_close = data["Close"].iloc[-1]
    prev_close = data["Close"].iloc[-2]
    volume = data["Volume"].iloc[-1]
    if pd.isna(last_close) or pd.isna(prev_close) or pd.isna(volume):
        return None

    closes = data["Close"].dropna()
    new_high = new_low = False
    if len(data) >= BREADTH_HIGH_LOW_WINDOW:
        prior = data.iloc[-BREADTH_HIGH_LOW_WINDOW:-1]
        new_high = bool(data["High"].iloc[-1] > prior["High"].max())
        new_low = bool(data["Low"].iloc[-1] < prior["Low"].min())
    session = None
    if "Date" in data.columns and pd.notna(data["Date"].iloc[-1]):
        session = pd.Timestamp(data["Date"].iloc[-1]).date()
    return BreadthObservation(
        symbol=symbol,
        session=session,
        above_sma20=_above_sma(closes, BREADTH_SHORT_SMA),
        above_sma50=_above_sma(closes, BREADTH_LONG_SMA),
        up_close=None if last_close == prev_close else bool(last_close > prev_close),
        volume=float(volume),
        new_high=new_high,
        new_low=new_low,
    )


def _pct(flags: list[bool | None]) -> float | None:
    known = [f for f in flags if f is not None]
    if not known:
        return None
    return round(100.0 * sum(known) / len(known), 1)


def summarize_breadth(
    observations: Iterable[BreadthObservation], members: Collection[str]
) -> BreadthSummary:
    """Summarise the observations of ``members`` (the sp500) into the scan's breadth.

    Each symbol counts once (the first observation wins), and only observations
    dated on the latest session among the members count: a symbol whose data
    ends a session early would otherwise add a prior session to the totals.
    Percentages are over the counted symbols whose SMA is defined (None when
    none is). ``up_down_volume_ratio`` is last-session volume of up closes over
    down closes (unchanged closes count in neither); None when there is no down
    volume, so the result always serialises as strict JSON.
    """
    by_symbol: dict[str, BreadthObservation] = {}
    for o in observations:
        if o.symbol in members:
            by_symbol.setdefault(o.symbol, o)
    sessions = [o.session for o in by_symbol.values() if o.session is not None]
    latest = max(sessions, default=None)
    counted = [o for o in by_symbol.values() if latest is None or o.session == latest]
    up_volume = sum(o.volume for o in counted if o.up_close is True)
    down_volume = sum(o.volume for o in counted if o.up_close is False)
    ratio = up_volume / down_volume if down_volume > 0 else None
    new_highs = sum(o.new_high for o in counted)
    new_lows = sum(o.new_low for o in counted)
    return {
        "universe": BREADTH_UNIVERSE,
        "symbols_counted": len(counted),
        "pct_above_sma20": _pct([o.above_sma20 for o in counted]),
        "pct_above_sma50": _pct([o.above_sma50 for o in counted]),
        "up_volume": float(up_volume),
        "down_volume": float(down_volume),
        "up_down_volume_ratio": (
            round(ratio, 2) if ratio is not None and math.isfinite(ratio) else None
        ),
        "new_highs": new_highs,
        "new_lows": new_lows,
        "net_new_highs": new_highs - new_lows,
    }
