"""Option-market snapshot for a deep-analysed symbol (PDE-247).

Pure functions over an :class:`OptionChain` (fetched through
``DataSource.fetch_option_chain``):
pick the expiry nearest the holding period, then read the at-the-money (ATM)
call and put for real implied volatility, the IV / HV20 ratio, the straddle's
expected move to expiry, and liquidity flags.

A chain whose ATM quotes look stale or unusable (zero or one-sided bids before
the open, crossed quotes, yfinance's ~0 placeholder IV) yields
``{"available": False, "reason": ...}`` rather than numbers, so bad data never
reaches the model. Illiquidity is a flag on an otherwise available block; it
never demotes or drops a pick.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from .payloads import OptionMarket

# Per-leg ATM open interest below which the chain is flagged thin.
MIN_LEG_OPEN_INTEREST = 100
# Widest ATM leg bid-ask spread, as % of mid, before the chain is flagged wide.
MAX_SPREAD_PCT = 10.0
# IV / HV20 bands: above RICH options price more movement than realised, below CHEAP less.
IV_HV_RICH = 1.2
IV_HV_CHEAP = 0.8
# yfinance reports ~1e-5 IV for strikes it has no live quote for; treat as missing.
MIN_USABLE_IV = 0.01

_LEG_COLUMNS = ("strike", "bid", "ask", "impliedVolatility", "openInterest")


@dataclass(frozen=True)
class OptionChain:
    """One expiry's option chain, as returned by ``DataSource.fetch_option_chain``.

    ``calls`` and ``puts`` carry at least ``strike, bid, ask, impliedVolatility,
    openInterest`` (yfinance's column names); ``impliedVolatility`` is an
    annualised fraction.
    """

    expiry: date
    calls: pd.DataFrame
    puts: pd.DataFrame


def select_expiry(expiries: Iterable[date], *, today: date, target_dte: int) -> date | None:
    """Return the listed expiry closest to ``target_dte`` days out, or None.

    Same-day and past expiries are skipped. A tie prefers the later expiry
    (more time value, less gamma).
    """
    future = [e for e in expiries if (e - today).days > 0]
    if not future:
        return None
    return min(future, key=lambda e: (abs((e - today).days - target_dte), -e.toordinal()))


def _unavailable(reason: str) -> OptionMarket:
    return {"available": False, "reason": reason}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None


def _usable_leg(leg: pd.Series) -> tuple[float, float, float, int] | str:
    """Return ``(mid, iv, spread % of mid, open interest)``, or why the leg is unusable."""
    bid, ask, iv = _finite(leg["bid"]), _finite(leg["ask"]), _finite(leg["impliedVolatility"])
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return "ATM quotes have no live bid/ask (stale or pre-market)"
    if ask < bid:
        return "ATM quotes are crossed"
    if iv is None or iv < MIN_USABLE_IV:
        return "ATM implied volatility is missing"
    mid = (bid + ask) / 2
    oi = _finite(leg["openInterest"])
    return mid, iv, (ask - bid) / mid * 100, 0 if oi is None else int(oi)


def option_market_snapshot(
    chain: OptionChain, *, price: float, hv20: float | None, today: date
) -> OptionMarket:
    """Summarise the ATM straddle of ``chain`` against ``price`` and realised vol.

    Args:
        chain: One expiry's calls and puts (``strike, bid, ask,
            impliedVolatility, openInterest``).
        price: Underlying reference price (latest close).
        hv20: Annualised 20-day historical volatility as a fraction, or None.
        today: The briefing date, for days to expiry.

    Returns:
        An ``OptionMarket`` block; ``available`` is False with a ``reason``
        whenever the chain cannot be trusted.
    """
    spot = _finite(price)
    if spot is None or spot <= 0:
        return _unavailable("no usable underlying price")
    days = (chain.expiry - today).days
    if days <= 0:
        return _unavailable("expiry is not in the future")
    if any(c not in frame.columns for frame in (chain.calls, chain.puts) for c in _LEG_COLUMNS):
        return _unavailable("option chain is missing quote columns")

    calls = chain.calls.drop_duplicates("strike").set_index("strike")
    puts = chain.puts.drop_duplicates("strike").set_index("strike")
    strikes = calls.index.intersection(puts.index)
    if strikes.empty:
        return _unavailable("no strike listed for both calls and puts")
    strike = float(min(strikes, key=lambda k: abs(float(k) - spot)))
    call, put = calls.loc[strike], puts.loc[strike]

    legs = []
    for leg in (call, put):
        usable = _usable_leg(leg)
        if isinstance(usable, str):
            return _unavailable(usable)
        legs.append(usable)

    atm_iv = (legs[0][1] + legs[1][1]) / 2
    straddle = legs[0][0] + legs[1][0]
    spread_pct = max(leg[2] for leg in legs)
    hv = _finite(hv20)
    ratio = atm_iv / hv if hv is not None and hv > 0 else None
    if ratio is None:
        label = None
    elif ratio > IV_HV_RICH:
        label = "rich"
    elif ratio < IV_HV_CHEAP:
        label = "cheap"
    else:
        label = "fair"

    flags = []
    if min(leg[3] for leg in legs) < MIN_LEG_OPEN_INTEREST:
        flags.append(f"low ATM open interest (<{MIN_LEG_OPEN_INTEREST} per leg)")
    if spread_pct > MAX_SPREAD_PCT:
        flags.append(f"wide ATM bid-ask spread ({spread_pct:.0f}% of mid)")

    return {
        "available": True,
        "expiry": chain.expiry.isoformat(),
        "days_to_expiry": days,
        "atm_strike": strike,
        "atm_iv": round(atm_iv, 4),
        "iv_hv20_ratio": None if ratio is None else round(ratio, 2),
        "iv_vs_hv": label,
        "expected_move_dollars": round(straddle, 2),
        "expected_move_pct": round(straddle / spot * 100, 2),
        "atm_open_interest": legs[0][3] + legs[1][3],
        "atm_spread_pct": round(spread_pct, 2),
        "liquid": not flags,
        "liquidity_flags": flags,
    }
