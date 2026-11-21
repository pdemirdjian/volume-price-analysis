"""Option-market snapshot for a deep-analysed symbol (PDE-247).

Pure functions over an :class:`OptionChain` (fetched through
``DataSource.fetch_option_chain``):
pick the expiry nearest the holding period, then read the at-the-money (ATM)
call and put for real implied volatility, the IV / HV20 ratio, the straddle's
expected move to expiry, and liquidity flags.

A chain whose ATM quotes are unusable (zero or one-sided bids before the open,
crossed quotes, yfinance's ~0 placeholder IV) or whose expiry falls outside
``MIN_DTE``..``MAX_DTE`` yields
``{"available": False, "reason": ...}`` rather than numbers, so bad data never
reaches the model. Illiquidity (including an ATM leg that has not traded
recently) is a flag on an otherwise available block; it never demotes or drops a pick.
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
# Oldest ATM ``lastTradeDate`` (calendar days before ``today``) not flagged as idle.
# Five days spans a weekend plus a market holiday (Friday trade, Monday holiday,
# Tuesday pre-market briefing = 4 days); an ATM leg idle longer than that has not
# traded for a full week of sessions. Its two-sided quote is still reported, flagged.
MAX_QUOTE_AGE_DAYS = 5
# Farthest the nearest common strike may sit from spot, as a fraction of spot. Beyond
# this the "ATM" straddle is dominated by intrinsic value and overstates the move.
MAX_ATM_DISTANCE = 0.10
# How far (as % of spot) a leg's mid may sit below its intrinsic value before the
# quotes are judged inconsistent with the reference price. 1% absorbs the lag
# between the latest close and the quote snapshot plus penny rounding; a larger
# shortfall means the quotes are stale or belong to a different underlying price.
INTRINSIC_TOLERANCE_PCT = 1.0
# Expiry window (calendar days to expiry). Below MIN_DTE the straddle is mostly gamma
# (e.g. OPEX week on a monthly-only chain); beyond MAX_DTE it no longer describes
# a ~14-day hold.
MIN_DTE = 7
MAX_DTE = 45

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

    Only expiries ``MIN_DTE``..``MAX_DTE`` days out qualify. A tie prefers the
    later expiry (more time value, less gamma).
    """
    future = [e for e in expiries if MIN_DTE <= (e - today).days <= MAX_DTE]
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


def _quote_age_days(leg: pd.Series, today: date) -> int | None:
    """Calendar days since the leg's ``lastTradeDate``; None when absent or unparseable."""
    raw = leg.get("lastTradeDate")
    if raw is None:
        return None
    try:
        traded = pd.Timestamp(raw)
    except TypeError, ValueError:
        return None
    if pd.isna(traded):
        return None
    return int((today - traded.date()).days)


def _usable_leg(leg: pd.Series) -> tuple[float, float, float, int | None] | str:
    """Return ``(mid, iv, spread % of mid, open interest or None)``, or why it is unusable."""
    bid, ask, iv = _finite(leg["bid"]), _finite(leg["ask"]), _finite(leg["impliedVolatility"])
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return "ATM quotes have no live bid/ask (stale or pre-market)"
    if ask < bid:
        return "ATM quotes are crossed"
    if iv is None or iv < MIN_USABLE_IV:
        return "ATM implied volatility is missing"
    mid = (bid + ask) / 2
    oi = _finite(leg["openInterest"])
    return mid, iv, (ask - bid) / mid * 100, None if oi is None else int(oi)


def _by_strike(legs: pd.DataFrame) -> pd.DataFrame:
    """Index legs by numeric strike, dropping junk, non-finite and non-positive strikes.

    A duplicated strike keeps the standard contract over an adjusted one (e.g.
    ``AAPL1...`` after a split), whose root carries a digit.
    """
    # float64 turns pd.NA (nullable dtypes) into NaN; NaN and inf fail both comparisons.
    strikes = pd.to_numeric(legs["strike"], errors="coerce").astype("float64")
    valid = (strikes > 0) & (strikes < math.inf)
    cleaned = legs.loc[valid].assign(strike=strikes[valid])
    if "contractSymbol" in cleaned.columns:
        # OCC symbol = root + YYMMDD + C/P + 8-digit strike; the root is all but the last 15.
        adjusted = cleaned["contractSymbol"].astype(str).str[:-15].str.contains(r"\d")
        cleaned = cleaned.loc[adjusted.sort_values(kind="stable").index]
    return cleaned.drop_duplicates("strike").set_index("strike")


def option_market_snapshot(
    chain: OptionChain, *, price: float, hv20: float | None, today: date
) -> OptionMarket:
    """Summarise the ATM straddle of ``chain`` against ``price`` and realised vol.

    Args:
        chain: One expiry's calls and puts (``strike, bid, ask,
            impliedVolatility, openInterest``, optionally ``lastTradeDate``).
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
    if not MIN_DTE <= days <= MAX_DTE:
        return _unavailable(f"expiry {days} DTE is outside the {MIN_DTE}-{MAX_DTE} DTE window")
    if any(c not in frame.columns for frame in (chain.calls, chain.puts) for c in _LEG_COLUMNS):
        return _unavailable("option chain is missing quote columns")

    calls, puts = _by_strike(chain.calls), _by_strike(chain.puts)
    strikes = calls.index.intersection(puts.index)
    if strikes.empty:
        return _unavailable("no strike listed for both calls and puts")
    strike = float(min(strikes, key=lambda k: abs(float(k) - spot)))
    if abs(strike - spot) / spot > MAX_ATM_DISTANCE:
        return _unavailable("no strike near the money")
    call, put = calls.loc[strike], puts.loc[strike]

    legs = []
    for leg in (call, put):
        usable = _usable_leg(leg)
        if isinstance(usable, str):
            return _unavailable(usable)
        legs.append(usable)
    tolerance = spot * INTRINSIC_TOLERANCE_PCT / 100
    for (mid, *_), intrinsic in zip(
        legs, (max(spot - strike, 0.0), max(strike - spot, 0.0)), strict=True
    ):
        if mid < intrinsic - tolerance:
            return _unavailable("ATM quotes are below intrinsic value (stale or mismatched)")

    atm_iv = (legs[0][1] + legs[1][1]) / 2
    straddle = legs[0][0] + legs[1][0]
    # Flag on the stored spread at its displayed 1-dp precision so they never disagree.
    spread_pct = round(max(leg[2] for leg in legs), 1)
    hv = _finite(hv20)
    # Classify on the displayed (rounded) ratio so the label never contradicts it.
    ratio = round(atm_iv / hv, 2) if hv is not None and hv > 0 else None
    if ratio is None:
        label = None
    elif ratio > IV_HV_RICH:
        label = "rich"
    elif ratio < IV_HV_CHEAP:
        label = "cheap"
    else:
        label = "fair"

    flags = []
    open_interest = [leg[3] for leg in legs]
    known_oi = [oi for oi in open_interest if oi is not None]
    if len(known_oi) < len(open_interest):
        flags.append("ATM open interest unknown")
    if known_oi and min(known_oi) < MIN_LEG_OPEN_INTEREST:
        flags.append(f"low ATM open interest (<{MIN_LEG_OPEN_INTEREST} per leg)")
    if spread_pct > MAX_SPREAD_PCT:
        flags.append(f"wide ATM bid-ask spread ({spread_pct:.1f}% of mid)")
    ages = [a for a in (_quote_age_days(leg, today) for leg in (call, put)) if a is not None]
    if ages and max(ages) > MAX_QUOTE_AGE_DAYS:
        flags.append(f"ATM not traded in {max(ages)} days")

    return {
        "available": True,
        "expiry": chain.expiry.isoformat(),
        "days_to_expiry": days,
        "atm_strike": strike,
        "atm_iv": round(atm_iv, 4),
        "iv_hv20_ratio": ratio,
        "iv_vs_hv": label,
        "expected_move_dollars": round(straddle, 2),
        "expected_move_pct": round(straddle / spot * 100, 2),
        "atm_open_interest": None if len(known_oi) < 2 else sum(known_oi),
        "atm_spread_pct": spread_pct,
        "liquid": not flags,
        "liquidity_flags": flags,
    }
