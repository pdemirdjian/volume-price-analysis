"""Sector context from the scan's own bars (PDE-248).

Two pieces, both pure and offline:

- Each sp500 member's broad sector, read from the pytickersymbols payload's
  ``industries`` list (:func:`sector_from_industries`), so every scan candidate
  carries a ``sector`` (None for ETFs and anything unmapped).
- Sector relative strength (:func:`sector_relative_strength`): each SPDR sector
  ETF's trailing 5- and 20-session return minus SPY's, ranked on the 20-session
  figure. The returns ride on the scan worker's ``BreadthObservation``, so no
  extra fetch happens; a universe without the sector ETFs simply yields no rows.

Sector context is advisory: it never changes a score, a conviction, or a filter.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .breadth import BreadthObservation
from .payloads import SectorStrength

# GICS sector -> its SPDR Select Sector ETF (all eleven are in the full_market universe).
SECTOR_ETFS: dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# Non-GICS spellings (casefolded) -> sector; used only when no case-exact GICS name
# (a ``SECTOR_ETFS`` key) appears. The payload writes GICS names in Title Case and
# free-text tags in sentence case, so "Information technology" beside Alphabet's
# "Communication Services", or "Technology" beside Visa's "Financials", is an alias.
_SECTOR_ALIASES: dict[str, str] = {
    **{sector.casefold(): sector for sector in SECTOR_ETFS},
    "healthcare": "Health Care",
    "technology": "Information Technology",
}

RS_BENCHMARK = "SPY"


def sector_from_industries(industries: Iterable[Any] | None) -> str | None:
    """The broad sector named in a pytickersymbols ``industries`` list, or None.

    The payload mixes the GICS sector with free-text industries and sometimes
    names two sectors (e.g. ``['Energy', 'Utilities']`` for a utility).
    Case-exact GICS names win over aliases; among them the last entry wins,
    which matches GICS for the payload's multi-sector utilities and REITs. An
    alias (``"Technology"``, ``"Real estate"``) is used only when no GICS name
    appears. Known residual mislabels from the payload's own tags: Garmin,
    Moody's, and MSCI read Industrials; Steris (STE) reads Utilities.
    """
    canonical = alias = None
    for entry in industries or ():
        if isinstance(entry, str):
            name = entry.strip()
            if name in SECTOR_ETFS:
                canonical = name
            alias = _SECTOR_ALIASES.get(name.casefold(), alias)
    return canonical or alias


def sector_relative_strength(observations: Iterable[BreadthObservation]) -> list[SectorStrength]:
    """Rank the sector ETFs by 20-session return minus SPY's, strongest first.

    Reads the SPY and sector-ETF observations (anything else is ignored; the
    first observation per symbol wins). Empty when SPY is missing or lacks a
    20-session return. A sector ETF is skipped when it lacks a 20-session return
    or its last session differs from SPY's. ``rank`` is 1 for the strongest.
    """
    by_symbol: dict[str, BreadthObservation] = {}
    for o in observations:
        by_symbol.setdefault(o.symbol, o)
    spy = by_symbol.get(RS_BENCHMARK)
    if spy is None or spy.return_20d is None:
        return []

    ranked: list[tuple[float, SectorStrength]] = []
    for sector, etf in SECTOR_ETFS.items():
        etf_obs = by_symbol.get(etf)
        if etf_obs is None or etf_obs.return_20d is None or etf_obs.session != spy.session:
            continue
        ret_5d, ret_20d = etf_obs.return_5d, etf_obs.return_20d
        rs_5d = (
            None if ret_5d is None or spy.return_5d is None else round(ret_5d - spy.return_5d, 2)
        )
        rs_20d = ret_20d - spy.return_20d
        ranked.append(
            (
                rs_20d,
                {
                    "sector": sector,
                    "etf": etf,
                    "return_5d": None if ret_5d is None else round(ret_5d, 2),
                    "return_20d": round(ret_20d, 2),
                    "rs_5d": rs_5d,
                    "rs_20d": round(rs_20d, 2),
                    "rank": 0,
                },
            )
        )
    # Rank on the unrounded RS so distinct values that round equal keep their order.
    ranked.sort(key=lambda item: (-item[0], item[1]["etf"]))
    rows = [row for _, row in ranked]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def sector_conflict(
    candidate: Mapping[str, Any], strength: Sequence[Mapping[str, Any]]
) -> str | None:
    """A note when a candidate's direction fights its sector's 20-session RS rank.

    A bullish candidate in the weakest sector or a bearish one in the strongest
    gets a note such as ``"bullish setup in the weakest sector (Energy, XLE)"``.
    ``strength`` is ``sector_relative_strength``'s output, strongest first. None
    without a sector, a score, or at least two ranked sectors.
    """
    sector = candidate.get("sector")
    score = candidate.get("composite_score")
    if not sector or len(strength) < 2:
        return None
    try:
        direction = "bullish" if float(score) >= 0 else "bearish"  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None
    against = strength[-1] if direction == "bullish" else strength[0]
    if against.get("sector") != sector:
        return None
    label = "weakest" if direction == "bullish" else "strongest"
    return f"{direction} setup in the {label} sector ({sector}, {against.get('etf')})"
