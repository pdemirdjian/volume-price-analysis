"""Option-market snapshot (PDE-247): real ATM IV, IV/HV20, straddle move, liquidity."""

import math
from datetime import date

import pandas as pd
import pytest

from volume_price_analysis.option_market import (
    MAX_SPREAD_PCT,
    MIN_LEG_OPEN_INTEREST,
    OptionChain,
    option_market_snapshot,
    select_expiry,
)

TODAY = date(2026, 11, 20)
EXPIRY = date(2026, 12, 4)


def _legs(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    """Rows of (strike, bid, ask, impliedVolatility, openInterest)."""
    return pd.DataFrame(rows, columns=["strike", "bid", "ask", "impliedVolatility", "openInterest"])


def _chain(
    call_atm=(100.0, 2.9, 3.1, 0.30, 500.0),
    put_atm=(100.0, 2.4, 2.6, 0.34, 400.0),
) -> OptionChain:
    calls = _legs([(95.0, 6.0, 6.2, 0.31, 200.0), call_atm, (105.0, 1.0, 1.1, 0.29, 300.0)])
    puts = _legs([(95.0, 1.0, 1.1, 0.36, 200.0), put_atm, (105.0, 6.0, 6.3, 0.33, 300.0)])
    return OptionChain(expiry=EXPIRY, calls=calls, puts=puts)


def test_select_expiry_picks_closest_to_target_dte():
    expiries = [date(2026, 11, 27), date(2026, 12, 4), date(2026, 12, 18)]
    assert select_expiry(expiries, today=TODAY, target_dte=14) == date(2026, 12, 4)


def test_select_expiry_skips_same_day_and_past_expiries():
    expiries = [date(2026, 11, 19), TODAY, date(2026, 12, 18)]
    assert select_expiry(expiries, today=TODAY, target_dte=1) == date(2026, 12, 18)


def test_select_expiry_tie_prefers_the_later_expiry():
    expiries = [date(2026, 11, 30), date(2026, 12, 10)]
    assert select_expiry(expiries, today=TODAY, target_dte=15) == date(2026, 12, 10)


def test_select_expiry_returns_none_when_nothing_is_listed():
    assert select_expiry([], today=TODAY, target_dte=14) is None


def test_snapshot_computes_atm_iv_ratio_expected_move_and_liquidity():
    snap = option_market_snapshot(_chain(), price=100.4, hv20=0.25, today=TODAY)

    assert snap["available"] is True
    assert snap["expiry"] == "2026-12-04"
    assert snap["days_to_expiry"] == 14
    assert snap["atm_strike"] == 100.0
    assert snap["atm_iv"] == pytest.approx(0.32)
    assert snap["iv_hv20_ratio"] == pytest.approx(1.28)
    assert snap["iv_vs_hv"] == "rich"
    # Straddle mid = 3.0 + 2.5 = 5.5 → 5.5 / 100.4.
    assert snap["expected_move_dollars"] == pytest.approx(5.5)
    assert snap["expected_move_pct"] == pytest.approx(5.5 / 100.4 * 100, abs=0.01)
    assert snap["atm_open_interest"] == 900
    # Widest leg spread: put (2.6 - 2.4) / 2.5 = 8%.
    assert snap["atm_spread_pct"] == pytest.approx(8.0)
    assert snap["liquid"] is True
    assert snap["liquidity_flags"] == []


@pytest.mark.parametrize(
    ("hv20", "label"),
    [(0.40, "cheap"), (0.32, "fair"), (0.25, "rich")],
)
def test_iv_vs_hv_label(hv20, label):
    snap = option_market_snapshot(_chain(), price=100.0, hv20=hv20, today=TODAY)
    assert snap["iv_vs_hv"] == label


@pytest.mark.parametrize("hv20", [None, 0.0, float("nan")])
def test_missing_hv_leaves_the_ratio_unknown_but_keeps_the_block(hv20):
    snap = option_market_snapshot(_chain(), price=100.0, hv20=hv20, today=TODAY)
    assert snap["available"] is True
    assert snap["iv_hv20_ratio"] is None
    assert snap["iv_vs_hv"] is None


def test_low_open_interest_is_flagged_not_hidden():
    thin = (100.0, 2.9, 3.1, 0.30, MIN_LEG_OPEN_INTEREST - 1)
    snap = option_market_snapshot(_chain(call_atm=thin), price=100.0, hv20=0.25, today=TODAY)
    assert snap["available"] is True
    assert snap["liquid"] is False
    assert any("open interest" in f for f in snap["liquidity_flags"])


def test_wide_spread_is_flagged():
    wide = (100.0, 2.0, 4.0, 0.30, 500.0)  # (4 - 2) / 3 = 66% of mid
    snap = option_market_snapshot(_chain(call_atm=wide), price=100.0, hv20=0.25, today=TODAY)
    assert snap["atm_spread_pct"] > MAX_SPREAD_PCT
    assert snap["liquid"] is False
    assert any("spread" in f for f in snap["liquidity_flags"])


def test_missing_open_interest_counts_as_zero():
    no_oi = (100.0, 2.9, 3.1, 0.30, float("nan"))
    snap = option_market_snapshot(_chain(call_atm=no_oi), price=100.0, hv20=0.25, today=TODAY)
    assert snap["atm_open_interest"] == 400
    assert snap["liquid"] is False


@pytest.mark.parametrize(
    "call_atm",
    [
        (100.0, 0.0, 0.0, 0.30, 500.0),  # zero bids pre-market
        (100.0, 0.0, 3.1, 0.30, 500.0),  # one-sided quote
        (100.0, 3.2, 3.0, 0.30, 500.0),  # crossed quote
        (100.0, 2.9, 3.1, 0.00001, 500.0),  # yfinance's placeholder IV
        (100.0, 2.9, 3.1, float("nan"), 500.0),
    ],
)
def test_stale_or_unusable_quotes_mark_the_block_unavailable(call_atm):
    snap = option_market_snapshot(_chain(call_atm=call_atm), price=100.0, hv20=0.25, today=TODAY)
    assert snap["available"] is False
    assert snap["reason"]
    assert "atm_iv" not in snap


def test_no_common_strike_is_unavailable():
    chain = OptionChain(
        expiry=EXPIRY,
        calls=_legs([(100.0, 2.9, 3.1, 0.3, 500.0)]),
        puts=_legs([(110.0, 2.4, 2.6, 0.3, 500.0)]),
    )
    snap = option_market_snapshot(chain, price=100.0, hv20=0.25, today=TODAY)
    assert snap == {"available": False, "reason": "no strike listed for both calls and puts"}


def test_empty_or_malformed_chain_is_unavailable():
    empty = OptionChain(expiry=EXPIRY, calls=pd.DataFrame(), puts=pd.DataFrame())
    assert option_market_snapshot(empty, price=100.0, hv20=0.25, today=TODAY)["available"] is False


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan")])
def test_bad_price_is_unavailable(price):
    snap = option_market_snapshot(_chain(), price=price, hv20=0.25, today=TODAY)
    assert snap["available"] is False


def test_expired_chain_is_unavailable():
    snap = option_market_snapshot(_chain(), price=100.0, hv20=0.25, today=EXPIRY)
    assert snap["available"] is False


def test_numbers_are_finite_and_json_friendly():
    snap = option_market_snapshot(_chain(), price=100.0, hv20=0.25, today=TODAY)
    for key in ("atm_iv", "iv_hv20_ratio", "expected_move_pct", "atm_spread_pct"):
        assert isinstance(snap[key], float) and math.isfinite(snap[key])
    assert isinstance(snap["atm_open_interest"], int)
