"""Causality sweep: no indicator or score may change when future bars are appended.

The evidence harness (``backtest.causal_score_at``) and the scan's ``as_of`` cutoff are only
honest if every value reported for bar ``t`` depends on bars ``<= t`` alone. This module sweeps
every public function in ``indicators.py`` plus ``score_symbol`` and ``causal_score_at`` over
seeded synthetic OHLCV frames and fails CI when one reads the future.

Every public function in ``indicators.py`` must appear in exactly one of three tables, or
``test_every_public_indicator_is_classified`` fails (and a stale name fails too):

``SERIES``
    Functions that return per-bar output (a Series, a tuple of Series, or a dict carrying
    Series). The check: compute on ``df[:n]`` and on ``df[:n + k]`` and require every per-bar
    output to agree at position ``n - 1``. Both frames start at the same bar, so Wilder/EWM
    recursions seeded from the first bar match too. ``ANCHORS`` additionally ties a dict's
    last-bar scalar (e.g. ``adx``) to the as-of-``n - 1`` value of its own per-bar series on the
    longer frame, so a snapshot read from the wrong bar is caught as well.

``SNAPSHOT``
    Functions that report only the last bar of the frame they are handed. They cannot read bars
    they are not given, so per-bar use is causal exactly when the caller slices first, which is
    what ``causal_score_at`` and the scan's ``as_of`` cutoff do; those two get their own
    as-of tests below. The sweep still asserts a snapshot exposes no per-bar Series (anything
    that could be indexed at a past bar belongs in ``SERIES``) and that its snapshot of
    ``df[:n]`` equals the as-of-``n - 1`` slice of the longer frame.

``NON_CAUSAL``
    The allow-list. Each entry needs a non-blank reason. A real leak found by the sweep goes
    here with a reason starting ``KNOWN LEAK:`` until it is fixed.

Decisions on the issue's open questions:

* Whole-window outputs: the two volume-profile functions bin volume over the full price range
  of the window and have no per-bar meaning, so they are allow-listed. ``find_pivots`` is
  allow-listed because marking bar ``i`` needs bars ``i + 1 .. i + right`` (the confirmation
  window); its consumer ``rsi_divergence_signal_series`` gates on confirmation and is swept.
  The HV percentile (``calculate_iv_percentile``) uses a *trailing* 252-bar window ending at
  the last bar, so it is a causal snapshot, not whole-window.
* Tolerance: ``np.isclose`` with ``rtol=1e-9``/``atol=1e-12`` and NaN == NaN at the same
  position. Same-start prefixes should match exactly; the slack only absorbs platform float
  differences, and is far below any real leak.
* Data: synthetic only. ``tests/golden`` holds no OHLCV fixture frames.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from volume_price_analysis import indicators
from volume_price_analysis.analysis import analyze_single_symbol, score_symbol
from volume_price_analysis.backtest import causal_score_at
from volume_price_analysis.data_fetcher import InMemoryDataSource

# Prefix lengths and future-bar counts. n covers warm-up edges for 7-20 bar windows; the largest
# k crosses the 252-bar HV percentile lookback. Seeds give independent random paths.
PREFIX_LENGTHS = (45, 97, 160)
EXTRA_BARS = (1, 7, 40, 260)
SEEDS = (7, 2024)

RTOL = 1e-9
ATOL = 1e-12


def _synthetic_ohlcv(seed: int, length: int) -> pd.DataFrame:
    """Seeded random-walk OHLCV with High >= max(Open, Close), Low <= min(Open, Close)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, length)))
    open_ = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0.0, 0.005, length))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.02, length))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.02, length))
    volume = rng.integers(200_000, 5_000_000, length)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-02", periods=length),
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


_FRAMES = {seed: _synthetic_ohlcv(seed, max(PREFIX_LENGTHS) + max(EXTRA_BARS)) for seed in SEEDS}

CASES = [(seed, n, k) for seed in SEEDS for n in PREFIX_LENGTHS for k in EXTRA_BARS]


def _frames(seed: int, n: int, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(df[:n], df[:n + k])`` as independent copies of the seeded frame."""
    full = _FRAMES[seed]
    return full.iloc[:n].copy(), full.iloc[: n + k].copy()


# --------------------------------------------------------------------------------------------
# Registries
# --------------------------------------------------------------------------------------------

Fn = Callable[[pd.DataFrame], Any]


def _rsi_divergence_series(df: pd.DataFrame) -> pd.Series:
    return indicators.rsi_divergence_signal_series(df, indicators.calculate_rsi(df, 7), 10)


#: Per-bar functions, called on an OHLCV frame. Keys are ``indicators`` function names.
SERIES: dict[str, Fn] = {
    "calculate_obv": indicators.calculate_obv,
    "calculate_vwap": indicators.calculate_vwap,
    "calculate_vpt": indicators.calculate_vpt,
    "calculate_mfi": lambda df: indicators.calculate_mfi(df, 7),
    "calculate_historical_volatility": indicators.calculate_historical_volatility,
    "calculate_atr": indicators.calculate_atr,
    "calculate_bollinger_bands": indicators.calculate_bollinger_bands,
    "calculate_accumulation_distribution": indicators.calculate_accumulation_distribution,
    "calculate_chaikin_money_flow": indicators.calculate_chaikin_money_flow,
    "calculate_relative_volume": indicators.calculate_relative_volume,
    "calculate_vwma": indicators.calculate_vwma,
    "calculate_price_roc": indicators.calculate_price_roc,
    "calculate_adx": lambda df: indicators.calculate_adx(df, 10),
    "calculate_rsi": lambda df: indicators.calculate_rsi(df, 7),
    "rsi_divergence_signal_series": _rsi_divergence_series,
    "calculate_rsi_with_divergence": lambda df: indicators.calculate_rsi_with_divergence(df, 7),
}

#: Last-bar scalar -> the per-bar series it must equal at ``n - 1`` on the longer frame.
ANCHORS: dict[str, dict[str, str]] = {
    "calculate_relative_volume": {"current_rvol": "rvol_series"},
    "calculate_price_roc": {"current_roc": "roc_series"},
    "calculate_adx": {"adx": "adx_series", "plus_di": "plus_di_series"},
    "calculate_rsi_with_divergence": {"rsi": "rsi_series"},
}

#: Last-bar-only functions, called on an OHLCV frame.
SNAPSHOT: dict[str, Fn] = {
    "trend_verdict": lambda df: indicators.trend_verdict(df["Close"]),
    "analyze_volume_trends": indicators.analyze_volume_trends,
    "detect_bollinger_squeeze": indicators.detect_bollinger_squeeze,
    "detect_volume_breakout": indicators.detect_volume_breakout,
    "detect_rsi_divergence": lambda df: indicators.detect_rsi_divergence(
        df, indicators.calculate_rsi(df, 7), 10
    ),
    "calculate_rsi_divergence": indicators.calculate_rsi_divergence,
    "calculate_iv_percentile": indicators.calculate_iv_percentile,
    "calculate_expected_move": indicators.calculate_expected_move,
    "calculate_composite_score": indicators.calculate_composite_score,
}

#: Allow-list: function name -> why it is exempt from the sweep. A reason is mandatory.
NON_CAUSAL: dict[str, str] = {
    "calculate_volume_profile": (
        "Whole-window summary by design: bins span the window's full Low..High range, so the "
        "output has no per-bar meaning. Callers must slice to the as-of bar first."
    ),
    "calculate_enhanced_volume_profile": (
        "Whole-window summary by design: POC/VAH/VAL come from calculate_volume_profile over "
        "the whole window. Callers must slice to the as-of bar first."
    ),
    "find_pivots": (
        "Marks bar i using bars i+1..i+right (the confirmation window) by design. Consumers "
        "must gate pivots on confirmation; rsi_divergence_signal_series does and is swept."
    ),
    "threshold_verdict": "Maps a scalar to a label; takes no bar data.",
    "composite_adx_period": "Maps a holding period to an ADX lookback; takes no bar data.",
}


# --------------------------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------------------------


def _per_bar_outputs(result: Any) -> dict[str, pd.Series]:
    """Collect every per-bar Series in a result, keyed by its path."""
    if isinstance(result, pd.Series):
        return {"": result}
    if isinstance(result, tuple):
        return {f"[{i}]": s for i, s in enumerate(result) if isinstance(s, pd.Series)}
    if isinstance(result, dict):
        return {key: value for key, value in result.items() if isinstance(value, pd.Series)}
    return {}


def _same(a: Any, b: Any) -> bool:
    """Equality within tolerance, with NaN == NaN."""
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return bool(np.isclose(float(a), float(b), rtol=RTOL, atol=ATOL, equal_nan=True))
    return bool(a == b)


def _snapshot_diff(a: Any, b: Any, path: str = "") -> list[str]:
    """Paths where two (possibly nested) snapshot results differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys {sorted(a)} != {sorted(b)}"]
        return [d for key in a for d in _snapshot_diff(a[key], b[key], f"{path}.{key}")]
    return [] if _same(a, b) else [f"{path}: {a!r} != {b!r}"]


def series_lookahead(fn: Fn, prefix: pd.DataFrame, extended: pd.DataFrame) -> list[str]:
    """Per-bar outputs of ``fn`` that differ at the last prefix bar once future bars exist.

    Returns human-readable mismatches; empty means causal for this prefix/extension pair.
    """
    t = len(prefix) - 1
    short = _per_bar_outputs(fn(prefix))
    long = _per_bar_outputs(fn(extended))
    assert short, "function returned no per-bar Series; register it in SNAPSHOT instead"
    assert short.keys() == long.keys()
    return [
        f"{key or '<series>'}[{t}]: {short[key].iloc[t]!r} -> {long[key].iloc[t]!r}"
        for key in short
        if not _same(short[key].iloc[t], long[key].iloc[t])
    ]


# --------------------------------------------------------------------------------------------
# Registry hygiene
# --------------------------------------------------------------------------------------------


def _public_indicator_functions() -> set[str]:
    return {
        name
        for name, obj in inspect.getmembers(indicators, inspect.isfunction)
        if obj.__module__ == indicators.__name__ and not name.startswith("_")
    }


def test_every_public_indicator_is_classified() -> None:
    classified = SERIES.keys() | SNAPSHOT.keys() | NON_CAUSAL.keys()
    missing = _public_indicator_functions() - classified
    assert not missing, (
        f"Unclassified public indicators {sorted(missing)}: add each to SERIES or SNAPSHOT in "
        "tests/test_causality.py, or to NON_CAUSAL with a reason."
    )


def test_registries_have_no_stale_entries() -> None:
    stale = (SERIES.keys() | SNAPSHOT.keys() | NON_CAUSAL.keys()) - _public_indicator_functions()
    assert not stale, f"Registry entries name functions that no longer exist: {sorted(stale)}"


def test_registries_do_not_overlap() -> None:
    tables = {"SERIES": SERIES.keys(), "SNAPSHOT": SNAPSHOT.keys(), "NON_CAUSAL": NON_CAUSAL}
    names = [name for keys in tables.values() for name in keys]
    assert len(names) == len(set(names)), "a function is registered in more than one table"


def test_anchors_reference_series_entries() -> None:
    assert ANCHORS.keys() <= SERIES.keys()


@pytest.mark.parametrize("name", sorted(NON_CAUSAL))
def test_non_causal_entry_has_reason(name: str) -> None:
    assert _reason_ok(NON_CAUSAL[name]), f"NON_CAUSAL[{name!r}] needs a non-blank reason"


def _reason_ok(reason: object) -> bool:
    return isinstance(reason, str) and bool(reason.strip())


@pytest.mark.parametrize("reason", ["", "   ", "\n\t", None])
def test_blank_reason_is_rejected(reason: object) -> None:
    assert not _reason_ok(reason)


# --------------------------------------------------------------------------------------------
# Self-checks: the sweep must catch lookahead
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leaky",
    [
        pytest.param(lambda df: df["Close"].rolling(5, center=True).mean(), id="centred-window"),
        pytest.param(lambda df: df["Close"].shift(-1), id="shift-minus-one"),
        pytest.param(
            lambda df: (df["Close"] - df["Close"].mean()) / df["Close"].std(),
            id="full-series-zscore",
        ),
        pytest.param(lambda df: df["Volume"].rank(pct=True), id="full-series-percentile"),
    ],
)
def test_sweep_detects_lookahead(leaky: Fn) -> None:
    prefix, extended = _frames(SEEDS[0], 97, 7)
    assert series_lookahead(leaky, prefix, extended)


def test_sweep_detects_find_pivots_confirmation_window() -> None:
    # find_pivots is allow-listed because it reads bars after the pivot. Prove the sweep sees
    # that on real data: some prefix end becomes a pivot once its right window prints.
    def pivots(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return indicators.find_pivots(df["High"], 3, 3)

    hits = [n for n in range(30, 160) if series_lookahead(pivots, *_frames(SEEDS[0], n, 3))]
    assert hits


def test_snapshot_guard_rejects_per_bar_output() -> None:
    prefix, _ = _frames(SEEDS[0], 97, 1)
    assert _per_bar_outputs(indicators.calculate_adx(prefix))


# --------------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SERIES))
def test_series_indicator_is_causal(name: str) -> None:
    fn = SERIES[name]
    failures = []
    for seed, n, k in CASES:
        prefix, extended = _frames(seed, n, k)
        failures += [f"seed={seed} n={n} k={k} {m}" for m in series_lookahead(fn, prefix, extended)]
    assert not failures, f"{name} reads future bars:\n" + "\n".join(failures[:10])


@pytest.mark.parametrize("name", sorted(ANCHORS))
def test_snapshot_scalar_is_as_of_value_of_its_series(name: str) -> None:
    fn = SERIES[name]
    failures = []
    for seed, n, k in CASES:
        prefix, extended = _frames(seed, n, k)
        snapshot, long = fn(prefix), fn(extended)
        for scalar, series in ANCHORS[name].items():
            if not _same(snapshot[scalar], long[series].iloc[n - 1]):
                failures.append(
                    f"seed={seed} n={n} k={k} {scalar}={snapshot[scalar]!r} "
                    f"!= {series}[{n - 1}]={long[series].iloc[n - 1]!r}"
                )
    assert not failures, f"{name} snapshot disagrees with its causal series:\n" + "\n".join(
        failures[:10]
    )


@pytest.mark.parametrize("name", sorted(SNAPSHOT))
def test_snapshot_indicator_is_causal(name: str) -> None:
    fn = SNAPSHOT[name]
    for seed, n, k in CASES:
        prefix, extended = _frames(seed, n, k)
        snapshot = fn(prefix)
        assert not _per_bar_outputs(snapshot), (
            f"{name} exposes per-bar Series {sorted(_per_bar_outputs(snapshot))}; "
            "move it to SERIES so the sweep checks them"
        )
        diff = _snapshot_diff(snapshot, fn(extended.iloc[:n]))
        assert not diff, f"{name} seed={seed} n={n} k={k}: {diff[:5]}"


# --------------------------------------------------------------------------------------------
# Score-level as-of views
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("holding_period", [7, 21, 30])
def test_causal_score_at_ignores_future_bars(holding_period: int) -> None:
    for seed, n, k in CASES:
        prefix, extended = _frames(seed, n, k)
        short = causal_score_at(prefix, n - 1, holding_period)
        long = causal_score_at(extended, n - 1, holding_period)
        assert not _snapshot_diff(short, long), f"seed={seed} n={n} k={k}: {short} != {long}"
        # The as-of view must also be the composite of exactly df[:n], not of the whole frame.
        composite = indicators.calculate_composite_score(prefix, holding_period)
        assert _same(long["composite_score"], composite["composite_score"])


def _score(data: pd.DataFrame) -> dict[str, Any] | None:
    return score_symbol(data, "SYN", 14, 0.0, 0.0, 100.0, "both")  # type: ignore[return-value]


def test_score_symbol_as_of_cutoff_ignores_future_bars() -> None:
    # Production scores the bars before ``as_of``; the candidate must equal score_symbol on the
    # prefix no matter how many later bars the data source returned.
    for seed, n, k in CASES:
        prefix, extended = _frames(seed, n, k)
        as_of = extended["Date"].iloc[n].date()
        source = InMemoryDataSource({"SYN": extended})
        via_cutoff = analyze_single_symbol(
            "SYN", "1y", 14, 0.0, 0.0, 100.0, "both", data_source=source, as_of=as_of
        )
        expected = _score(prefix)
        assert expected is not None
        assert not _snapshot_diff(dict(expected), dict(via_cutoff or {})), (
            f"seed={seed} n={n} k={k}"
        )
