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
    Functions that report only the last bar of the frame they are handed. Where a snapshot has a
    per-bar counterpart, ``SNAPSHOT_ANCHORS`` ties the snapshot of ``df[:n]`` to that counterpart
    computed on the *unsliced* ``df[:n + k]`` at position ``n - 1`` (e.g. the divergence verdict
    against ``rsi_divergence_signal_series``, ``current_hv`` against
    ``calculate_historical_volatility``), so a snapshot built from a centred window or a
    whole-series statistic fails. Snapshots with no per-bar counterpart (``trend_verdict``,
    ``analyze_volume_trends``, ``detect_volume_breakout``) are only guarded against exposing a
    per-bar Series (anything indexable at a past bar belongs in ``SERIES``). Their causality
    rests on callers slicing to the as-of bar first, which ``causal_score_at`` and the scan's
    ``as_of`` cutoff do and which the score-level tests below check.

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
  position (exercised by a warm-up prefix where per-bar outputs are still NaN). Same-start
  prefixes should match exactly; the slack only absorbs platform float differences, and is far
  below any real leak.
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

# Prefix lengths and future-bar counts. Every n is past the 7-20 bar warm-ups; the largest k
# appends more future bars than any indicator window. Seeds give independent random paths.
# WARMUP_PREFIX ends inside the warm-up so NaN outputs are compared; HV_PERCENTILE_PREFIX is long
# enough that the 252-bar HV percentile lookback (after the 20-bar HV warm-up) is full.
PREFIX_LENGTHS = (45, 97, 160)
EXTRA_BARS = (1, 7, 40, 260)
SEEDS = (7, 2024)
WARMUP_PREFIX = 12
HV_PERCENTILE_PREFIX = 300

RTOL = 1e-9
ATOL = 1e-12


def _synthetic_ohlcv(seed: int, length: int, volatile_head: int = 0) -> pd.DataFrame:
    """Seeded random-walk OHLCV with High >= max(Open, Close), Low <= min(Open, Close).

    ``volatile_head`` bars at the start get 4x the return volatility, so an HV extreme sits
    outside a trailing lookback that ends far enough later.
    """
    rng = np.random.default_rng(seed)
    sigma = np.where(np.arange(length) < volatile_head, 0.08, 0.02)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma)))
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
    "calculate_adx": {
        "adx": "adx_series",
        "plus_di": "plus_di_series",
        "minus_di": "minus_di_series",
    },
    "calculate_rsi_with_divergence": {"rsi": "rsi_series", "current_rsi": "rsi_series"},
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

# --------------------------------------------------------------------------------------------
# Snapshot anchors: last-bar fields -> their per-bar counterpart on the unsliced longer frame
# --------------------------------------------------------------------------------------------

#: Given the unsliced longer frame and ``n``, the values a snapshot of ``df[:n]`` must report.
#: Keys are dotted paths into the snapshot (``""`` is the whole snapshot).
Expected = Callable[[pd.DataFrame, int], dict[str, Any]]


def _or(value: Any, default: float) -> float:
    return default if pd.isna(value) else float(value)


def _hv_at(extended: pd.DataFrame, n: int) -> float:
    return float(indicators.calculate_historical_volatility(extended, 20).iloc[n - 1])


def _divergence_at(extended: pd.DataFrame, n: int, rsi_period: int) -> dict[str, Any]:
    """Divergence fields as of bar ``n - 1``, read off the per-bar causal signal series."""
    rsi = indicators.calculate_rsi(extended, rsi_period)
    signal = indicators.rsi_divergence_signal_series(extended, rsi, 10).iloc[n - 1]
    current_rsi = _or(rsi.iloc[n - 1], 50.0)
    return {
        "bullish_divergence": signal == 1,
        "bearish_divergence": signal == -1,
        "divergence_type": {1: "bullish", -1: "bearish"}.get(int(signal), "none"),
        "current_rsi": current_rsi,
    }


def _rsi_divergence_tool_at(extended: pd.DataFrame, n: int) -> dict[str, Any]:
    expected = _divergence_at(extended, n, 14)
    return {**expected, "rsi": expected["current_rsi"]}


def _squeeze_at(extended: pd.DataFrame, n: int) -> dict[str, Any]:
    """Squeeze as documented: bandwidth[n-1] below 0.7x the mean of the trailing 20 valid values."""
    window = indicators.SQUEEZE_WINDOW
    bandwidth = indicators.calculate_bollinger_bands(extended, window)["bandwidth"].iloc[:n]
    latest, valid = bandwidth.iloc[-1], bandwidth.dropna()
    if pd.isna(latest) or len(valid) < indicators.MIN_SQUEEZE_DETECTION_PERIODS:
        return {"": False}
    baseline = valid.iloc[-window:].mean()
    return {"": bool(latest < baseline * indicators.SQUEEZE_THRESHOLD)}


def _iv_percentile_at(extended: pd.DataFrame, n: int) -> dict[str, Any]:
    """HV percentile of bar ``n - 1`` within the trailing 252 valid HV values ending there."""
    hv = indicators.calculate_historical_volatility(extended, 20).iloc[:n]
    current = hv.iloc[-1]
    valid = hv.dropna()
    if len(valid) < 20:
        return {"iv_percentile": 50.0, "current_hv": _or(current, 0.0)}
    trailing = valid.iloc[-252:]
    low, high = float(trailing.min()), float(trailing.max())
    pct = (current - low) / (high - low) * 100 if high > low else 50.0
    return {
        "iv_percentile": _or(pct, 50.0),
        "current_hv": _or(current, 0.0),
        "hv_min": low,
        "hv_max": high,
        "lookback_days": len(trailing),
    }


def _expected_move_at(extended: pd.DataFrame, n: int) -> dict[str, Any]:
    return {
        "current_price": float(extended["Close"].iloc[n - 1]),
        "historical_volatility": _or(_hv_at(extended, n), 0.20),
    }


def _composite_at(extended: pd.DataFrame, n: int) -> dict[str, Any]:
    # calculate_composite_score's default holding_period=14 uses MFI(7), CMF/VWMA(10), RSI(7),
    # ADX(10) and divergence lookback 10.
    t = n - 1
    close = extended["Close"].iloc[t]
    adx = indicators.calculate_adx(extended, 10)["adx_series"].iloc[t]
    return {
        "indicator_summary.price_above_vwap": close > indicators.calculate_vwap(extended).iloc[t],
        "indicator_summary.price_above_vwma": close
        > indicators.calculate_vwma(extended, 10).iloc[t],
        "indicator_summary.mfi": _or(indicators.calculate_mfi(extended, 7).iloc[t], 50.0),
        "indicator_summary.cmf": _or(
            indicators.calculate_chaikin_money_flow(extended, 10).iloc[t], 0.0
        ),
        "indicator_summary.rsi": _or(indicators.calculate_rsi(extended, 7).iloc[t], 50.0),
        "indicator_summary.rsi_divergence": _divergence_at(extended, n, 7)["divergence_type"],
        "indicator_summary.adx": _or(adx, 0.0),
    }


SNAPSHOT_ANCHORS: dict[str, Expected] = {
    "detect_rsi_divergence": lambda extended, n: _divergence_at(extended, n, 7),
    "calculate_rsi_divergence": _rsi_divergence_tool_at,
    "calculate_rsi_with_divergence": lambda extended, n: _divergence_at(extended, n, 7),
    "detect_bollinger_squeeze": _squeeze_at,
    "calculate_iv_percentile": _iv_percentile_at,
    "calculate_expected_move": _expected_move_at,
    "calculate_composite_score": _composite_at,
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


def _lookup(snapshot: Any, path: str) -> Any:
    for key in filter(None, path.split(".")):
        snapshot = snapshot[key]
    return snapshot


def snapshot_lookahead(
    fn: Fn, expected: Expected, prefix: pd.DataFrame, extended: pd.DataFrame
) -> list[str]:
    """Snapshot fields of ``fn(prefix)`` that disagree with their as-of per-bar counterpart."""
    snapshot = fn(prefix)
    return [
        f"{path or '<snapshot>'}: {_lookup(snapshot, path)!r} != {want!r}"
        for path, want in expected(extended, len(prefix)).items()
        if not _same(_lookup(snapshot, path), want)
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
    assert SNAPSHOT_ANCHORS.keys() <= SERIES.keys() | SNAPSHOT.keys()


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


@pytest.mark.parametrize(
    "leaky",
    [
        pytest.param(
            lambda df: {
                "current_price": float(df["Close"].iloc[-1]),
                "historical_volatility": float(
                    indicators.calculate_historical_volatility(df, 20).mean()
                ),
            },
            id="whole-series-mean",
        ),
        pytest.param(
            lambda df: {
                "current_price": float(df["Close"].iloc[-1]),
                "historical_volatility": float(
                    (df["Close"] / df["Close"].shift(1))
                    .apply(np.log)
                    .rolling(20, center=True, min_periods=1)
                    .std()
                    .iloc[-1]
                    * np.sqrt(252)
                ),
            },
            id="centred-window",
        ),
    ],
)
def test_snapshot_check_detects_lookahead(leaky: Fn) -> None:
    prefix, extended = _frames(SEEDS[0], 97, 7)
    assert snapshot_lookahead(leaky, _expected_move_at, prefix, extended)


def test_same_treats_aligned_nans_as_equal() -> None:
    assert _same(float("nan"), np.nan)
    assert not _same(float("nan"), 1.0)
    assert not _same(1.0, 1.0 + 1e-6)


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


def test_series_sweep_compares_warm_up_nans() -> None:
    # A prefix ending inside the warm-up: the compared bar is NaN on both frames for windowed
    # indicators, so this exercises the NaN == NaN path the main sweep never reaches.
    nan_hits = 0
    for name, fn in SERIES.items():
        for seed in SEEDS:
            prefix, extended = _frames(seed, WARMUP_PREFIX, 40)
            assert not series_lookahead(fn, prefix, extended), name
            nan_hits += sum(pd.isna(s.iloc[-1]) for s in _per_bar_outputs(fn(prefix)).values())
    assert nan_hits


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
def test_snapshot_exposes_no_per_bar_series(name: str) -> None:
    prefix, _ = _frames(SEEDS[0], 97, 1)
    per_bar = _per_bar_outputs(SNAPSHOT[name](prefix))
    assert not per_bar, f"{name} exposes per-bar Series {sorted(per_bar)}; move it to SERIES"


@pytest.mark.parametrize("name", sorted(SNAPSHOT_ANCHORS))
def test_snapshot_matches_its_per_bar_counterpart(name: str) -> None:
    fn = SERIES.get(name) or SNAPSHOT[name]
    failures = []
    for seed, n, k in CASES:
        prefix, extended = _frames(seed, n, k)
        mismatches = snapshot_lookahead(fn, SNAPSHOT_ANCHORS[name], prefix, extended)
        failures += [f"seed={seed} n={n} k={k} {m}" for m in mismatches]
    assert not failures, f"{name} snapshot disagrees with its as-of value:\n" + "\n".join(
        failures[:10]
    )


@pytest.mark.parametrize("k", [1, 60])
def test_iv_percentile_with_full_252_bar_lookback(k: int) -> None:
    # The volatile head puts the HV maximum before the trailing 252-bar window, so a percentile
    # taken over the whole frame instead of the trailing window reports a different hv_max.
    for seed in SEEDS:
        full = _synthetic_ohlcv(seed, HV_PERCENTILE_PREFIX + k, volatile_head=30)
        prefix, extended = full.iloc[:HV_PERCENTILE_PREFIX].copy(), full.copy()
        assert indicators.calculate_iv_percentile(prefix)["lookback_days"] == 252
        whole_frame_max = indicators.calculate_historical_volatility(prefix, 20).max()
        assert whole_frame_max > indicators.calculate_iv_percentile(prefix)["hv_max"]
        mismatches = snapshot_lookahead(
            indicators.calculate_iv_percentile, _iv_percentile_at, prefix, extended
        )
        assert not mismatches, f"seed={seed} k={k}: {mismatches}"


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
