"""Strictly-causal backtest / evidence harness for scoring changes.

This module measures the *forward predictive power* of the composite score
without any lookahead bias, so that scoring changes (e.g. weight tweaks or new
gate thresholds) can be evaluated with before/after evidence rather than
intuition.

Design — the no-lookahead guarantee
-----------------------------------
The signal at bar ``t`` is computed by **slicing the history to ``data[:t+1]``**
and calling the existing :func:`calculate_composite_score`. Because future bars
are *physically absent* from that slice, no indicator can leak them no matter
what it does internally. The forward return is the *outcome* only and is never
fed back into the signal; we evaluate a bar only when its forward bar ``t+N``
exists. See ``tests/test_backtest.py`` for the future-invariance contract tests.

There is **no MCP surface** here — this is an internal evidence tool. It is
intentionally simple (slice-and-recompute) rather than vectorized, trading
performance for an airtight causality guarantee.

Realism note: the forward return is measured close-to-close (enter at the close
of bar ``t`` that produced the signal, exit at the close of ``t+N``). This is the
standard convention for measuring a signal's predictive information. yfinance
data is ~15-20 min delayed; this is an analysis tool, not a live-trading model.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .data_fetcher import fetch_stock_data
from .indicators import calculate_adx, calculate_composite_score, calculate_iv_percentile

logger = logging.getLogger(__name__)

# IV percentile window and ADX period used by the production scan
# (see analysis.analyze_single_symbol). The high-conviction gate must mirror the
# scan, so the harness uses these same fixed values rather than the scorer's
# holding-period-adaptive ADX period.
_IV_WINDOW = 20
_ADX_PERIOD = 14

# Score-band buckets matching the product's recommendation labels. The
# predicates form an *exact* partition of the score range (no gaps, no overlaps)
# and mirror calculate_composite_score's recommendation thresholds: a boundary
# value goes to the more-extreme bucket on its side (e.g. score 2.0 -> bullish,
# -2.0 -> bearish, 5.0 -> strong_bullish, -5.0 -> strong_bearish).
_SCORE_BUCKETS: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("strong_bearish", lambda s: s <= -5.0),
    ("bearish", lambda s: (s > -5.0) & (s <= -2.0)),
    ("neutral", lambda s: (s > -2.0) & (s < 2.0)),
    ("bullish", lambda s: (s >= 2.0) & (s < 5.0)),
    ("strong_bullish", lambda s: s >= 5.0),
)

# High-conviction gate used by run_scan (|score|>=4, ADX>=28, IV<=50).
_HC_MIN_ABS_SCORE = 4.0
_HC_MIN_ADX = 28.0
_HC_MAX_IV = 50.0


def forward_returns(data: pd.DataFrame, horizon: int) -> pd.Series:
    """Close-to-close forward return over ``horizon`` bars, aligned to bar ``t``.

    ``forward_returns[t] = Close[t + horizon] / Close[t] - 1``. The final
    ``horizon`` bars have no forward bar and are ``NaN`` (never leaked).

    Args:
        data: DataFrame with a ``Close`` column.
        horizon: Number of bars to look forward. Must be >= 1.

    Returns:
        Series of forward returns aligned to the input index.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    close = data["Close"].astype(float)
    # A zero entry price has no defined return; map it to NaN so it becomes a
    # dropped (not inf) observation rather than corrupting downstream stats.
    denom = close.where(close != 0)
    future_close = close.shift(-horizon)
    return future_close / denom - 1.0


def causal_score_at(
    data: pd.DataFrame,
    bar_index: int,
    holding_period: int = 14,
    vwap_window: int | None = None,
) -> dict[str, float]:
    """Compute the composite-score snapshot *as of* bar ``bar_index``.

    Only data up to and including ``bar_index`` is used — the history is sliced
    to ``data.iloc[:bar_index + 1]`` before any indicator runs, so future bars
    cannot influence the result.

    Args:
        data: Full OHLCV DataFrame.
        bar_index: Positional index of the bar to evaluate.
        holding_period: Holding period passed through to the scorer.
        vwap_window: VWAP anchoring for ``price_vs_vwap`` (``None`` = cumulative,
            int = rolling trailing-window). Threaded into the scorer so the
            harness can A/B anchoring variants under the same causal guarantee.

    Returns:
        Dict with ``composite_score``, ``adx`` and ``iv_percentile`` at ``t``.

    Raises:
        IndexError: If ``bar_index`` is out of range for ``data``.
    """
    if bar_index < 0 or bar_index >= len(data):
        raise IndexError(f"bar_index {bar_index} out of range for data of length {len(data)}")

    window = data.iloc[: bar_index + 1]
    composite = calculate_composite_score(window, holding_period, vwap_window=vwap_window)

    # ADX and IV come from the same fixed parameters the production scan gates
    # on (period-14 ADX, 20-bar IV), so the high-conviction gate is faithful.
    adx = float(calculate_adx(window, _ADX_PERIOD)["adx"])
    try:
        iv_pct = float(calculate_iv_percentile(window, _IV_WINDOW)["iv_percentile"])
    except Exception:  # pragma: no cover - defensive; IV proxy is a soft gate
        iv_pct = float("nan")

    return {
        "composite_score": float(composite["composite_score"]),
        "adx": adx,
        "iv_percentile": iv_pct,
    }


def compute_observations(
    data: pd.DataFrame,
    horizon: int,
    holding_period: int = 14,
    min_history: int = 50,
    step: int = 1,
    symbol: str | None = None,
    vwap_window: int | None = None,
) -> pd.DataFrame:
    """Build the (signal, forward-return) observation set for one symbol.

    For every evaluable bar ``t`` (``t >= min_history - 1`` and a forward bar
    ``t + horizon`` exists), records the as-of-``t`` composite score and the
    realized close-to-close forward return.

    Args:
        data: OHLCV DataFrame with ``Date`` and ``Close`` columns.
        horizon: Forward-return window in bars.
        holding_period: Holding period for the scorer's adaptive tuning.
        min_history: Minimum bars of history before the first evaluable bar
            (lets indicators warm up).
        step: Evaluate every ``step``-th bar to reduce overlap between windows.
        symbol: Optional label stored in a ``symbol`` column.
        vwap_window: VWAP anchoring threaded into the scorer (``None`` =
            cumulative, int = rolling trailing-window).

    Returns:
        DataFrame with columns ``bar``, ``date``, ``composite_score``, ``adx``,
        ``iv_percentile``, ``forward_return`` (plus ``symbol`` if provided). Each
        row's ``forward_return`` is non-NaN by construction.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    n = len(data)
    fwd = forward_returns(data, horizon).to_numpy()
    dates = data["Date"].to_numpy() if "Date" in data.columns else np.arange(n)

    first = max(min_history - 1, 0)
    last = n - 1 - horizon  # last bar with a realized forward return

    rows: list[dict[str, Any]] = []
    for t in range(first, last + 1, step):
        fwd_t = float(fwd[t])
        snap = causal_score_at(data, t, holding_period, vwap_window=vwap_window)
        # Drop dirty bars: a data gap (NaN Close) can produce a NaN forward
        # return or a NaN score; never let one leak into the evidence set.
        if np.isnan(fwd_t) or np.isnan(snap["composite_score"]):
            continue
        row: dict[str, Any] = {
            "bar": t,
            "date": dates[t],
            "composite_score": snap["composite_score"],
            "adx": snap["adx"],
            "iv_percentile": snap["iv_percentile"],
            "forward_return": fwd_t,
        }
        if symbol is not None:
            row = {"symbol": symbol, **row}
        rows.append(row)

    columns = ["bar", "date", "composite_score", "adx", "iv_percentile", "forward_return"]
    if symbol is not None:
        columns = ["symbol", *columns]
    return pd.DataFrame(rows, columns=columns)


def pool_observations(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-symbol observation frames into one pooled set."""
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _ic(scores: pd.Series, fwd: pd.Series, method: str) -> float | None:
    """Information coefficient (signal/forward-return correlation), or None.

    ``method="spearman"`` is computed as the Pearson correlation of the ranks
    (the definition of Spearman's rho) to avoid a hard scipy dependency, which
    pandas' built-in spearman path requires.
    """
    if len(scores) < 2:
        return None
    a, b = (scores.rank(), fwd.rank()) if method == "spearman" else (scores, fwd)
    # Correlation is undefined when either series is constant; short-circuit to
    # avoid a divide-by-zero RuntimeWarning from the underlying numpy call.
    if a.nunique() < 2 or b.nunique() < 2:
        return None
    value = a.corr(b, method="pearson")
    if value is None or pd.isna(value):
        return None
    return float(value)


def _directional_stats(scores: np.ndarray, fwd: np.ndarray) -> dict[str, float | int | None]:
    """Hit rate and directional return for a set of observations.

    ``hit_rate_directional`` and ``mean_directional_return`` are both measured
    over directional bars only (``score != 0``): a neutral bar takes no position,
    so including it would dilute the directional metrics. ``mean_forward_return``
    is the raw mean over all bars (a market-exposure baseline).
    """
    n = len(scores)
    if n == 0:
        return {
            "n": 0,
            "n_directional": 0,
            "hit_rate_directional": None,
            "mean_directional_return": None,
            "mean_forward_return": None,
        }
    sign = np.sign(scores)
    directional = sign != 0
    n_dir = int(directional.sum())
    hits = (sign == np.sign(fwd)) & directional
    hit_rate = float(hits.sum() / n_dir) if n_dir > 0 else None
    dir_return = float(np.mean((sign * fwd)[directional])) if n_dir > 0 else None
    return {
        "n": n,
        "n_directional": n_dir,
        "hit_rate_directional": hit_rate,
        "mean_directional_return": dir_return,
        "mean_forward_return": float(np.mean(fwd)),
    }


def evaluate_observations(
    obs: pd.DataFrame,
    gate_thresholds: Sequence[float] = (2.0, 3.0, 4.0, 5.0),
) -> dict[str, Any]:
    """Aggregate hit-rate / IC statistics from an observation set.

    Args:
        obs: Output of :func:`compute_observations` (possibly pooled).
        gate_thresholds: ``|score|`` cut-offs to evaluate as entry gates.

    Returns:
        Dict of pooled metrics: overall hit rate, directional return, Spearman
        and Pearson IC, per-score-bucket stats, per-gate-threshold stats, and the
        production high-conviction gate (|score|>=4 & ADX>=28 & IV<=50).
    """
    if obs.empty:
        return {
            "n": 0,
            "n_directional": 0,
            "hit_rate_directional": None,
            "mean_directional_return": None,
            "mean_forward_return": None,
            "spearman_ic": None,
            "pearson_ic": None,
            "by_score_bucket": [],
            "by_gate_threshold": [],
            "high_conviction_gate": _directional_stats(np.array([]), np.array([])),
        }

    scores = obs["composite_score"].astype(float)
    fwd = obs["forward_return"].astype(float)
    scores_np = scores.to_numpy()
    fwd_np = fwd.to_numpy()

    overall = _directional_stats(scores_np, fwd_np)

    # Score buckets (exact partition via per-bucket predicates)
    by_bucket: list[dict[str, Any]] = []
    for name, predicate in _SCORE_BUCKETS:
        mask = predicate(scores_np)
        stats = _directional_stats(scores_np[mask], fwd_np[mask])
        by_bucket.append({"bucket": name, **stats})

    # Gate-threshold sweep on |score|
    by_gate: list[dict[str, Any]] = []
    for thr in gate_thresholds:
        mask = np.abs(scores_np) >= thr
        stats = _directional_stats(scores_np[mask], fwd_np[mask])
        by_gate.append({"min_abs_score": float(thr), **stats})

    # Production high-conviction gate
    adx_np = obs["adx"].to_numpy() if "adx" in obs.columns else np.full(len(obs), np.nan)
    iv_np = (
        obs["iv_percentile"].to_numpy()
        if "iv_percentile" in obs.columns
        else np.full(len(obs), np.nan)
    )
    hc_mask = (
        (np.abs(scores_np) >= _HC_MIN_ABS_SCORE) & (adx_np >= _HC_MIN_ADX) & (iv_np <= _HC_MAX_IV)
    )
    high_conviction = _directional_stats(scores_np[hc_mask], fwd_np[hc_mask])

    return {
        **overall,
        "spearman_ic": _ic(scores, fwd, "spearman"),
        "pearson_ic": _ic(scores, fwd, "pearson"),
        "by_score_bucket": by_bucket,
        "by_gate_threshold": by_gate,
        "high_conviction_gate": high_conviction,
    }


def run_symbol_backtest(
    data: pd.DataFrame,
    horizon: int,
    holding_period: int = 14,
    min_history: int = 50,
    step: int = 1,
    symbol: str | None = None,
    vwap_window: int | None = None,
) -> dict[str, Any]:
    """Compute observations and evaluate them for a single symbol's data."""
    obs = compute_observations(
        data, horizon, holding_period, min_history, step, symbol=symbol, vwap_window=vwap_window
    )
    return {"observations": obs, "evaluation": evaluate_observations(obs)}


# --------------------------------------------------------------------------- #
# Reporting / CLI
# --------------------------------------------------------------------------- #


def _fmt_pct(value: float | None) -> str:
    return "   n/a" if value is None else f"{value * 100:6.2f}%"


def _fmt_ic(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:+.3f}"


def format_report(evaluation: dict[str, Any], meta: dict[str, Any]) -> str:
    """Render an evaluation dict as a readable plain-text report."""
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append("VPA EVIDENCE HARNESS — strictly-causal forward-return read")
    lines.append("=" * 68)
    lines.append(
        f"Symbols: {meta.get('symbols', '?')}   "
        f"Horizon: {meta.get('horizon', '?')}d   "
        f"Holding: {meta.get('holding_period', '?')}d   "
        f"Step: {meta.get('step', 1)}"
    )
    lines.append(f"VWAP anchor: {meta.get('vwap_anchor', 'cumulative')}")
    lines.append(
        f"Observations: {evaluation['n']}  (directional: {evaluation.get('n_directional', '?')})"
    )
    lines.append("")
    lines.append("OVERALL")
    lines.append(f"  Hit rate (directional) : {_fmt_pct(evaluation['hit_rate_directional'])}")
    lines.append(f"  Mean forward return    : {_fmt_pct(evaluation['mean_forward_return'])}")
    lines.append(f"  Mean directional return: {_fmt_pct(evaluation['mean_directional_return'])}")
    lines.append(f"  Spearman IC            : {_fmt_ic(evaluation['spearman_ic'])}")
    lines.append(f"  Pearson IC             : {_fmt_ic(evaluation['pearson_ic'])}")
    lines.append("")
    lines.append("BY SCORE BUCKET")
    lines.append(f"  {'bucket':<16}{'n':>6}{'hit':>9}{'dir_ret':>10}{'fwd_ret':>10}")
    for b in evaluation["by_score_bucket"]:
        lines.append(
            f"  {b['bucket']:<16}{b['n']:>6}"
            f"{_fmt_pct(b['hit_rate_directional']):>9}"
            f"{_fmt_pct(b['mean_directional_return']):>10}"
            f"{_fmt_pct(b['mean_forward_return']):>10}"
        )
    lines.append("")
    lines.append("BY GATE THRESHOLD (|score| >=)")
    lines.append(f"  {'gate':<16}{'n':>6}{'hit':>9}{'dir_ret':>10}{'fwd_ret':>10}")
    for g in evaluation["by_gate_threshold"]:
        lines.append(
            f"  {'|score|>=' + str(g['min_abs_score']):<16}{g['n']:>6}"
            f"{_fmt_pct(g['hit_rate_directional']):>9}"
            f"{_fmt_pct(g['mean_directional_return']):>10}"
            f"{_fmt_pct(g['mean_forward_return']):>10}"
        )
    hc = evaluation["high_conviction_gate"]
    lines.append("")
    lines.append("HIGH-CONVICTION GATE (|score|>=4 & ADX>=28 & IV<=50)")
    lines.append(
        f"  n={hc['n']}  hit={_fmt_pct(hc['hit_rate_directional'])}  "
        f"dir_ret={_fmt_pct(hc['mean_directional_return'])}  "
        f"fwd_ret={_fmt_pct(hc['mean_forward_return'])}"
    )
    lines.append("=" * 68)
    return "\n".join(lines)


def run_evidence(
    symbols: Sequence[str],
    period: str = "2y",
    horizon: int = 14,
    holding_period: int = 14,
    min_history: int = 50,
    step: int = 1,
    vwap_window: int | None = None,
) -> dict[str, Any]:
    """Fetch data for symbols, compute pooled observations, and evaluate.

    Per-symbol fetch failures are isolated and skipped so one bad symbol does
    not sink the run. ``vwap_window`` selects the VWAP anchoring fed to the
    scorer (``None`` = cumulative, int = rolling trailing-window) so cumulative
    and rolling anchors can be compared on the same pooled universe.
    """
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for sym in symbols:
        try:
            data = fetch_stock_data(sym, None, None, period)
            obs = compute_observations(
                data,
                horizon,
                holding_period,
                min_history,
                step,
                symbol=sym,
                vwap_window=vwap_window,
            )
            if obs.empty:
                errors.append(f"{sym}: insufficient history")
            else:
                frames.append(obs)
        except Exception as exc:  # isolate per-symbol failures
            errors.append(f"{sym}: {exc}")

    pooled = pool_observations(frames)
    evaluation = evaluate_observations(pooled)
    return {"observations": pooled, "evaluation": evaluation, "errors": errors}


def _parse_vwap_window(value: str) -> int | None:
    """Parse the ``--vwap-window`` flag: ``cumulative``/``none`` -> None, else int."""
    normalized = value.strip().lower()
    if normalized in ("cumulative", "none", ""):
        return None
    try:
        window = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--vwap-window must be 'cumulative' or a positive integer, got {value!r}"
        ) from exc
    if window < 1:
        raise argparse.ArgumentTypeError(f"--vwap-window must be >= 1, got {window}")
    return window


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``vpa-backtest AAPL MSFT --horizon 14 --period 2y``."""
    parser = argparse.ArgumentParser(description="VPA strictly-causal evidence harness")
    parser.add_argument("symbols", nargs="+", help="Ticker symbol(s) to evaluate")
    parser.add_argument("--period", default="2y", help="History period (default: 2y)")
    parser.add_argument("--horizon", type=int, default=14, help="Forward-return window in bars")
    parser.add_argument("--holding-period", type=int, default=14, help="Scorer holding period")
    parser.add_argument("--min-history", type=int, default=50, help="Warm-up bars before first")
    parser.add_argument("--step", type=int, default=1, help="Evaluate every Nth bar")
    parser.add_argument(
        "--vwap-window",
        type=_parse_vwap_window,
        default=None,
        help="VWAP anchor for price_vs_vwap: 'cumulative' (default) or a rolling window size",
    )
    args = parser.parse_args(argv)

    result = run_evidence(
        args.symbols,
        period=args.period,
        horizon=args.horizon,
        holding_period=args.holding_period,
        min_history=args.min_history,
        step=args.step,
        vwap_window=args.vwap_window,
    )
    anchor = "cumulative" if args.vwap_window is None else f"rolling-{args.vwap_window}"
    print(
        format_report(
            result["evaluation"],
            meta={
                "symbols": ", ".join(args.symbols),
                "horizon": args.horizon,
                "holding_period": args.holding_period,
                "step": args.step,
                "vwap_anchor": anchor,
            },
        )
    )
    if result["errors"]:
        print("\nErrors:")
        for err in result["errors"]:
            print(f"  - {err}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
