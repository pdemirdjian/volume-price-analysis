"""The ledger: an append-only record of delivered picks and how they turned out (PDE-195).

The briefing emits picks every morning; the backtest measures the composite
score on reconstructed history, but only this file records what was actually
delivered. It is JSONL at ``LEDGER_PATH``, never rewritten, holding two kinds of
``schema: 1`` row (the format is documented in ``CONTEXT.md`` under "ledger"):

- a **pick row** per pick per briefing date, appended after a successful send;
- a **grade row** per pick row per elapsed horizon, appended by the grader.

Horizons are counted in trading sessions from the briefing date: the entry bar
is the last session before it (the close the scan priced the pick at), and
horizon ``N`` exits at the close of the ``N``-th session on or after it.

Usage:
    python -m volume_price_analysis.agent.ledger grade [--dry-run]
    python -m volume_price_analysis.agent.ledger report
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..backtest import forward_returns
from ..data_fetcher import DataSource, get_default_data_source
from .config import AgentConfig
from .picks import CONVICTIONS, Pick

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
# Every pick is graded at these horizons plus its own holding period.
FIXED_HORIZONS = (5, 10)
# The briefing's scorecard covers picks whose briefing date is this recent.
SCORECARD_WINDOW_DAYS = 90

# Calendar days fetched ahead of the briefing date so the entry bar is present.
_ENTRY_LOOKBACK_DAYS = 10
# A horizon still ungraded this many weekdays after it was due is abandoned
# (delisted symbol, dirty data), so it is not refetched on every run forever.
_STALE_AFTER_SESSIONS = 20
_MAX_CONCURRENT_FETCHES = 4

Row = dict[str, Any]


def _finite(value: float | None) -> float | None:
    """The value as a float, or None when it is missing or not finite (JSON has no NaN)."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """The identity a row is deduplicated on; ``horizon`` is None for pick rows."""
    return (
        row.get("kind", "pick"),
        row.get("briefing_date"),
        row.get("symbol"),
        row.get("horizon"),
    )


def pick_rows(
    picks: Iterable[Pick], *, briefing_date: date, regime: str | None, holding_period: int
) -> list[Row]:
    """One ledger pick row per pick, in pick-table order."""
    return [
        {
            "schema": SCHEMA_VERSION,
            "kind": "pick",
            "briefing_date": briefing_date.isoformat(),
            "symbol": p.symbol,
            "direction": p.direction,
            "conviction": p.conviction,
            "score": _finite(p.score),
            "price": _finite(p.price),
            "stop": _finite(p.stop),
            "target": _finite(p.target),
            "holding_period": holding_period,
            "regime": regime,
            "regime_conflict": p.regime_conflict,
            "earnings_warning": p.earnings_warning,
        }
        for p in picks
    ]


def read_ledger(path: Path) -> list[Row]:
    """Every ``schema: 1`` row in the ledger; a missing file is an empty ledger.

    Unparseable lines (a write cut short) and rows of another schema version are
    skipped rather than raised: one bad line must not hide the rest.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    rows: list[Row] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            logger.warning("Ledger %s line %d is not valid JSON; skipped", path, number)
            continue
        if isinstance(row, dict) and row.get("schema") == SCHEMA_VERSION:
            rows.append(row)
    return rows


def append_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Append the rows the ledger does not already hold; return how many were written.

    Idempotent on (kind, briefing date, symbol, horizon), so re-running a
    briefing date or the grader writes nothing new. Existing lines are never
    touched, and nothing is created when there is nothing to write.
    """
    seen = {_key(row) for row in read_ledger(path)}
    fresh = []
    for row in rows:
        if _key(row) not in seen:
            seen.add(_key(row))
            fresh.append(row)
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    # A write cut short leaves no trailing newline; start on a line of our own.
    torn = path.exists() and path.read_bytes()[-1:] not in (b"", b"\n")
    with path.open("a", encoding="utf-8", newline="\n") as ledger:
        if torn:
            ledger.write("\n")
        for row in fresh:
            ledger.write(json.dumps(row, allow_nan=False) + "\n")
    return len(fresh)


def horizons_for(row: Mapping[str, Any]) -> list[int]:
    """The horizons a pick row is graded at: 5, 10 and its holding period."""
    horizons = set(FIXED_HORIZONS)
    holding_period = row.get("holding_period")
    if isinstance(holding_period, int) and holding_period >= 1:
        horizons.add(holding_period)
    return sorted(horizons)


def _first_touch(
    window: pd.DataFrame, *, bullish: bool, stop: float | None, target: float | None
) -> tuple[bool | None, bool | None]:
    """(stop_hit, target_hit) for the first bar in ``window`` to touch either level.

    Daily Low/High decide a touch. A bar touching both counts as the stop: the
    intraday order is unknowable from daily bars, so take the pessimistic read.
    A level that is None is reported as None, never False.
    """
    for low, high in zip(window["Low"], window["High"], strict=True):
        stopped = stop is not None and (low <= stop if bullish else high >= stop)
        reached = target is not None and (high >= target if bullish else low <= target)
        if stopped:
            return True, None if target is None else False
        if reached:
            return None if stop is None else False, True
    return None if stop is None else False, None if target is None else False


def grade_pick(
    row: Mapping[str, Any],
    data: pd.DataFrame,
    *,
    as_of: date,
    horizons: Sequence[int] | None = None,
) -> list[Row]:
    """Grade rows for each requested horizon of a pick row that has elapsed in ``data``.

    Strictly causal: bars dated on or after ``as_of`` are ignored, so a run
    during the session never grades against a partial bar. The return is
    close-to-close from the entry bar (``backtest.forward_returns``), signed so
    that positive means the pick's direction was right.
    """
    briefing_date = date.fromisoformat(row["briefing_date"])
    frame = data[pd.to_datetime(data["Date"]).dt.date < as_of].reset_index(drop=True)
    dates = pd.to_datetime(frame["Date"]).dt.date
    entry = int((dates < briefing_date).sum()) - 1
    if entry < 0:
        return []
    bullish = row.get("direction") != "bearish"
    # Fetched history is re-adjusted for splits and dividends paid since the
    # briefing, so levels recorded then are rescaled onto today's price basis.
    price = row.get("price")
    scale = float(frame["Close"].iloc[entry]) / price if price else 1.0
    stop, target = row.get("stop"), row.get("target")
    grades: list[Row] = []
    for horizon in horizons_for(row) if horizons is None else horizons:
        forward = float(forward_returns(frame, horizon).iloc[entry])
        if not math.isfinite(forward):
            continue  # not elapsed yet, or a data gap
        stop_hit, target_hit = _first_touch(
            frame.iloc[entry + 1 : entry + horizon + 1],
            bullish=bullish,
            stop=None if stop is None else stop * scale,
            target=None if target is None else target * scale,
        )
        grades.append(
            {
                "schema": SCHEMA_VERSION,
                "kind": "grade",
                "briefing_date": row["briefing_date"],
                "symbol": row["symbol"],
                "horizon": horizon,
                "exit_date": dates.iloc[entry + horizon].isoformat(),
                "signed_return": round(forward if bullish else -forward, 6),
                "stop_hit": stop_hit,
                "target_hit": target_hit,
            }
        )
    return grades


def grade_ledger(
    rows: Sequence[Mapping[str, Any]], source: DataSource, *, as_of: date
) -> list[Row]:
    """The grade rows now due for a ledger's pick rows. Writes nothing.

    One fetch per symbol with a due horizon. A symbol whose fetch or grading
    fails is logged and skipped; it is retried on the next run.
    """
    done = {_key(row) for row in rows if row.get("kind") == "grade"}
    due: dict[str, list[tuple[Mapping[str, Any], list[int]]]] = {}
    for row in rows:
        if row.get("kind", "pick") != "pick":
            continue
        try:
            # Weekdays in [briefing date, as_of): an upper bound on elapsed sessions.
            sessions = int(np.busday_count(row["briefing_date"], as_of))
            symbol = row["symbol"]
        except KeyError, TypeError, ValueError:
            continue
        horizons = [
            h
            for h in horizons_for(row)
            if h <= sessions <= h + _STALE_AFTER_SESSIONS
            and ("grade", row["briefing_date"], symbol, h) not in done
        ]
        if horizons:
            due.setdefault(symbol, []).append((row, horizons))
    if not due:
        return []

    def grade_symbol(symbol: str) -> list[Row]:
        picks = due[symbol]
        start = min(date.fromisoformat(row["briefing_date"]) for row, _ in picks)
        start -= timedelta(days=_ENTRY_LOOKBACK_DAYS)
        try:
            data = source.fetch(symbol, start=start.isoformat(), end=as_of.isoformat())
            return [
                grade
                for row, horizons in picks
                for grade in grade_pick(row, data, as_of=as_of, horizons=horizons)
            ]
        except Exception:
            logger.warning("Ledger grading failed for %s", symbol, exc_info=True)
            return []

    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_FETCHES) as pool:
        return [grade for grades in pool.map(grade_symbol, sorted(due)) for grade in grades]


_SCORECARD_HEADER = "| Picks | 5d | 10d | Holding period |"
_SCORECARD_RULE = "|---|---|---|---|"
_SCORECARD_GROUPS = (*CONVICTIONS, "aligned", "counter-regime")


def scorecard_table(
    rows: Iterable[Mapping[str, Any]],
    *,
    as_of: date | None = None,
    window_days: int = SCORECARD_WINDOW_DAYS,
) -> str:
    """Hit rate and mean signed return per horizon as a fixed markdown table.

    Rows are the conviction labels, then picks aligned with the regime against
    counter-regime ones. A hit is a positive signed return. ``as_of`` limits the
    table to picks whose briefing date falls in the trailing ``window_days``;
    None covers the whole ledger. Returns ``""`` when nothing is graded.
    """
    rows = list(rows)
    picks = {_key(row)[1:3]: row for row in rows if row.get("kind", "pick") == "pick"}
    cutoff = None if as_of is None else (as_of - timedelta(days=window_days)).isoformat()
    cells: dict[tuple[str, int], list[float]] = {}
    for grade in rows:
        pick = picks.get(_key(grade)[1:3])
        if grade.get("kind") != "grade" or pick is None:
            continue
        if cutoff is not None and str(grade.get("briefing_date")) <= cutoff:
            continue
        signed, horizon = grade.get("signed_return"), grade.get("horizon")
        if not isinstance(signed, (int, float)):
            continue
        columns = [i for i, fixed in enumerate(FIXED_HORIZONS) if horizon == fixed]
        if horizon == pick.get("holding_period"):
            columns.append(len(FIXED_HORIZONS))
        groups = (
            pick.get("conviction"),
            "counter-regime" if pick.get("regime_conflict") else "aligned",
        )
        for group in groups:
            for column in columns:
                cells.setdefault((str(group), column), []).append(float(signed))
    if not cells:
        return ""
    lines = [_SCORECARD_HEADER, _SCORECARD_RULE]
    for group in _SCORECARD_GROUPS:
        row_cells = []
        for column in range(len(FIXED_HORIZONS) + 1):
            returns = cells.get((group, column))
            if not returns:
                row_cells.append("—")
                continue
            hit_rate = sum(r > 0 for r in returns) / len(returns)
            mean = sum(returns) / len(returns)
            row_cells.append(f"{hit_rate:.0%} / {mean:+.2%} (n={len(returns)})")
        lines.append(f"| {group} | {' | '.join(row_cells)} |")
    return "\n".join(lines)


def _report(rows: Sequence[Mapping[str, Any]]) -> str:
    """The full dump: the all-time scorecard, then every pick row with its grades."""
    grades: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("kind") == "grade":
            grades.setdefault(_key(row)[1:3], []).append(row)
    lines = [scorecard_table(rows) or "No graded picks yet.", ""]
    for row in rows:
        if row.get("kind", "pick") != "pick":
            continue
        outcomes = []
        for grade in sorted(grades.get(_key(row)[1:3], []), key=lambda g: g["horizon"]):
            touched = (
                "stop" if grade.get("stop_hit") else "target" if grade.get("target_hit") else ""
            )
            outcomes.append(f"{grade['horizon']}d {grade['signed_return']:+.2%} {touched}".rstrip())
        lines.append(
            f"{row.get('briefing_date')} {row.get('symbol')} {row.get('direction')} "
            f"{row.get('conviction')}: {', '.join(outcomes) or 'ungraded'}"
        )
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    data_source: DataSource | None = None,
    today: date | None = None,
) -> int:
    """CLI: ``grade`` appends the grade rows now due; ``report`` dumps the ledger."""
    parser = argparse.ArgumentParser(description="Pick ledger: grade delivered picks, or report")
    commands = parser.add_subparsers(dest="command", required=True)
    grade = commands.add_parser("grade", help="Grade pick rows whose horizon has elapsed")
    grade.add_argument("--dry-run", action="store_true", help="Print grades without writing them")
    commands.add_parser("report", help="Print the all-time scorecard and every pick's grades")
    args = parser.parse_args(argv)

    ledger_path = AgentConfig.from_env().ledger_path
    if not ledger_path:
        print("LEDGER_PATH is not set", file=sys.stderr)
        return 2
    path = Path(ledger_path)
    rows = read_ledger(path)
    if args.command == "report":
        print(_report(rows))
        return 0
    if today is None:
        today = datetime.now(ZoneInfo("America/New_York")).date()
    source = data_source if data_source is not None else get_default_data_source()
    new_grades = grade_ledger(rows, source, as_of=today)
    for row in new_grades:
        print(json.dumps(row))
    if args.dry_run:
        print(f"Dry run: {len(new_grades)} grade row(s) not written")
    else:
        print(f"Appended {append_rows(path, new_grades)} grade row(s) to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
