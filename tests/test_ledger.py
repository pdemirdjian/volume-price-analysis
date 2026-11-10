"""Tests for the pick ledger: pick rows, the grader, the scorecard table and the CLI."""

import json
import logging
from datetime import date

import pandas as pd
import pytest

from volume_price_analysis.agent.ledger import (
    append_rows,
    grade_ledger,
    grade_pick,
    horizons_for,
    main,
    pick_rows,
    read_ledger,
    scorecard_table,
)
from volume_price_analysis.agent.picks import Pick
from volume_price_analysis.data_fetcher import InMemoryDataSource

# Monday. The entry bar is Friday 2026-08-07 (index 4 of `_frame`), so the
# 5-, 10- and 14-session horizons exit at indexes 9, 14 and 18.
BRIEFING_DATE = date(2026, 8, 10)
ENTRY, EXIT_5, EXIT_10, EXIT_14 = 4, 9, 14, 18
AS_OF = date(2026, 9, 7)  # after every bar in `_frame`


def _frame(closes=None, lows=None, highs=None) -> pd.DataFrame:
    """25 sessions from 2026-08-03, flat at 100 unless a bar is overridden by index."""
    close = [100.0] * 25
    for index, value in (closes or {}).items():
        close[index] = value
    low = [c - 1 for c in close]
    high = [c + 1 for c in close]
    for index, value in (lows or {}).items():
        low[index] = value
    for index, value in (highs or {}).items():
        high[index] = value
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-08-03", periods=25),
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": [1_000_000] * 25,
        }
    )


def _pick_row(**overrides) -> dict:
    return {
        "schema": 1,
        "kind": "pick",
        "briefing_date": BRIEFING_DATE.isoformat(),
        "symbol": "AAPL",
        "direction": "bullish",
        "conviction": "HIGH",
        "score": 5.2,
        "price": 100.0,
        "stop": 90.0,
        "target": 120.0,
        "holding_period": 14,
        "regime": "bullish",
        "regime_conflict": False,
        "earnings_warning": None,
        **overrides,
    }


def _grade_row(horizon, signed_return, **overrides) -> dict:
    return {
        "schema": 1,
        "kind": "grade",
        "briefing_date": BRIEFING_DATE.isoformat(),
        "symbol": "AAPL",
        "horizon": horizon,
        "signed_return": signed_return,
        "stop_hit": False,
        "target_hit": False,
        **overrides,
    }


def _by_horizon(grades) -> dict:
    return {g["horizon"]: g for g in grades}


class TestPickRows:
    def test_one_row_per_pick_with_the_schema_fields(self):
        picks = [
            Pick("AAPL", "bullish", "HIGH", 5.2, price=190.1, stop=184.1, target=202.1),
            Pick(
                "TSLA",
                "bearish",
                "MEDIUM",
                -4.1,
                regime_conflict=True,
                earnings_warning="EARNINGS in 5 day(s)",
            ),
        ]

        rows = pick_rows(picks, briefing_date=date(2026, 9, 4), regime="bullish", holding_period=14)

        assert rows[0] == {
            "schema": 1,
            "kind": "pick",
            "briefing_date": "2026-09-04",
            "symbol": "AAPL",
            "direction": "bullish",
            "conviction": "HIGH",
            "score": 5.2,
            "price": 190.1,
            "stop": 184.1,
            "target": 202.1,
            "holding_period": 14,
            "regime": "bullish",
            "regime_conflict": False,
            "earnings_warning": None,
        }
        assert (rows[1]["price"], rows[1]["stop"], rows[1]["target"]) == (None, None, None)
        assert rows[1]["regime_conflict"] is True
        assert rows[1]["earnings_warning"] == "EARNINGS in 5 day(s)"

    def test_non_finite_numbers_become_null(self):
        rows = pick_rows(
            [Pick("AAPL", "bullish", "LOW", 1.0, price=float("nan"))],
            briefing_date=date(2026, 9, 4),
            regime=None,
            holding_period=14,
        )
        assert rows[0]["price"] is None
        json.dumps(rows[0], allow_nan=False)


class TestAppendAndRead:
    def test_round_trips_one_json_line_per_row(self, tmp_path):
        path = tmp_path / "nested" / "picks.jsonl"
        rows = [_pick_row(), _pick_row(symbol="MSFT")]

        assert append_rows(path, rows) == 2

        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == rows
        assert read_ledger(path) == rows

    def test_same_briefing_date_and_symbol_is_written_once(self, tmp_path):
        path = tmp_path / "picks.jsonl"
        append_rows(path, [_pick_row()])
        before = path.read_bytes()

        assert append_rows(path, [_pick_row(conviction="LOW"), _pick_row()]) == 0
        assert path.read_bytes() == before

    def test_grades_are_deduplicated_per_horizon(self, tmp_path):
        path = tmp_path / "picks.jsonl"
        append_rows(path, [_pick_row(), _grade_row(5, 0.01)])

        assert append_rows(path, [_grade_row(5, 0.01), _grade_row(10, 0.02)]) == 1
        assert [r.get("horizon") for r in read_ledger(path)] == [None, 5, 10]

    def test_nothing_to_write_creates_no_file(self, tmp_path):
        path = tmp_path / "picks.jsonl"
        assert append_rows(path, []) == 0
        assert not path.exists()

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert read_ledger(tmp_path / "absent.jsonl") == []

    def test_skips_torn_lines_and_other_schema_versions(self, tmp_path, caplog):
        path = tmp_path / "picks.jsonl"
        good = json.dumps(_pick_row())
        path.write_text(
            f'{good}\n\n{json.dumps({**_pick_row(), "schema": 2})}\n[1]\n{{"schema": 1, "sym',
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            assert read_ledger(path) == [_pick_row()]
        assert "line 5 is not valid JSON" in caplog.text

    def test_append_after_a_torn_line_starts_a_new_line(self, tmp_path):
        path = tmp_path / "picks.jsonl"
        path.write_text('{"schema": 1, "sym', encoding="utf-8")

        append_rows(path, [_pick_row()])

        assert read_ledger(path) == [_pick_row()]


class TestHorizons:
    def test_fixed_horizons_plus_the_holding_period(self):
        assert horizons_for({"holding_period": 14}) == [5, 10, 14]

    def test_holding_period_equal_to_a_fixed_horizon_is_not_repeated(self):
        assert horizons_for({"holding_period": 10}) == [5, 10]

    def test_missing_holding_period_keeps_the_fixed_horizons(self):
        assert horizons_for({}) == [5, 10]


class TestGradeReturns:
    FRAME_KWARGS = {"closes": {EXIT_5: 105.0, EXIT_10: 110.0, EXIT_14: 90.0}}

    def test_bullish_returns_at_5_10_and_holding_period(self):
        grades = _by_horizon(grade_pick(_pick_row(), _frame(**self.FRAME_KWARGS), as_of=AS_OF))

        assert {h: g["signed_return"] for h, g in grades.items()} == {
            5: pytest.approx(0.05),
            10: pytest.approx(0.10),
            14: pytest.approx(-0.10),
        }
        assert grades[5]["exit_date"] == "2026-08-14"
        assert grades[14] == {
            "schema": 1,
            "kind": "grade",
            "briefing_date": "2026-08-10",
            "symbol": "AAPL",
            "horizon": 14,
            "exit_date": "2026-08-27",
            "signed_return": pytest.approx(-0.10),
            # The exit bar closes at 90 with a low of 89: through the 90 stop.
            "stop_hit": True,
            "target_hit": False,
        }

    def test_bearish_returns_are_direction_signed(self):
        row = _pick_row(direction="bearish", stop=None, target=None)
        grades = _by_horizon(grade_pick(row, _frame(**self.FRAME_KWARGS), as_of=AS_OF))

        assert grades[5]["signed_return"] == pytest.approx(-0.05)
        assert grades[14]["signed_return"] == pytest.approx(0.10)

    def test_holding_period_other_than_14_is_graded(self):
        grades = _by_horizon(
            grade_pick(_pick_row(holding_period=7), _frame(closes={ENTRY + 7: 103.0}), as_of=AS_OF)
        )
        assert sorted(grades) == [5, 7, 10]
        assert grades[7]["signed_return"] == pytest.approx(0.03)

    def test_only_elapsed_horizons_are_graded(self):
        # Bars dated on or after as_of do not exist yet: the 5-session exit
        # bar is 2026-08-14, so a run that morning cannot grade it.
        frame = _frame(**self.FRAME_KWARGS)
        assert grade_pick(_pick_row(), frame, as_of=date(2026, 8, 14)) == []
        assert sorted(_by_horizon(grade_pick(_pick_row(), frame, as_of=date(2026, 8, 17)))) == [5]

    def test_no_entry_bar_before_the_briefing_date_grades_nothing(self):
        frame = _frame().iloc[ENTRY + 1 :].reset_index(drop=True)
        assert grade_pick(_pick_row(), frame, as_of=AS_OF) == []

    def test_requested_horizons_limit_the_grades(self):
        grades = grade_pick(_pick_row(), _frame(), as_of=AS_OF, horizons=[10])
        assert [g["horizon"] for g in grades] == [10]


class TestGradeStopTarget:
    """First touch over the horizon on daily Low/High; stop 90, target 120."""

    def _grades(self, row=None, **frame_kwargs):
        return _by_horizon(grade_pick(row or _pick_row(), _frame(**frame_kwargs), as_of=AS_OF))

    def test_neither_level_touched(self):
        grade = self._grades()[5]
        assert (grade["stop_hit"], grade["target_hit"]) == (False, False)

    def test_stop_first_touch(self):
        grade = self._grades(lows={ENTRY + 2: 89.0}, highs={ENTRY + 4: 121.0})[5]
        assert (grade["stop_hit"], grade["target_hit"]) == (True, False)

    def test_target_first_touch(self):
        grade = self._grades(highs={ENTRY + 2: 121.0}, lows={ENTRY + 4: 89.0})[5]
        assert (grade["stop_hit"], grade["target_hit"]) == (False, True)

    def test_same_bar_touching_both_counts_the_stop(self):
        grade = self._grades(lows={ENTRY + 2: 89.0}, highs={ENTRY + 2: 121.0})[5]
        assert (grade["stop_hit"], grade["target_hit"]) == (True, False)

    def test_touch_on_the_exact_level_counts(self):
        grade = self._grades(highs={ENTRY + 1: 120.0})[5]
        assert grade["target_hit"] is True

    def test_touch_after_a_horizon_only_counts_for_the_longer_ones(self):
        grades = self._grades(lows={ENTRY + 7: 89.0})
        assert grades[5]["stop_hit"] is False
        assert grades[10]["stop_hit"] is True
        assert grades[14]["stop_hit"] is True

    def test_the_entry_bar_itself_is_not_part_of_the_horizon(self):
        grade = self._grades(lows={ENTRY: 80.0})[5]
        assert grade["stop_hit"] is False

    def test_bearish_levels_are_mirrored(self):
        row = _pick_row(direction="bearish", stop=110.0, target=80.0)
        assert self._grades(row, highs={ENTRY + 2: 111.0})[5]["stop_hit"] is True
        assert self._grades(row, lows={ENTRY + 2: 79.0})[5]["target_hit"] is True

    def test_null_stop_and_target_grade_as_null(self):
        grade = self._grades(_pick_row(stop=None, target=None), lows={ENTRY + 2: 1.0})[5]
        assert (grade["stop_hit"], grade["target_hit"]) == (None, None)

    def test_levels_follow_a_split_since_the_briefing(self):
        # Recorded at 200 before a 2:1 split; refetched history has the entry
        # bar at 100, so the 180 stop is 90 on today's price basis.
        row = _pick_row(price=200.0, stop=180.0, target=240.0)
        assert self._grades(row, lows={ENTRY + 2: 91.0})[5]["stop_hit"] is False
        assert self._grades(row, lows={ENTRY + 2: 89.0})[5]["stop_hit"] is True


class TestGradeLedger:
    def test_grades_due_rows_through_the_data_source(self):
        source = InMemoryDataSource(frames={"AAPL": _frame(closes={EXIT_5: 105.0})})

        grades = grade_ledger([_pick_row()], source, as_of=AS_OF)

        assert [g["horizon"] for g in grades] == [5, 10, 14]
        assert grades[0]["signed_return"] == pytest.approx(0.05)
        assert source.fetch_calls == ["AAPL"]

    def test_already_graded_horizons_are_skipped(self):
        source = InMemoryDataSource(frames={"AAPL": _frame()})
        rows = [_pick_row(), _grade_row(5, 0.01), _grade_row(10, 0.01)]

        assert [g["horizon"] for g in grade_ledger(rows, source, as_of=AS_OF)] == [14]

    def test_rows_with_no_horizon_due_cost_no_fetch(self):
        source = InMemoryDataSource(frames={"AAPL": _frame()})
        fully_graded = [_pick_row(), *(_grade_row(h, 0.0) for h in (5, 10, 14))]

        # Four weekdays after the briefing date, fully graded, and long stale.
        assert grade_ledger([_pick_row()], source, as_of=date(2026, 8, 14)) == []
        assert grade_ledger(fully_graded, source, as_of=AS_OF) == []
        assert grade_ledger([_pick_row()], source, as_of=date(2027, 1, 4)) == []
        assert source.fetch_calls == []

    def test_one_fetch_per_symbol_and_a_failing_symbol_is_skipped(self, caplog):
        source = InMemoryDataSource(
            frames={"AAPL": _frame()}, errors={"GONE": ValueError("No data found")}
        )
        rows = [
            _pick_row(),
            _pick_row(briefing_date="2026-08-11"),
            _pick_row(symbol="GONE"),
            {"schema": 1, "kind": "pick"},
        ]

        with caplog.at_level(logging.WARNING):
            grades = grade_ledger(rows, source, as_of=AS_OF)

        assert {(g["symbol"], g["briefing_date"]) for g in grades} == {
            ("AAPL", "2026-08-10"),
            ("AAPL", "2026-08-11"),
        }
        assert sorted(source.fetch_calls) == ["AAPL", "GONE"]
        assert "Ledger grading failed for GONE" in caplog.text


class TestScorecardTable:
    ROWS = (
        _pick_row(),
        _grade_row(5, 0.04),
        _grade_row(14, -0.02),
        _pick_row(symbol="MSFT"),
        _grade_row(5, -0.01, symbol="MSFT"),
        _pick_row(symbol="TSLA", conviction="MEDIUM", regime_conflict=True, direction="bearish"),
        _grade_row(5, 0.03, symbol="TSLA"),
        _grade_row(10, 0.05, symbol="TSLA"),
    )

    def test_hit_rate_and_mean_return_by_conviction_and_regime_alignment(self):
        assert scorecard_table(self.ROWS, as_of=date(2026, 9, 4)) == (
            "| Picks | 5d | 10d | Holding period |\n"
            "|---|---|---|---|\n"
            "| HIGH | 50% / +1.50% (n=2) | — | 0% / -2.00% (n=1) |\n"
            "| MEDIUM | 100% / +3.00% (n=1) | 100% / +5.00% (n=1) | — |\n"
            "| LOW | — | — | — |\n"
            "| aligned | 50% / +1.50% (n=2) | — | 0% / -2.00% (n=1) |\n"
            "| counter-regime | 100% / +3.00% (n=1) | 100% / +5.00% (n=1) | — |"
        )

    def test_holding_period_matching_a_fixed_horizon_fills_both_columns(self):
        rows = [_pick_row(holding_period=10), _grade_row(10, 0.02)]
        assert "| HIGH | — | 100% / +2.00% (n=1) | 100% / +2.00% (n=1) |" in scorecard_table(rows)

    def test_empty_without_grades(self):
        assert scorecard_table([]) == ""
        assert scorecard_table([_pick_row()]) == ""
        assert scorecard_table([_grade_row(5, 0.01)]) == ""  # grade without its pick row

    def test_trailing_window_drops_old_briefing_dates(self):
        assert scorecard_table(self.ROWS, as_of=date(2026, 12, 1)) == ""
        assert scorecard_table(self.ROWS, as_of=date(2026, 12, 1), window_days=365) != ""
        assert scorecard_table(self.ROWS) != ""


class TestCli:
    @pytest.fixture
    def ledger(self, tmp_path, monkeypatch):
        path = tmp_path / "picks.jsonl"
        append_rows(path, [_pick_row(), _pick_row(symbol="MSFT", briefing_date="2026-09-04")])
        monkeypatch.setenv("LEDGER_PATH", str(path))
        return path

    def _source(self):
        return InMemoryDataSource(frames={"AAPL": _frame(closes={EXIT_5: 105.0})})

    def test_grade_dry_run_prints_without_writing(self, ledger, capsys):
        before = ledger.read_bytes()

        assert main(["grade", "--dry-run"], data_source=self._source(), today=AS_OF) == 0

        out = capsys.readouterr().out.splitlines()
        assert [json.loads(line)["horizon"] for line in out[:3]] == [5, 10, 14]
        assert json.loads(out[0])["signed_return"] == pytest.approx(0.05)
        assert out[3] == "Dry run: 3 grade row(s) not written"
        assert ledger.read_bytes() == before

    def test_grade_appends_and_is_idempotent(self, ledger, capsys):
        assert main(["grade"], data_source=self._source(), today=AS_OF) == 0
        assert f"Appended 3 grade row(s) to {ledger}" in capsys.readouterr().out
        assert [r["kind"] for r in read_ledger(ledger)] == ["pick", "pick", *["grade"] * 3]

        assert main(["grade"], data_source=self._source(), today=AS_OF) == 0
        assert "Appended 0 grade row(s)" in capsys.readouterr().out
        assert len(read_ledger(ledger)) == 5

    def test_report_dumps_the_scorecard_and_every_pick(self, ledger, capsys):
        append_rows(ledger, [_grade_row(5, 0.05, target_hit=True), _grade_row(10, -0.02)])

        assert main(["report"]) == 0

        out = capsys.readouterr().out
        assert "| HIGH | 100% / +5.00% (n=1) | 0% / -2.00% (n=1) | — |" in out
        assert "2026-08-10 AAPL bullish HIGH: 5d +5.00% target, 10d -2.00%" in out
        assert "2026-09-04 MSFT bullish HIGH: ungraded" in out

    def test_report_on_an_ungraded_ledger(self, ledger, capsys):
        assert main(["report"]) == 0
        assert capsys.readouterr().out.startswith("No graded picks yet.")

    def test_unset_ledger_path_is_an_error(self, capsys):
        assert main(["report"]) == 2
        assert "LEDGER_PATH is not set" in capsys.readouterr().err
