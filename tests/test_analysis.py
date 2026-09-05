"""Tests for analysis.py - extracted scan and options analysis logic."""

import pandas as pd
import pytest

from volume_price_analysis.analysis import (
    UNIVERSES,
    InsufficientDataError,
    _adaptive_periods,
    _build_sp500_symbols,
    _cached_universes,
    analyze_single_symbol,
    build_headline,
    get_universes,
    run_options_analysis,
    run_scan,
    score_symbol,
)
from volume_price_analysis.data_fetcher import InMemoryDataSource
from volume_price_analysis.indicators import (
    SQUEEZE_WINDOW,
    calculate_adx,
    calculate_composite_score,
    detect_bollinger_squeeze,
)

# A frame with too little history to analyze (< MIN_SCAN_HISTORY), so scans skip it fast.
SMALL_FRAME = pd.DataFrame(
    {
        "Date": pd.date_range("2024-01-01", periods=10),
        "Open": [100] * 10,
        "High": [101] * 10,
        "Low": [99] * 10,
        "Close": [100] * 10,
        "Volume": [1000000] * 10,
    }
)


def source_for(symbols, frame, errors=None):
    """An InMemoryDataSource serving ``frame`` for every symbol in ``symbols``."""
    return InMemoryDataSource(frames=dict.fromkeys(symbols, frame), errors=errors)


class TestUniverses:
    """Test universe definitions."""

    def test_universes_has_expected_keys(self):
        expected = ["sp500", "etfs", "full_market"]
        for key in expected:
            assert key in UNIVERSES, f"Missing universe: {key}"

    def test_full_market_is_superset(self):
        """full_market should contain all S&P 500 stocks plus ETFs."""
        full = set(UNIVERSES["full_market"])
        sp500 = set(UNIVERSES["sp500"])
        etfs = set(UNIVERSES["etfs"])
        assert sp500.issubset(full)
        assert etfs.issubset(full)

    def test_no_empty_universes(self):
        for name, symbols in UNIVERSES.items():
            assert len(symbols) > 0, f"Universe '{name}' is empty"

    def test_full_market_has_expected_size(self):
        """full_market should have ~550 symbols (S&P 500 + ETFs, deduplicated)."""
        assert len(UNIVERSES["full_market"]) >= 450
        assert len(UNIVERSES["full_market"]) <= 600


class TestBuildSp500Symbols:
    """Test _build_sp500_symbols fallback behavior."""

    def test_fallback_on_failure(self, mocker):
        """When pytickersymbols fails, should return hardcoded fallback symbols."""
        mocker.patch(
            "volume_price_analysis.analysis.PyTickerSymbols",
            side_effect=RuntimeError("simulated failure"),
        )
        result = _build_sp500_symbols()
        assert len(result) > 0
        assert "AAPL" in result
        assert "MSFT" in result


class TestRunOptionsAnalysis:
    """Test run_options_analysis with real fixture data."""

    def test_returns_expected_keys(self, sample_stock_data):
        result = run_options_analysis("TEST", sample_stock_data, holding_period=14)
        assert result["symbol"] == "TEST"
        assert "composite_signal" in result
        assert "trend_analysis" in result
        assert "volume_indicators" in result
        assert "price_indicators" in result
        assert "volatility_analysis" in result
        assert "volume_profile" in result
        assert "time_decay" in result
        assert "options_insights" in result
        assert "latest_price" in result

    def test_iv_percentile_proxy_is_hv_flagged(self, sample_stock_data):
        """The deep-analysis volatility proxy carries the HV basis + hv_percentile."""
        result = run_options_analysis("TEST", sample_stock_data, holding_period=14)
        proxy = result["volatility_analysis"]["iv_percentile_proxy"]
        assert proxy["basis"] == "historical_volatility"
        assert proxy["is_proxy"] is True
        assert proxy["hv_percentile"] == proxy["percentile"]

    def test_composite_signal_has_score(self, sample_stock_data):
        result = run_options_analysis("TEST", sample_stock_data)
        signal = result["composite_signal"]
        assert "score" in signal
        assert "recommendation" in signal
        assert "action" in signal
        assert -10 <= signal["score"] <= 10

    def test_includes_headline(self, sample_stock_data):
        """Additive top-line headline summarises the composite call (O4)."""
        result = run_options_analysis("TEST", sample_stock_data)
        headline = result["headline"]
        assert set(headline) == {
            "recommendation",
            "composite_score",
            "signal_quality",
            "rationale",
        }
        # Headline must agree with the detailed composite_signal it summarises.
        assert headline["recommendation"] == result["composite_signal"]["recommendation"]
        assert headline["composite_score"] == round(result["composite_signal"]["score"], 2)

    def test_squeeze_verdict_consistent_across_holding_periods(self, sample_stock_data):
        """The squeeze verdict is holding-period-independent (canonical 20-period bands)."""
        results = {
            hp: run_options_analysis("TEST", sample_stock_data, holding_period=hp)
            for hp in (10, 21, 30)
        }
        verdicts = {
            hp: r["volatility_analysis"]["bollinger_bands"]["squeeze_detected"]
            for hp, r in results.items()
        }
        canonical = detect_bollinger_squeeze(sample_stock_data)
        assert set(verdicts.values()) == {canonical}
        # The parameters block documents the canonical window the verdict used,
        # independent of the adaptive volume_window.
        for r in results.values():
            assert r["parameters"]["squeeze_window"] == SQUEEZE_WINDOW

    def test_adaptive_periods_short(self, sample_stock_data):
        result = run_options_analysis("TEST", sample_stock_data, holding_period=14)
        assert result["parameters"]["holding_period"] == 14
        assert result["parameters"]["mfi_period"] == 7
        assert result["parameters"]["volume_window"] == 10

    def test_adaptive_periods_medium(self, sample_stock_data):
        result = run_options_analysis("TEST", sample_stock_data, holding_period=21)
        assert result["parameters"]["mfi_period"] == 10
        assert result["parameters"]["volume_window"] == 14

    def test_adaptive_periods_long(self, sample_stock_data):
        result = run_options_analysis("TEST", sample_stock_data, holding_period=30)
        assert result["parameters"]["mfi_period"] == 14
        assert result["parameters"]["volume_window"] == 20

    def test_time_decay_risk_levels(self, sample_stock_data):
        # Short DTE = critical
        result = run_options_analysis("TEST", sample_stock_data, days_to_expiration=5)
        assert result["time_decay"]["theta_risk"] == "critical"

        # Long DTE = low
        result = run_options_analysis("TEST", sample_stock_data, days_to_expiration=25)
        assert result["time_decay"]["theta_risk"] == "low"

    def test_options_insights_is_list(self, sample_stock_data):
        result = run_options_analysis("TEST", sample_stock_data)
        assert isinstance(result["options_insights"], list)
        assert len(result["options_insights"]) > 0

    def test_uptrend_produces_positive_indicators(self, uptrend_data):
        result = run_options_analysis("TEST", uptrend_data)
        # In a clear uptrend, VWAP position should be "above"
        assert result["price_indicators"]["vwap"]["position"] == "above"

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise ValueError."""
        import pandas as pd

        with pytest.raises(ValueError, match="Empty DataFrame"):
            run_options_analysis("TEST", pd.DataFrame())

    def test_missing_columns_raises(self):
        """DataFrame missing required columns should raise ValueError."""
        import pandas as pd

        df = pd.DataFrame({"Close": [100], "Volume": [1000]})
        with pytest.raises(ValueError, match="missing required columns"):
            run_options_analysis("TEST", df)


class TestAnalyzeSingleSymbol:
    """Test analyze_single_symbol against an injected in-memory data source."""

    def test_raises_insufficient_data_when_too_few_rows(self):
        """Should raise InsufficientDataError (a 'skip', not an error) for <30 rows."""
        source = source_for(["TEST"], SMALL_FRAME)

        with pytest.raises(InsufficientDataError):
            analyze_single_symbol("TEST", "3mo", 14, 2.0, 20, 100, "any", data_source=source)

    def test_unknown_symbol_raises_value_error(self):
        """The in-memory source mirrors production: an unknown symbol is a ValueError."""
        source = InMemoryDataSource()

        with pytest.raises(ValueError, match="No data found for symbol"):
            analyze_single_symbol("NOPE", "3mo", 14, 2.0, 20, 100, "any", data_source=source)

    def test_fetches_with_the_requested_period(self, sample_stock_data):
        """The period argument reaches the data source rather than being dropped."""

        class RecordingSource(InMemoryDataSource):
            def __init__(self):
                super().__init__(frames={"TEST": sample_stock_data})
                self.periods = []

            def fetch(self, symbol, *, period="1mo", start=None, end=None, timeout=30):
                self.periods.append(period)
                return super().fetch(symbol, period=period, start=start, end=end, timeout=timeout)

        source = RecordingSource()
        analyze_single_symbol("TEST", "6mo", 14, 0, 0, 100, "any", data_source=source)
        assert source.periods == ["6mo"]

    def test_candidate_includes_hv_percentile(self, sample_stock_data):
        """A returned candidate carries the honestly-labeled hv_percentile twin."""
        source = source_for(["TEST"], sample_stock_data)

        candidate = analyze_single_symbol("TEST", "3mo", 14, 0, 0, 100, "any", data_source=source)
        assert candidate is not None
        assert "hv_percentile" in candidate
        assert candidate["hv_percentile"] == candidate["iv_percentile"]

    def test_candidate_adx_is_composite_adaptive_adx(self, sample_stock_data):
        """Reported ADX must be the composite's adaptive ADX, coherent with signal_quality.

        Regression for HOM-48: the scan previously reported a fixed ADX(14) that was
        incoherent with the composite's adaptive ADX(10) used for scoring/signal_quality.
        """
        source = source_for(["TEST"], sample_stock_data)

        candidate = analyze_single_symbol("TEST", "3mo", 14, 0, 0, 100, "any", data_source=source)
        assert candidate is not None

        composite = calculate_composite_score(sample_stock_data, 14)
        # holding_period <= 14 -> adaptive ADX(10)
        assert candidate["adx_period"] == 10
        assert candidate["adx_period"] == composite["adx_period"]
        assert candidate["adx"] == pytest.approx(round(composite["adx_summary"]["adx"], 1))
        assert candidate["trend_strength"] == composite["adx_summary"]["trend_strength"]
        assert candidate["trend_direction"] == composite["adx_summary"]["trend_direction"]

    def test_candidate_adx_differs_from_legacy_fixed_adx14(self, sample_stock_data):
        """On this fixture the adaptive ADX(10) and legacy ADX(14) actually diverge,
        so the change is observable (not a no-op)."""
        source = source_for(["TEST"], sample_stock_data)

        candidate = analyze_single_symbol("TEST", "3mo", 14, 0, 0, 100, "any", data_source=source)
        assert candidate is not None

        # On sample_stock_data, ADX(10) ~= 20.6 vs ADX(14) ~= 20.8, so they round
        # differently at 1 dp -- the adaptive switch is observable, not a no-op.
        legacy_adx14 = round(calculate_adx(sample_stock_data, 14)["adx"], 1)
        assert candidate["adx"] != legacy_adx14
        assert candidate["adx"] == pytest.approx(
            round(calculate_adx(sample_stock_data, 10)["adx"], 1)
        )


class TestRunScan:
    """Test run_scan orchestration against an injected in-memory data source."""

    @pytest.mark.asyncio
    async def test_custom_symbols_override_universe(self):
        """When custom symbols are provided, universe should be 'custom'."""
        source = source_for(["AAPL", "MSFT"], SMALL_FRAME)

        result = await run_scan(symbols=["AAPL", "MSFT"], universe="tech", data_source=source)
        assert result["scan_parameters"]["universe"] == "custom"
        assert result["scan_parameters"]["symbols_in_universe"] == 2

    @pytest.mark.asyncio
    async def test_invalid_universe_falls_back_to_full_market(self):
        """Unknown universe should fall back to full_market."""
        source = source_for(UNIVERSES["full_market"], SMALL_FRAME)

        result = await run_scan(universe="nonexistent", symbols=None, data_source=source)
        assert result["scan_parameters"]["universe"] == "full_market"

    @pytest.mark.asyncio
    async def test_invalid_universe_logs_warning(self, mocker):
        """Unknown universe should emit a WARNING naming the bad value and the fallback."""
        source = source_for(UNIVERSES["full_market"], SMALL_FRAME)
        mock_warning = mocker.patch("volume_price_analysis.analysis.logger.warning")

        await run_scan(universe="tech", symbols=None, data_source=source)

        assert mock_warning.called, "Expected logger.warning to be called for unknown universe"
        calls_text = " ".join(str(c) for c in mock_warning.call_args_list)
        assert "tech" in calls_text, f"Expected 'tech' in warning call; got: {calls_text}"
        assert "full_market" in calls_text, f"Expected 'full_market' in warning; got: {calls_text}"

    @pytest.mark.asyncio
    async def test_scan_result_structure(self):
        """Verify scan results have expected structure."""
        source = source_for(["TEST"], SMALL_FRAME)

        result = await run_scan(symbols=["TEST"], data_source=source)
        assert "scan_parameters" in result
        assert "summary" in result
        assert "high_conviction_setups" in result
        assert "top_bullish" in result
        assert "top_bearish" in result
        # HOM-39 additive fields
        assert result["scan_parameters"]["volatility_basis"] == "historical_volatility"
        assert isinstance(result["scan_parameters"]["symbols_scanned"], int)
        assert "skipped" in result["summary"]
        assert isinstance(result["errors"], list)
        # HOM-48: scan reports which ADX period backs the reported adx / thresholds
        assert result["scan_parameters"]["adx_period"] == 10  # holding_period default 14

    @pytest.mark.asyncio
    async def test_scan_parameters_adx_period_tracks_holding_period(self, sample_stock_data):
        """The reported adx_period is adaptive: ADX(10) for short holds, ADX(14) otherwise.

        Lets clients interpret the now-adaptive `adx`, `min_adx`, and the
        high_conviction (adx >= 28) gate against the correct period (HOM-48).
        """
        source = source_for(["TEST"], sample_stock_data)

        short = await run_scan(
            symbols=["TEST"], holding_period=14, min_score=0, min_adx=0, data_source=source
        )
        assert short["scan_parameters"]["adx_period"] == 10

        long = await run_scan(
            symbols=["TEST"], holding_period=21, min_score=0, min_adx=0, data_source=source
        )
        assert long["scan_parameters"]["adx_period"] == 14
        # Each candidate's own adx_period agrees with the scan-level value.
        for candidate in long["top_bullish"] + long["top_bearish"]:
            assert candidate["adx_period"] == 14

    @pytest.mark.asyncio
    async def test_scan_handles_errors_gracefully(self):
        """Errors for individual symbols should be captured, not crash the scan."""
        # BADTICKER is absent from the source, so it fails exactly as production would.
        source = InMemoryDataSource()

        result = await run_scan(symbols=["BADTICKER"], data_source=source)
        assert result["summary"]["errors"] >= 1
        assert result["summary"]["total_candidates"] == 0
        # A genuine fetch error is an error, not a skip.
        assert result["summary"]["skipped"] == 0

    @pytest.mark.asyncio
    async def test_summary_reports_skipped_count(self):
        """Insufficient-data symbols are counted as 'skipped', distinct from errors."""
        source = source_for(["AAA", "BBB", "CCC"], SMALL_FRAME)

        result = await run_scan(symbols=["AAA", "BBB", "CCC"], data_source=source)
        assert result["summary"]["skipped"] == 3
        assert result["summary"]["errors"] == 0
        assert result["summary"]["total_candidates"] == 0
        # Skipped symbols are NOT counted as scanned.
        assert result["scan_parameters"]["symbols_scanned"] == 0

    @pytest.mark.asyncio
    async def test_errors_field_is_always_a_list(self):
        """The top-level 'errors' field must be a list even when empty (never None)."""
        source = source_for(["AAA"], SMALL_FRAME)

        result = await run_scan(symbols=["AAA"], data_source=source)
        assert isinstance(result["errors"], list)
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_scan_accounting_is_complete(self):
        """scanned + skipped + errors must equal the universe size (honest diagnostics)."""
        source = source_for(["AAA", "BBB"], SMALL_FRAME)  # BADX deliberately absent

        result = await run_scan(symbols=["AAA", "BBB", "BADX"], data_source=source)
        summary = result["summary"]
        params = result["scan_parameters"]
        total = params["symbols_scanned"] + summary["skipped"] + summary["errors"]
        assert total == params["symbols_in_universe"] == 3
        assert summary["skipped"] == 2
        assert summary["errors"] == 1

    @pytest.mark.asyncio
    async def test_all_three_buckets_with_real_candidate(self, mocker):
        """A scanned candidate, a skip, and an error each land in exactly one bucket."""

        def _analyze(symbol, *args, **kwargs):
            if symbol == "SMALL":
                raise InsufficientDataError("SMALL: insufficient history")
            if symbol == "BADX":
                raise ValueError("No data found")
            return {
                "symbol": symbol,
                "composite_score": 3.0,
                "recommendation": "bullish",
                "signal_quality": "medium",
                "adx": 25.0,
                "trend_strength": "moderate",
                "trend_direction": "up",
                "rsi": 55.0,
                "rsi_divergence": "none",
                "iv_percentile": 40.0,
                "hv_percentile": 40.0,
                "iv_implication": "neutral",
                "expected_move_pct": 3.0,
                "rvol": 1.2,
                "latest_price": 100.0,
                "key_levels": {"upper_target": 103.0, "lower_target": 97.0},
            }

        mocker.patch("volume_price_analysis.analysis.analyze_single_symbol", side_effect=_analyze)

        result = await run_scan(symbols=["GOOD", "SMALL", "BADX"], min_score=0, min_adx=0)
        summary = result["summary"]
        params = result["scan_parameters"]

        assert params["symbols_scanned"] == 1
        assert summary["skipped"] == 1
        assert summary["errors"] == 1
        assert summary["total_candidates"] == 1
        # Exhaustive, mutually-exclusive accounting.
        assert (
            params["symbols_scanned"] + summary["skipped"] + summary["errors"]
            == params["symbols_in_universe"]
            == 3
        )
        assert [c["symbol"] for c in result["top_bullish"]] == ["GOOD"]
        assert result["top_bullish"][0]["hv_percentile"] == 40.0
        assert result["errors"][0]["symbol"] == "BADX"

    @pytest.mark.asyncio
    async def test_rejects_too_many_symbols(self):
        """Passing more than 500 symbols should raise ValueError."""
        symbols = [f"SYM{i}" for i in range(501)]
        with pytest.raises(ValueError, match="Maximum is 500"):
            await run_scan(symbols=symbols)

    @pytest.mark.asyncio
    async def test_accepts_max_symbols(self):
        """Passing exactly 500 symbols should NOT raise a symbol-limit error."""
        symbols = [f"SYM{i}" for i in range(500)]
        source = source_for(symbols, SMALL_FRAME)

        result = await run_scan(symbols=symbols, data_source=source)
        # Should complete without the "Too many symbols" error
        assert result["scan_parameters"]["symbols_in_universe"] == 500

    @pytest.mark.asyncio
    async def test_zero_score_candidate_in_bullish(self, mocker):
        """A candidate with composite_score == 0 should appear in top_bullish when min_score=0."""
        mocker.patch(
            "volume_price_analysis.analysis.analyze_single_symbol",
            return_value={
                "symbol": "FLAT",
                "composite_score": 0.0,
                "recommendation": "neutral",
                "signal_quality": "low",
                "adx": 25.0,
                "trend_strength": "weak",
                "trend_direction": "neutral",
                "rsi": 50.0,
                "rsi_divergence": "none",
                "iv_percentile": 50.0,
                "hv_percentile": 50.0,
                "iv_implication": "average",
                "expected_move_pct": 3.0,
                "rvol": 1.0,
                "latest_price": 100.0,
                "key_levels": {"upper_target": 103.0, "lower_target": 97.0},
            },
        )

        result = await run_scan(symbols=["FLAT"], min_score=0, min_adx=0)
        assert result["summary"]["total_candidates"] == 1
        assert len(result["top_bullish"]) == 1
        assert result["top_bullish"][0]["symbol"] == "FLAT"

    @pytest.mark.asyncio
    async def test_invalid_direction_raises(self):
        """An invalid direction value should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid direction"):
            await run_scan(symbols=["AAPL"], direction="bulish")

    @pytest.mark.asyncio
    async def test_scan_timeout(self, mocker):
        """When asyncio.wait_for raises TimeoutError, run_scan should raise ValueError."""
        mocker.patch(
            "volume_price_analysis.analysis.asyncio.wait_for",
            side_effect=TimeoutError,
        )

        with pytest.raises(ValueError, match="timed out"):
            await run_scan(symbols=["AAPL"])


class TestBuildHeadline:
    """Test the additive top-line headline derived from a composite-score result."""

    def _composite(self, score, recommendation, quality, breakdown):
        return {
            "composite_score": score,
            "recommendation": recommendation,
            "signal_quality": quality,
            "quality_note": "note",
            "score_breakdown": breakdown,
        }

    def test_returns_expected_keys(self):
        composite = self._composite(
            6.0, "strong_bullish", "high", {"price_vs_vwap": 2, "obv_momentum": 2}
        )
        headline = build_headline(composite)
        assert set(headline) == {
            "recommendation",
            "composite_score",
            "signal_quality",
            "rationale",
        }
        assert headline["recommendation"] == "strong_bullish"
        assert headline["composite_score"] == 6.0
        assert headline["signal_quality"] == "high"

    def test_bullish_rationale_names_aligned_drivers(self):
        composite = self._composite(
            4.0,
            "bullish",
            "high",
            {"price_vs_vwap": 2, "obv_momentum": 2, "rsi": -1, "cmf": 0},
        )
        rationale = build_headline(composite)["rationale"]
        assert "bullish" in rationale.lower()
        # Strongest positive drivers should be named; negative ones excluded.
        assert "VWAP" in rationale
        assert "OBV" in rationale
        assert "RSI" not in rationale

    def test_bearish_rationale_names_aligned_drivers(self):
        composite = self._composite(
            -5.0,
            "strong_bearish",
            "high",
            {"price_vs_vwap": -2, "rsi": -2, "mfi": -1, "obv_momentum": 1},
        )
        rationale = build_headline(composite)["rationale"]
        assert "bearish" in rationale.lower()
        assert "RSI" in rationale
        # A bullish-signed driver must not be cited for a bearish call.
        assert "OBV" not in rationale

    def test_neutral_rationale_flags_no_edge(self):
        composite = self._composite(0.7, "neutral", "low", {"price_vs_vwap": 1, "rsi": -1})
        rationale = build_headline(composite)["rationale"]
        assert "no clear" in rationale.lower() or "mixed" in rationale.lower()

    def test_rounds_score(self):
        composite = self._composite(4.66667, "bullish", "medium", {"price_vs_vwap": 2})
        assert build_headline(composite)["composite_score"] == 4.67

    def test_handles_missing_breakdown(self):
        composite = {
            "composite_score": 3.0,
            "recommendation": "bullish",
            "signal_quality": "medium",
        }
        headline = build_headline(composite)
        assert headline["recommendation"] == "bullish"
        assert isinstance(headline["rationale"], str)
        assert headline["rationale"]


def _synthetic_ohlcv(periods: int = 60, trend: float = 1.0):
    """Build a deterministic OHLCV frame for pure scoring tests (no mocking)."""
    import pandas as pd

    closes = [100.0 + trend * i + (i % 3) * 0.5 for i in range(periods)]
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=periods, freq="D"),
            "Open": [c - 0.5 for c in closes],
            "High": [c + 1.5 for c in closes],
            "Low": [c - 1.5 for c in closes],
            "Close": closes,
            "Volume": [1_000_000 + (i % 7) * 25_000 for i in range(periods)],
        }
    )


class TestScoreSymbol:
    """score_symbol is pure: synthetic DataFrames in, candidate dict out."""

    def test_raises_insufficient_data(self):
        data = _synthetic_ohlcv(periods=10)
        with pytest.raises(InsufficientDataError, match="10 bars"):
            score_symbol(data, "TEST", 14, 2.0, 20, 100, "any")

    def test_returns_candidate_shape(self):
        candidate = score_symbol(_synthetic_ohlcv(), "TEST", 14, 0, 0, 100, "any")
        assert candidate is not None
        assert candidate["symbol"] == "TEST"
        for key in (
            "composite_score",
            "recommendation",
            "signal_quality",
            "adx",
            "adx_period",
            "rsi",
            "iv_percentile",
            "hv_percentile",
            "expected_move_pct",
            "rvol",
            "latest_price",
            "key_levels",
        ):
            assert key in candidate
        assert candidate["hv_percentile"] == candidate["iv_percentile"]

    def test_min_score_filter_rejects(self):
        data = _synthetic_ohlcv()
        assert score_symbol(data, "TEST", 14, 0, 0, 100, "any") is not None
        # No symbol can score above 10, so an 11 floor must reject everything.
        assert score_symbol(data, "TEST", 14, 11, 0, 100, "any") is None

    def test_min_adx_filter_rejects(self):
        data = _synthetic_ohlcv()
        assert score_symbol(data, "TEST", 14, 0, 1000, 100, "any") is None

    def test_max_iv_filter_rejects(self):
        data = _synthetic_ohlcv()
        assert score_symbol(data, "TEST", 14, 0, 0, -1, "any") is None

    def test_min_volume_filter_rejects(self):
        data = _synthetic_ohlcv()
        assert score_symbol(data, "TEST", 14, 0, 0, 100, "any", min_avg_volume=10**12) is None
        assert score_symbol(data, "TEST", 14, 0, 0, 100, "any", min_avg_volume=1) is not None

    def test_direction_filter(self):
        """An uptrending series scores positive, so 'bearish' must reject it."""
        data = _synthetic_ohlcv(trend=1.0)
        bull = score_symbol(data, "TEST", 14, 0, 0, 100, "bullish")
        assert bull is not None
        assert bull["composite_score"] > 0
        assert score_symbol(data, "TEST", 14, 0, 0, 100, "bearish") is None

    def test_adx_period_follows_holding_period(self):
        data = _synthetic_ohlcv()
        fast = score_symbol(data, "TEST", 14, 0, 0, 100, "any")
        slow = score_symbol(data, "TEST", 30, 0, 0, 100, "any")
        assert fast is not None and slow is not None
        assert fast["adx_period"] == 10
        assert slow["adx_period"] == 14

    def test_matches_analyze_single_symbol(self, sample_stock_data):
        """analyze_single_symbol is fetch + score_symbol; results must agree."""
        source = InMemoryDataSource(frames={"TEST": sample_stock_data})
        via_fetch = analyze_single_symbol("TEST", "3mo", 14, 0, 0, 100, "any", data_source=source)
        direct = score_symbol(sample_stock_data, "TEST", 14, 0, 0, 100, "any")
        assert via_fetch == direct


class TestAdaptivePeriods:
    """Boundary behavior of the <=14 / <=21 / else breakpoints."""

    FAST = {
        "mfi_period": 7,
        "volume_window": 10,
        "rsi_period": 7,
        "adx_period": 10,
        "hv_window": 10,
    }
    BALANCED = {
        "mfi_period": 10,
        "volume_window": 14,
        "rsi_period": 10,
        "adx_period": 14,
        "hv_window": 14,
    }
    STANDARD = {
        "mfi_period": 14,
        "volume_window": 20,
        "rsi_period": 14,
        "adx_period": 14,
        "hv_window": 20,
    }

    def test_boundary_14(self):
        assert _adaptive_periods(14) == self.FAST

    def test_boundary_15(self):
        assert _adaptive_periods(15) == self.BALANCED

    def test_boundary_21(self):
        assert _adaptive_periods(21) == self.BALANCED

    def test_boundary_22(self):
        assert _adaptive_periods(22) == self.STANDARD

    def test_matches_reported_parameters(self):
        """run_options_analysis must report exactly these periods."""
        for hp in (14, 15, 21, 22):
            result = run_options_analysis("TEST", _synthetic_ohlcv(), holding_period=hp)
            expected = _adaptive_periods(hp)
            for key, value in expected.items():
                assert result["parameters"][key] == value


class TestGetUniverses:
    """Lazy universe construction."""

    def test_returns_expected_keys(self):
        universes = get_universes()
        assert set(universes) == {"sp500", "etfs", "full_market"}

    def test_cached_between_calls(self):
        """The expensive build runs once: repeat calls are lru_cache hits.

        Equality alone would pass even without caching, so assert on the cache
        itself and on identity of the inner lists (get_universes copies only the
        outer mapping).
        """
        get_universes()  # ensure the cache is warm
        before = _cached_universes.cache_info().hits
        first = get_universes()
        second = get_universes()
        assert _cached_universes.cache_info().hits == before + 2
        assert first["sp500"] is second["sp500"]
        assert first["full_market"] is second["full_market"]

    def test_caller_copy_does_not_mutate_cache(self):
        universes = get_universes()
        universes["injected"] = ["ZZZ"]
        assert "injected" not in get_universes()

    def test_legacy_module_attribute_still_works(self):
        from volume_price_analysis import analysis

        assert analysis.UNIVERSES["full_market"] == get_universes()["full_market"]

    def test_unknown_module_attribute_raises(self):
        from volume_price_analysis import analysis

        with pytest.raises(AttributeError):
            _ = analysis.NOT_A_REAL_ATTRIBUTE


class TestRunScanInjectedUniverses:
    """run_scan accepts an injected universe mapping and a timeout budget."""

    @pytest.mark.asyncio
    async def test_universes_override_is_used(self, mocker):
        mocker.patch(
            "volume_price_analysis.analysis.analyze_single_symbol",
            return_value=None,
        )
        result = await run_scan(
            universe="tiny",
            universes={"tiny": ["AAA", "BBB"], "full_market": ["ZZZ"]},
        )
        assert result["scan_parameters"]["universe"] == "tiny"
        assert result["scan_parameters"]["symbols_in_universe"] == 2

    @pytest.mark.asyncio
    async def test_unknown_universe_falls_back_within_override(self, mocker):
        mocker.patch(
            "volume_price_analysis.analysis.analyze_single_symbol",
            return_value=None,
        )
        result = await run_scan(
            universe="nope",
            universes={"tiny": ["AAA"], "full_market": ["ZZZ"]},
        )
        assert result["scan_parameters"]["universe"] == "full_market"
        assert result["scan_parameters"]["symbols_in_universe"] == 1

    @pytest.mark.asyncio
    async def test_unknown_universe_without_full_market_raises(self, mocker):
        """An override with no 'full_market' has nothing to fall back to."""
        mocker.patch(
            "volume_price_analysis.analysis.analyze_single_symbol",
            return_value=None,
        )
        with pytest.raises(ValueError, match="no 'full_market' fallback"):
            await run_scan(universe="nope", universes={"tiny": ["AAA"]})

    @pytest.mark.asyncio
    async def test_timeout_seconds_is_reported_in_error(self, mocker):
        mocker.patch(
            "volume_price_analysis.analysis.asyncio.wait_for",
            side_effect=TimeoutError,
        )
        with pytest.raises(ValueError, match="timed out after 5 seconds"):
            await run_scan(symbols=["AAPL"], timeout_seconds=5)
