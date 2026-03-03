"""Tests for analysis.py - extracted scan and options analysis logic."""

import pytest

from volume_price_analysis.analysis import (
    UNIVERSES,
    analyze_single_symbol,
    run_options_analysis,
    run_scan,
)


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

    def test_composite_signal_has_score(self, sample_stock_data):
        result = run_options_analysis("TEST", sample_stock_data)
        signal = result["composite_signal"]
        assert "score" in signal
        assert "recommendation" in signal
        assert "action" in signal
        assert -10 <= signal["score"] <= 10

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


class TestAnalyzeSingleSymbol:
    """Test analyze_single_symbol with mocked data fetching."""

    def test_returns_none_when_insufficient_data(self, mocker):
        """Should return None when data has fewer than 30 rows."""
        import pandas as pd

        small_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Open": [100] * 10,
                "High": [101] * 10,
                "Low": [99] * 10,
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            }
        )
        mocker.patch(
            "volume_price_analysis.analysis.fetch_stock_data",
            return_value=small_data,
        )

        result = analyze_single_symbol("TEST", "3mo", 14, 2.0, 20, 100, "any")
        assert result is None


class TestRunScan:
    """Test run_scan orchestration."""

    @pytest.mark.asyncio
    async def test_custom_symbols_override_universe(self, mocker):
        """When custom symbols are provided, universe should be 'custom'."""
        import pandas as pd

        # Mock fetch_stock_data to return insufficient data (quick skip)
        small_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Open": [100] * 10,
                "High": [101] * 10,
                "Low": [99] * 10,
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            }
        )
        mocker.patch(
            "volume_price_analysis.analysis.fetch_stock_data",
            return_value=small_data,
        )

        result = await run_scan(symbols=["AAPL", "MSFT"], universe="tech")
        assert result["scan_parameters"]["universe"] == "custom"
        assert result["scan_parameters"]["symbols_in_universe"] == 2

    @pytest.mark.asyncio
    async def test_invalid_universe_falls_back_to_full_market(self, mocker):
        """Unknown universe should fall back to full_market."""
        import pandas as pd

        small_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Open": [100] * 10,
                "High": [101] * 10,
                "Low": [99] * 10,
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            }
        )
        mocker.patch(
            "volume_price_analysis.analysis.fetch_stock_data",
            return_value=small_data,
        )

        result = await run_scan(universe="nonexistent", symbols=None)
        assert result["scan_parameters"]["universe"] == "full_market"

    @pytest.mark.asyncio
    async def test_scan_result_structure(self, mocker):
        """Verify scan results have expected structure."""
        import pandas as pd

        small_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Open": [100] * 10,
                "High": [101] * 10,
                "Low": [99] * 10,
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            }
        )
        mocker.patch(
            "volume_price_analysis.analysis.fetch_stock_data",
            return_value=small_data,
        )

        result = await run_scan(symbols=["TEST"])
        assert "scan_parameters" in result
        assert "summary" in result
        assert "high_conviction_setups" in result
        assert "top_bullish" in result
        assert "top_bearish" in result

    @pytest.mark.asyncio
    async def test_scan_handles_errors_gracefully(self, mocker):
        """Errors for individual symbols should be captured, not crash the scan."""
        mocker.patch(
            "volume_price_analysis.analysis.fetch_stock_data",
            side_effect=ValueError("No data found"),
        )

        result = await run_scan(symbols=["BADTICKER"])
        assert result["summary"]["errors"] >= 1
        assert result["summary"]["total_candidates"] == 0

    @pytest.mark.asyncio
    async def test_rejects_too_many_symbols(self):
        """Passing more than 500 symbols should raise ValueError."""
        symbols = [f"SYM{i}" for i in range(501)]
        with pytest.raises(ValueError, match="Maximum is 500"):
            await run_scan(symbols=symbols)

    @pytest.mark.asyncio
    async def test_accepts_max_symbols(self, mocker):
        """Passing exactly 500 symbols should NOT raise a symbol-limit error."""
        import pandas as pd

        small_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Open": [100] * 10,
                "High": [101] * 10,
                "Low": [99] * 10,
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            }
        )
        mocker.patch(
            "volume_price_analysis.analysis.fetch_stock_data",
            return_value=small_data,
        )

        symbols = [f"SYM{i}" for i in range(500)]
        result = await run_scan(symbols=symbols)
        # Should complete without the "Too many symbols" error
        assert result["scan_parameters"]["symbols_in_universe"] == 500

    @pytest.mark.asyncio
    async def test_scan_timeout(self, mocker):
        """When asyncio.wait_for raises TimeoutError, run_scan should raise ValueError."""
        mocker.patch(
            "volume_price_analysis.analysis.asyncio.wait_for",
            side_effect=TimeoutError,
        )

        with pytest.raises(ValueError, match="timed out"):
            await run_scan(symbols=["AAPL"])
