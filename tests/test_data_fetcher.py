"""Tests for stock data fetching functionality."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from volume_price_analysis.data_fetcher import fetch_stock_data, validate_symbol


class TestFetchStockData:
    """Tests for fetch_stock_data function."""

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_fetch_stock_data_with_period(self, mock_ticker):
        """Test fetching stock data with a period parameter."""
        # Create mock data
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range(start="2024-01-01", periods=3, freq="D"),
        )
        mock_data.index.name = "Date"

        # Setup mock
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        # Fetch data
        result = fetch_stock_data("AAPL", period="1mo")

        # Assertions
        assert len(result) == 3
        assert "Date" in result.columns
        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns
        mock_ticker_instance.history.assert_called_once_with(period="1mo")

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_fetch_stock_data_with_dates(self, mock_ticker):
        """Test fetching stock data with start and end dates."""
        # Create mock data
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [101, 102],
                "Low": [99, 100],
                "Close": [100.5, 101.5],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range(start="2024-01-01", periods=2, freq="D"),
        )
        mock_data.index.name = "Date"

        # Setup mock
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        # Fetch data
        result = fetch_stock_data("MSFT", start_date="2024-01-01", end_date="2024-01-02")

        # Assertions
        assert len(result) == 2
        mock_ticker_instance.history.assert_called_once_with(start="2024-01-01", end="2024-01-02")

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_fetch_stock_data_empty_result(self, mock_ticker):
        """Test handling of empty data result."""
        # Setup mock to return empty DataFrame
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        # Should raise ValueError
        with pytest.raises(ValueError, match="No data found for symbol"):
            fetch_stock_data("INVALID")

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_fetch_stock_data_columns_filtered(self, mock_ticker):
        """Test that only required columns are returned."""
        # Create mock data with extra columns
        mock_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100.5],
                "Volume": [1000000],
                "Dividends": [0],
                "Stock Splits": [0],
            },
            index=pd.date_range(start="2024-01-01", periods=1, freq="D"),
        )
        mock_data.index.name = "Date"

        # Setup mock
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        # Fetch data
        result = fetch_stock_data("AAPL")

        # Only required columns should be present
        expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert list(result.columns) == expected_columns

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_fetch_stock_data_date_as_column(self, mock_ticker):
        """Test that Date is returned as a column, not index."""
        mock_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100.5],
                "Volume": [1000000],
            },
            index=pd.date_range(start="2024-01-01", periods=1, freq="D"),
        )
        mock_data.index.name = "Date"

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        result = fetch_stock_data("AAPL")

        # Date should be a column, not the index
        assert "Date" in result.columns
        assert result.index.name != "Date"


class TestValidateSymbol:
    """Tests for validate_symbol function."""

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_validate_symbol_valid(self, mock_ticker):
        """Test validation of a valid symbol."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {"symbol": "AAPL", "shortName": "Apple Inc."}
        mock_ticker.return_value = mock_ticker_instance

        result = validate_symbol("AAPL")
        assert result is True

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_validate_symbol_with_shortname_only(self, mock_ticker):
        """Test validation when only shortName is present."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {"shortName": "Apple Inc."}
        mock_ticker.return_value = mock_ticker_instance

        result = validate_symbol("AAPL")
        assert result is True

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_validate_symbol_invalid(self, mock_ticker):
        """Test validation of an invalid symbol."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {}
        mock_ticker.return_value = mock_ticker_instance

        result = validate_symbol("INVALID123")
        assert result is False

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_validate_symbol_exception(self, mock_ticker):
        """Test validation when exception is raised."""
        mock_ticker.side_effect = Exception("Network error")

        result = validate_symbol("AAPL")
        assert result is False


class TestIntegration:
    """Integration tests (require network access - skip in CI)."""

    @pytest.mark.skip(reason="Requires network access")
    def test_fetch_real_stock_data(self):
        """Integration test with real Yahoo Finance data."""
        # This test is skipped by default but can be run manually
        result = fetch_stock_data("AAPL", period="5d")

        assert len(result) > 0
        expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert all(col in result.columns for col in expected_cols)
        assert result["Close"].iloc[-1] > 0
        assert result["Volume"].iloc[-1] > 0

    @pytest.mark.skip(reason="Requires network access")
    def test_validate_real_symbol(self):
        """Integration test with real symbol validation."""
        assert validate_symbol("AAPL") is True
        assert validate_symbol("MSFT") is True
        assert validate_symbol("INVALIDXYZ123") is False
