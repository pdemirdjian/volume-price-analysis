"""Tests for stock data fetching functionality."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from volume_price_analysis.data_fetcher import (
    DEFAULT_TIMEOUT,
    VALID_PERIODS,
    DataSource,
    InMemoryDataSource,
    YFinanceDataSource,
    fetch_stock_data,
    get_default_data_source,
    validate_symbol,
    validate_symbol_format,
)


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
        mock_ticker_instance.history.assert_called_once_with(period="1mo", timeout=DEFAULT_TIMEOUT)

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
        mock_ticker_instance.history.assert_called_once_with(
            start="2024-01-01", end="2024-01-02", timeout=DEFAULT_TIMEOUT
        )

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
        mock_fast_info = Mock()
        mock_fast_info.last_price = 150.0
        mock_ticker_instance.fast_info = mock_fast_info
        mock_ticker.return_value = mock_ticker_instance

        result = validate_symbol("AAPL")
        assert result is True

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_validate_symbol_with_shortname_only(self, mock_ticker):
        """Test validation when fast_info has last_price."""
        mock_ticker_instance = Mock()
        mock_fast_info = Mock()
        mock_fast_info.last_price = 175.5
        mock_ticker_instance.fast_info = mock_fast_info
        mock_ticker.return_value = mock_ticker_instance

        result = validate_symbol("AAPL")
        assert result is True

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_validate_symbol_invalid(self, mock_ticker):
        """Test validation of an invalid symbol (no price data)."""
        mock_ticker_instance = Mock()
        mock_fast_info = Mock()
        mock_fast_info.last_price = None
        mock_ticker_instance.fast_info = mock_fast_info
        mock_ticker.return_value = mock_ticker_instance

        result = validate_symbol("INVALID")
        assert result is False

    def test_validate_symbol_invalid_format(self):
        """Test validation of symbol with invalid format."""
        # Too long
        assert validate_symbol("VERYLONGSYMBOL") is False
        # Invalid characters
        assert validate_symbol("AAPL!@#") is False
        # Empty
        assert validate_symbol("") is False

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_validate_symbol_exception(self, mock_ticker):
        """Test validation when exception is raised."""
        mock_ticker.side_effect = Exception("Network error")

        result = validate_symbol("AAPL")
        assert result is False


class TestValidateSymbolFormat:
    """Tests for validate_symbol_format function."""

    def test_valid_formats(self):
        """Test valid symbol formats."""
        assert validate_symbol_format("AAPL") is True
        assert validate_symbol_format("MSFT") is True
        assert validate_symbol_format("BRK-A") is True  # Berkshire A shares
        assert validate_symbol_format("BRK-B") is True  # Berkshire B shares
        assert validate_symbol_format("SPY") is True
        assert validate_symbol_format("VIX") is True
        assert validate_symbol_format("^GSPC") is True  # S&P 500 index
        assert validate_symbol_format("^DJI") is True  # Dow Jones

    def test_invalid_formats(self):
        """Test invalid symbol formats."""
        assert validate_symbol_format("") is False
        assert validate_symbol_format("VERYLONGSYMBOL") is False  # Too long
        assert validate_symbol_format("AAPL!@#") is False  # Invalid chars
        assert validate_symbol_format("AAPL ") is False  # Contains space
        assert validate_symbol_format(None) is False  # type: ignore[arg-type]

    def test_non_string_types(self):
        """Test that non-string types return False."""
        assert validate_symbol_format(123) is False  # type: ignore[arg-type]
        assert validate_symbol_format(45.67) is False  # type: ignore[arg-type]
        assert validate_symbol_format(["AAPL"]) is False  # type: ignore[arg-type]
        assert validate_symbol_format({"symbol": "AAPL"}) is False  # type: ignore[arg-type]


class TestInputValidation:
    """Tests for input validation in fetch_stock_data."""

    def test_invalid_period_raises_error(self):
        """Test that an invalid period value raises ValueError."""
        with pytest.raises(ValueError, match=r"Invalid period: 'invalid_period'"):
            fetch_stock_data("AAPL", period="invalid_period")

    @pytest.mark.parametrize("period", sorted(VALID_PERIODS))
    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_all_valid_periods_accepted(self, mock_ticker, period):
        """Test that every valid period is accepted without raising."""
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

        result = fetch_stock_data("AAPL", period=period)
        assert len(result) == 1

    def test_invalid_start_date_format_raises_error(self):
        """Test that an invalid start_date format raises ValueError."""
        with pytest.raises(ValueError, match=r"Invalid start_date format: 'not-a-date'"):
            fetch_stock_data("AAPL", start_date="not-a-date", end_date="2024-01-02")

    def test_invalid_end_date_format_raises_error(self):
        """Test that an invalid end_date format raises ValueError."""
        with pytest.raises(ValueError, match=r"Invalid end_date format: '01/02/2024'"):
            fetch_stock_data("AAPL", start_date="2024-01-01", end_date="01/02/2024")

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_period_ignored_when_dates_provided(self, mock_ticker):
        """Test that an invalid period is ignored when both dates are provided."""
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

        # Should NOT raise even though period is invalid — dates take priority
        result = fetch_stock_data(
            "AAPL", start_date="2024-01-01", end_date="2024-01-31", period="bogus"
        )
        assert len(result) == 1

    def test_reversed_date_range_raises_error(self):
        """Test that end_date before start_date raises ValueError."""
        with pytest.raises(ValueError, match="end_date must be on or after start_date"):
            fetch_stock_data("AAPL", start_date="2024-06-01", end_date="2024-01-01")

    @pytest.mark.parametrize(
        "bad_date",
        ["2024-13-45", "2024-02-30", "2024-00-01"],
    )
    def test_invalid_calendar_dates_raise_error(self, bad_date):
        """Test that syntactically valid but impossible calendar dates raise ValueError."""
        with pytest.raises(ValueError, match="Invalid calendar date"):
            fetch_stock_data("AAPL", start_date=bad_date, end_date="2025-01-01")

    def test_only_start_date_raises_error(self):
        """Test that providing only start_date raises ValueError."""
        with pytest.raises(ValueError, match="Both start_date and end_date must be provided"):
            fetch_stock_data("AAPL", start_date="2024-01-01")

    def test_only_end_date_raises_error(self):
        """Test that providing only end_date raises ValueError."""
        with pytest.raises(ValueError, match="Both start_date and end_date must be provided"):
            fetch_stock_data("AAPL", end_date="2024-01-31")

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_network_error_wrapped_in_valueerror(self, mock_ticker):
        """Test that yfinance network errors are wrapped in ValueError."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = ConnectionError("Network unreachable")
        mock_ticker.return_value = mock_ticker_instance

        with pytest.raises(ValueError, match="Failed to fetch data for AAPL"):
            fetch_stock_data("AAPL")

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_missing_critical_columns_raises_error(self, mock_ticker):
        """Test that missing OHLCV columns raise ValueError."""
        mock_data = pd.DataFrame(
            {"Close": [100.5], "Volume": [1000000]},
            index=pd.date_range(start="2024-01-01", periods=1, freq="D"),
        )
        mock_data.index.name = "Date"

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        with pytest.raises(ValueError, match="missing critical columns"):
            fetch_stock_data("AAPL")

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_extra_columns_are_stripped(self, mock_ticker):
        """Test that extra yfinance columns (Dividends, Stock Splits) are stripped."""
        mock_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100.5],
                "Volume": [1000000],
                "Dividends": [0.0],
                "Stock Splits": [0.0],
            },
            index=pd.date_range(start="2024-01-01", periods=1, freq="D"),
        )
        mock_data.index.name = "Date"

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        result = fetch_stock_data("AAPL")
        assert "Close" in result.columns
        assert "Dividends" not in result.columns
        assert "Stock Splits" not in result.columns

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_valid_date_formats_accepted(self, mock_ticker):
        """Test that valid YYYY-MM-DD date formats are accepted."""
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

        result = fetch_stock_data("AAPL", start_date="2024-01-01", end_date="2024-01-31")
        assert len(result) == 1


class TestDataSourceProtocol:
    """Both adapters must satisfy the DataSource seam."""

    def test_yfinance_adapter_is_a_data_source(self):
        assert isinstance(YFinanceDataSource(), DataSource)

    def test_in_memory_adapter_is_a_data_source(self):
        assert isinstance(InMemoryDataSource(), DataSource)

    def test_default_source_is_the_yfinance_adapter(self):
        assert isinstance(get_default_data_source(), YFinanceDataSource)

    def test_fetch_stock_data_delegates_to_the_default_source(self):
        """The legacy function is a thin wrapper, not a second code path."""
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=2),
                "Open": [1.0, 2.0],
                "High": [1.0, 2.0],
                "Low": [1.0, 2.0],
                "Close": [1.0, 2.0],
                "Volume": [10, 20],
            }
        )
        stub = InMemoryDataSource(frames={"AAPL": frame})
        with patch("volume_price_analysis.data_fetcher._DEFAULT_DATA_SOURCE", stub):
            result = fetch_stock_data("AAPL", "2024-01-01", "2024-01-31", "3mo", 12)

        assert list(result.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert stub.fetch_calls == ["AAPL"]


class TestInMemoryDataSource:
    """The test adapter must mirror production's failure contract."""

    def test_unknown_symbol_raises_value_error(self):
        with pytest.raises(ValueError, match="No data found for symbol"):
            InMemoryDataSource().fetch("AAPL")

    def test_returns_a_copy_so_callers_cannot_mutate_the_fixture(self):
        frame = pd.DataFrame({"Close": [1.0, 2.0]})
        source = InMemoryDataSource(frames={"AAPL": frame})

        returned = source.fetch("AAPL")
        returned.loc[:, "Close"] = 0.0

        assert list(frame["Close"]) == [1.0, 2.0]

    def test_configured_error_is_raised(self):
        source = InMemoryDataSource(errors={"AAPL": RuntimeError("boom")})
        with pytest.raises(RuntimeError, match="boom"):
            source.fetch("AAPL")

    def test_earnings_date_defaults_to_none(self):
        assert InMemoryDataSource().earnings_date("AAPL") is None

    def test_earnings_date_returns_configured_value(self):
        when = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
        assert InMemoryDataSource(earnings={"AAPL": when}).earnings_date("AAPL") == when


class TestYFinanceEarningsDate:
    """Parsing Yahoo's raw .info payload lives behind the adapter."""

    NOW = datetime(2026, 6, 28, 12, 0, tzinfo=UTC)

    def _source_with_info(self, info, mock_ticker):
        mock_ticker.return_value.info = info
        return YFinanceDataSource()

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_missing_earnings_date_returns_none(self, mock_ticker):
        source = self._source_with_info({}, mock_ticker)
        assert source.earnings_date("AAPL") is None

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_datetime_is_returned_unchanged(self, mock_ticker):
        when = self.NOW + timedelta(days=7)
        source = self._source_with_info({"earningsDate": when}, mock_ticker)
        assert source.earnings_date("AAPL") == when

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_epoch_int_is_converted_to_utc(self, mock_ticker):
        when = self.NOW + timedelta(days=5)
        source = self._source_with_info({"earningsDate": int(when.timestamp())}, mock_ticker)
        result = source.earnings_date("AAPL")
        assert result is not None
        assert result.tzinfo is not None
        assert int(result.timestamp()) == int(when.timestamp())

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_list_uses_first_element(self, mock_ticker):
        first = self.NOW + timedelta(days=3)
        second = first + timedelta(days=7)
        source = self._source_with_info({"earningsDate": [first, second]}, mock_ticker)
        assert source.earnings_date("AAPL") == first

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_naive_datetime_is_stamped_utc(self, mock_ticker):
        naive = datetime(2026, 7, 3, 12, 0)
        source = self._source_with_info({"earningsDate": naive}, mock_ticker)
        result = source.earnings_date("AAPL")
        assert result == naive.replace(tzinfo=UTC)

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_falls_back_to_earnings_timestamp(self, mock_ticker):
        when = self.NOW + timedelta(days=2)
        source = self._source_with_info({"earningsTimestamp": int(when.timestamp())}, mock_ticker)
        assert source.earnings_date("AAPL") is not None

    @patch("volume_price_analysis.data_fetcher.yf.Ticker")
    def test_unparseable_value_returns_none(self, mock_ticker):
        source = self._source_with_info({"earningsDate": "next Tuesday"}, mock_ticker)
        assert source.earnings_date("AAPL") is None

    @patch("volume_price_analysis.data_fetcher.yf.Ticker", side_effect=Exception("Network error"))
    def test_provider_failure_returns_none(self, mock_ticker):
        assert YFinanceDataSource().earnings_date("AAPL") is None


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
