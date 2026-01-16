"""Tests for MCP server functionality."""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from volume_price_analysis.server import (
    generate_summary,
    handle_call_tool,
    handle_list_tools,
)


class TestListTools:
    """Tests for tool listing."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_all_tools(self):
        """Test that all expected tools are listed."""
        tools = await handle_list_tools()

        tool_names = [tool.name for tool in tools]

        expected_tools = [
            "get_stock_data",
            "calculate_obv",
            "calculate_vwap",
            "calculate_volume_profile",
            "calculate_mfi",
            "analyze_volume_trends",
            "comprehensive_analysis",
            "options_analysis",
            "scan_candidates",
        ]

        assert len(tools) == 9
        for expected_tool in expected_tools:
            assert expected_tool in tool_names

    @pytest.mark.asyncio
    async def test_tool_schemas_valid(self):
        """Test that all tools have valid input schemas."""
        tools = await handle_list_tools()

        for tool in tools:
            assert tool.name
            assert tool.description
            assert tool.inputSchema
            assert "type" in tool.inputSchema
            assert tool.inputSchema["type"] == "object"
            assert "properties" in tool.inputSchema
            # scan_candidates uses "symbols" (plural) instead of "symbol"
            if tool.name == "scan_candidates":
                assert "symbols" in tool.inputSchema["properties"]
            else:
                assert "symbol" in tool.inputSchema["properties"]


class TestCallToolGetStockData:
    """Tests for get_stock_data tool."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_get_stock_data_basic(self, mock_fetch):
        """Test basic stock data retrieval."""
        # Setup mock
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=5, freq="D"),
                "Open": [100, 101, 102, 103, 104],
                "High": [101, 102, 103, 104, 105],
                "Low": [99, 100, 101, 102, 103],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )
        mock_fetch.return_value = mock_data

        # Call tool
        result = await handle_call_tool(
            name="get_stock_data", arguments={"symbol": "AAPL", "period": "5d"}
        )

        # Parse result
        assert len(result) == 1
        data = json.loads(result[0].text)

        assert data["symbol"] == "AAPL"
        assert data["data_points"] == 5
        assert data["latest_close"] == 104.5
        assert data["latest_volume"] == 1400000

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_get_stock_data_with_dates(self, mock_fetch):
        """Test stock data retrieval with date range."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=3, freq="D"),
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000000, 1100000, 1200000],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="get_stock_data",
            arguments={"symbol": "MSFT", "start_date": "2024-01-01", "end_date": "2024-01-03"},
        )

        data = json.loads(result[0].text)
        assert data["symbol"] == "MSFT"
        assert "2024-01-01 to 2024-01-03" in data["period"]


class TestCallToolOBV:
    """Tests for calculate_obv tool."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_calculate_obv(self, mock_fetch):
        """Test OBV calculation tool."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
                "Open": [100 + i for i in range(10)],
                "High": [101 + i for i in range(10)],
                "Low": [99 + i for i in range(10)],
                "Close": [100 + i for i in range(10)],
                "Volume": [1000000] * 10,
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_obv", arguments={"symbol": "AAPL", "period": "1mo"}
        )

        data = json.loads(result[0].text)

        assert data["symbol"] == "AAPL"
        assert data["indicator"] == "On-Balance Volume (OBV)"
        assert "latest_obv" in data
        assert "obv_trend" in data
        assert data["obv_trend"] in ["increasing", "decreasing"]


class TestCallToolVWAP:
    """Tests for calculate_vwap tool."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_calculate_vwap(self, mock_fetch):
        """Test VWAP calculation tool."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
                "Open": [100] * 10,
                "High": [102] * 10,
                "Low": [98] * 10,
                "Close": [101] * 10,
                "Volume": [1000000] * 10,
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_vwap", arguments={"symbol": "TSLA", "period": "1mo"}
        )

        data = json.loads(result[0].text)

        assert data["symbol"] == "TSLA"
        assert data["indicator"] == "Volume Weighted Average Price (VWAP)"
        assert "latest_vwap" in data
        assert "latest_close" in data
        assert "price_vs_vwap" in data
        assert "position" in data


class TestCallToolVolumeProfile:
    """Tests for calculate_volume_profile tool."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_calculate_volume_profile(self, mock_fetch):
        """Test volume profile calculation tool."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=20, freq="D"),
                "Open": [100] * 20,
                "High": [105] * 20,
                "Low": [95] * 20,
                "Close": [100] * 20,
                "Volume": [1000000] * 20,
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_volume_profile",
            arguments={"symbol": "NVDA", "period": "1mo", "num_bins": 15},
        )

        data = json.loads(result[0].text)

        assert data["symbol"] == "NVDA"
        assert data["indicator"] == "Volume Profile"
        assert data["num_price_levels"] == 15
        assert "point_of_control" in data
        assert "profile_data" in data
        assert len(data["profile_data"]) == 15


class TestCallToolMFI:
    """Tests for calculate_mfi tool."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_calculate_mfi(self, mock_fetch):
        """Test MFI calculation tool."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=20, freq="D"),
                "Open": [100 + i * 0.5 for i in range(20)],
                "High": [101 + i * 0.5 for i in range(20)],
                "Low": [99 + i * 0.5 for i in range(20)],
                "Close": [100 + i * 0.5 for i in range(20)],
                "Volume": [1000000 + i * 10000 for i in range(20)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_mfi",
            arguments={"symbol": "AMD", "period": "1mo", "mfi_period": 14},
        )

        data = json.loads(result[0].text)

        assert data["symbol"] == "AMD"
        assert "Money Flow Index" in data["indicator"]
        assert "latest_mfi" in data
        assert "condition" in data
        assert data["condition"] in ["Overbought (>80)", "Oversold (<20)", "Neutral (20-80)"]


class TestCallToolVolumeTrends:
    """Tests for analyze_volume_trends tool."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_analyze_volume_trends(self, mock_fetch):
        """Test volume trends analysis tool."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=30, freq="D"),
                "Open": [100] * 30,
                "High": [101] * 30,
                "Low": [99] * 30,
                "Close": [100] * 30,
                "Volume": [1000000 + i * 10000 for i in range(30)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="analyze_volume_trends",
            arguments={"symbol": "INTC", "period": "1mo", "window": 20},
        )

        data = json.loads(result[0].text)

        assert data["symbol"] == "INTC"
        assert data["analysis"] == "Volume Trend Analysis"
        assert "current_volume" in data
        assert "average_volume" in data
        assert "volume_vs_average" in data
        assert "divergence_detected" in data


class TestCallToolComprehensive:
    """Tests for comprehensive_analysis tool."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_comprehensive_analysis(self, mock_fetch):
        """Test comprehensive analysis tool."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=30, freq="D"),
                "Open": [100 + i * 0.5 for i in range(30)],
                "High": [102 + i * 0.5 for i in range(30)],
                "Low": [98 + i * 0.5 for i in range(30)],
                "Close": [101 + i * 0.5 for i in range(30)],
                "Volume": [1000000 + i * 20000 for i in range(30)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="comprehensive_analysis", arguments={"symbol": "SPY", "period": "1mo"}
        )

        data = json.loads(result[0].text)

        assert data["symbol"] == "SPY"
        assert "period" in data
        assert "latest_price" in data
        # Check for new categorized indicators structure
        assert "volume_indicators" in data
        assert "obv" in data["volume_indicators"]
        assert "mfi" in data["volume_indicators"]
        assert "price_indicators" in data
        assert "vwap" in data["price_indicators"]
        assert "volatility_indicators" in data
        assert "volume_profile" in data
        assert "volume_trends" in data
        assert "summary" in data
        assert isinstance(data["summary"], list)


class TestErrorHandling:
    """Tests for error handling in tools."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_invalid_symbol_error(self, mock_fetch):
        """Test handling of invalid symbol."""
        mock_fetch.side_effect = ValueError("No data found for symbol INVALID")

        result = await handle_call_tool(
            name="get_stock_data", arguments={"symbol": "INVALID", "period": "1mo"}
        )

        data = json.loads(result[0].text)
        assert "error" in data
        assert "No data found" in data["error"]

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        """Test handling of unknown tool name."""
        result = await handle_call_tool(name="unknown_tool", arguments={"symbol": "AAPL"})

        data = json.loads(result[0].text)
        assert "error" in data
        assert "Unknown tool" in data["error"]


class TestGenerateSummary:
    """Tests for summary generation."""

    def test_generate_summary_bullish(self):
        """Test summary generation for bullish conditions."""
        mock_data = pd.DataFrame({"Close": [100, 101, 102, 103, 104]})
        mock_obv = pd.Series([0, 1000000, 2000000, 3000000, 4000000])
        mock_vwap = pd.Series([100, 100.5, 101, 101.5, 102])
        mock_mfi = pd.Series([50, 55, 60, 65, 70])
        mock_trends = {"divergence_detected": False}

        summary = generate_summary(mock_data, mock_obv, mock_vwap, mock_mfi, mock_trends, 104, 102)

        assert len(summary) > 0
        assert any("above VWAP" in s for s in summary)
        assert any("increasing" in s.lower() for s in summary)

    def test_generate_summary_with_divergence(self):
        """Test summary generation with divergence."""
        mock_data = pd.DataFrame({"Close": [100, 101, 102, 103, 104]})
        mock_obv = pd.Series([0, 1000000, 2000000, 3000000, 4000000])
        mock_vwap = pd.Series([100, 100.5, 101, 101.5, 102])
        mock_mfi = pd.Series([50, 55, 60, 65, 85])
        mock_trends = {
            "divergence_detected": True,
            "divergence_type": "Price up, Volume down",
        }

        summary = generate_summary(mock_data, mock_obv, mock_vwap, mock_mfi, mock_trends, 104, 102)

        assert any("divergence" in s.lower() for s in summary)
        assert any("overbought" in s.lower() for s in summary)
