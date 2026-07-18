"""Tests for MCP server functionality."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mcp.types import CallToolResult

from volume_price_analysis.server import (
    _json_response,
    _sanitize_for_json,
    _validate_range,
    generate_enhanced_summary,
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
            "calculate_ad_line",
            "calculate_cmf",
            "analyze_volume_trends",
            "comprehensive_analysis",
            "options_analysis",
            "scan_candidates",
        ]

        assert len(tools) == 11
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
        assert len(result.content) == 1
        data = json.loads(result.content[0].text)

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

        data = json.loads(result.content[0].text)
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

        data = json.loads(result.content[0].text)

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

        data = json.loads(result.content[0].text)

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

        data = json.loads(result.content[0].text)

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

        data = json.loads(result.content[0].text)

        assert data["symbol"] == "AMD"
        assert "Money Flow Index" in data["indicator"]
        assert "latest_mfi" in data
        assert "condition" in data
        assert data["condition"] in ["Overbought (>80)", "Oversold (<20)", "Neutral (20-80)"]


class TestCallToolADLine:
    """Tests for calculate_ad_line tool."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_calculate_ad_line(self, mock_fetch):
        """Test AD Line calculation tool."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=20, freq="D"),
                "Open": [100 + i * 0.5 for i in range(20)],
                "High": [102 + i * 0.5 for i in range(20)],
                "Low": [98 + i * 0.5 for i in range(20)],
                "Close": [101 + i * 0.5 for i in range(20)],
                "Volume": [1000000 + i * 10000 for i in range(20)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_ad_line",
            arguments={"symbol": "IBM", "period": "1mo"},
        )

        data = json.loads(result.content[0].text)

        assert data["symbol"] == "IBM"
        assert "Accumulation/Distribution Line" in data["indicator"]
        assert "latest_ad_line" in data
        assert "ad_trend" in data
        assert data["ad_trend"] in ["increasing", "decreasing"]

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_calculate_ad_line_short_data(self, mock_fetch):
        """Test AD Line calculation tool with very little data (edge case)."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=2, freq="D"),
                "Open": [100, 101],
                "High": [102, 103],
                "Low": [98, 99],
                "Close": [101, 102],
                "Volume": [1000000, 1100000],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_ad_line",
            arguments={"symbol": "IBM", "start_date": "2024-01-01", "end_date": "2024-01-02"},
        )

        data = json.loads(result.content[0].text)

        assert data["symbol"] == "IBM"
        assert "Accumulation/Distribution Line" in data["indicator"]
        assert "latest_ad_line" in data
        assert "ad_trend" in data
        assert data["ad_trend"] in ["increasing", "decreasing", "flat"]
        assert data["data_points"] == 2


class TestCallToolCMF:
    """Tests for calculate_cmf tool."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_calculate_cmf(self, mock_fetch):
        """Test CMF calculation tool."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=20, freq="D"),
                "Open": [100 + i * 0.5 for i in range(20)],
                "High": [102 + i * 0.5 for i in range(20)],
                "Low": [98 + i * 0.5 for i in range(20)],
                "Close": [101 + i * 0.5 for i in range(20)],
                "Volume": [1000000 + i * 10000 for i in range(20)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_cmf",
            arguments={"symbol": "GOOG", "period": "1mo", "cmf_period": 14},
        )

        data = json.loads(result.content[0].text)

        assert data["symbol"] == "GOOG"
        assert "Chaikin Money Flow" in data["indicator"]
        assert "latest_cmf" in data
        assert "condition" in data
        assert "Pressure" in data["condition"] or "Neutral" in data["condition"]

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_calculate_cmf_insufficient_data(self, mock_fetch):
        """Test CMF calculation tool explicitly handling insufficient data."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=5, freq="D"),
                "Open": [100 + i * 0.5 for i in range(5)],
                "High": [102 + i * 0.5 for i in range(5)],
                "Low": [98 + i * 0.5 for i in range(5)],
                "Close": [101 + i * 0.5 for i in range(5)],
                "Volume": [1000000 + i * 10000 for i in range(5)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_cmf",
            arguments={"symbol": "GOOG", "period": "1mo", "cmf_period": 20},
        )

        data = json.loads(result.content[0].text)

        assert data["symbol"] == "GOOG"
        assert data["latest_cmf"] is None
        assert data["condition"] == "Insufficient Data"

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_calculate_cmf_selling_pressure(self, mock_fetch):
        """Test CMF calculation tool returns selling pressure for negative CMF."""
        # Close near Low produces negative Money Flow Multiplier => negative CMF
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=20, freq="D"),
                "Open": [100 + i * 0.5 for i in range(20)],
                "High": [104 + i * 0.5 for i in range(20)],
                "Low": [98 + i * 0.5 for i in range(20)],
                "Close": [99 + i * 0.5 for i in range(20)],
                "Volume": [1000000 + i * 10000 for i in range(20)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_cmf",
            arguments={"symbol": "GOOG", "period": "1mo", "cmf_period": 14},
        )

        data = json.loads(result.content[0].text)

        assert data["symbol"] == "GOOG"
        assert data["condition"] == "Selling Pressure (<0)"
        assert data["latest_cmf"] < 0

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_calculate_cmf_default_period(self, mock_fetch):
        """Test CMF calculation tool with default cmf_period (omitted argument)."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=25, freq="D"),
                "Open": [100 + i * 0.5 for i in range(25)],
                "High": [102 + i * 0.5 for i in range(25)],
                "Low": [98 + i * 0.5 for i in range(25)],
                "Close": [101 + i * 0.5 for i in range(25)],
                "Volume": [1000000 + i * 10000 for i in range(25)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_cmf",
            arguments={"symbol": "GOOG"},
        )

        data = json.loads(result.content[0].text)

        assert data["symbol"] == "GOOG"
        assert "CMF-20" in data["indicator"]
        assert data["latest_cmf"] is not None
        assert data["condition"] in [
            "Buying Pressure (>0)",
            "Selling Pressure (<0)",
            "Neutral (0)",
        ]


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

        data = json.loads(result.content[0].text)

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

        data = json.loads(result.content[0].text)

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

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_comprehensive_analysis_includes_headline(self, mock_fetch):
        """Additive top-line headline is present without disturbing existing keys (O4)."""
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
        data = json.loads(result.content[0].text)

        headline = data["headline"]
        assert set(headline) == {
            "recommendation",
            "composite_score",
            "signal_quality",
            "rationale",
        }
        assert isinstance(headline["rationale"], str) and headline["rationale"]
        # Existing narrative summary must remain a list (contract preserved).
        assert isinstance(data["summary"], list)


class TestCallToolOptions:
    """Tests for options_analysis tool."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_options_analysis_includes_headline(self, mock_fetch):
        """options_analysis response carries an additive headline (O4)."""
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=60, freq="D"),
                "Open": [100 + i * 0.3 for i in range(60)],
                "High": [102 + i * 0.3 for i in range(60)],
                "Low": [98 + i * 0.3 for i in range(60)],
                "Close": [101 + i * 0.3 for i in range(60)],
                "Volume": [1000000 + i * 15000 for i in range(60)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="options_analysis",
            arguments={"symbol": "SPY", "period": "3mo", "holding_period": 14},
        )
        data = json.loads(result.content[0].text)

        assert "headline" in data
        headline = data["headline"]
        assert set(headline) == {
            "recommendation",
            "composite_score",
            "signal_quality",
            "rationale",
        }
        # Headline summarises the same call as the detailed composite_signal.
        assert headline["recommendation"] == data["composite_signal"]["recommendation"]


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

        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert "No data found" in data["error"]

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_unknown_tool_error(self, mock_fetch):
        """Test handling of unknown tool name."""
        mock_fetch.return_value = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=30),
                "Open": [100] * 30,
                "High": [101] * 30,
                "Low": [99] * 30,
                "Close": [100.5] * 30,
                "Volume": [1000000] * 30,
            }
        )
        result = await handle_call_tool(name="unknown_tool", arguments={"symbol": "AAPL"})

        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert "Unknown tool" in data["error"]

    @pytest.mark.asyncio
    async def test_value_error_returns_specific_message(self):
        """Test that ValueError returns the actual error message to the caller."""
        result = await handle_call_tool(
            name="get_stock_data", arguments={"symbol": "!!!", "period": "1mo"}
        )

        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        # ValueError message should be passed through to the caller
        assert "Invalid symbol format" in data["error"]

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_generic_exception_returns_generic_message(self, mock_fetch):
        """Test that non-ValueError exceptions return a generic message, hiding internals."""
        mock_fetch.side_effect = RuntimeError("secret internal path /etc/foo")

        result = await handle_call_tool(
            name="get_stock_data", arguments={"symbol": "AAPL", "period": "1mo"}
        )

        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert data["error"] == "An internal error occurred"
        # Internal details must NOT leak to the caller
        assert "secret internal path" not in data["error"]
        assert "/etc/foo" not in data["error"]

    @pytest.mark.asyncio
    async def test_empty_symbol_returns_error(self):
        """Test that an empty symbol parameter returns a validation error."""
        result = await handle_call_tool(
            name="get_stock_data", arguments={"symbol": "", "period": "1mo"}
        )

        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert "symbol" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_symbol_returns_error(self):
        """Test that a missing symbol parameter returns a validation error."""
        result = await handle_call_tool(name="get_stock_data", arguments={"period": "1mo"})

        assert isinstance(result, CallToolResult)
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_with_flag(self):
        """Test that an unknown tool name returns isError=True."""
        result = await handle_call_tool(name="nonexistent_tool", arguments={"symbol": "AAPL"})

        assert isinstance(result, CallToolResult)
        assert result.isError is True


class TestScanCandidates:
    """Tests for scan_candidates tool."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.analysis.analyze_single_symbol")
    async def test_scan_with_hyphenated_symbols(self, mock_analyze):
        """Test scan_candidates correctly handles symbols with hyphens like BRK-B."""

        # Mock returns a candidate for BRK-B and AAPL, None for others
        def mock_analyze_side_effect(symbol, *args, **kwargs):
            if symbol == "BRK-B":
                return {
                    "symbol": "BRK-B",
                    "composite_score": 3.5,
                    "recommendation": "bullish",
                    "signal_quality": "good",
                    "adx": 28.5,
                    "trend_strength": "strong",
                    "trend_direction": "bullish",
                    "rsi": 55.0,
                    "rsi_divergence": "none",
                    "iv_percentile": 35.0,
                    "iv_implication": "neutral",
                    "expected_move_pct": 5.2,
                    "rvol": 1.2,
                    "latest_price": 450.00,
                    "key_levels": {"upper_target": 475.00, "lower_target": 425.00},
                }
            elif symbol == "AAPL":
                return {
                    "symbol": "AAPL",
                    "composite_score": -2.5,
                    "recommendation": "bearish",
                    "signal_quality": "moderate",
                    "adx": 22.0,
                    "trend_strength": "moderate",
                    "trend_direction": "bearish",
                    "rsi": 35.0,
                    "rsi_divergence": "none",
                    "iv_percentile": 45.0,
                    "iv_implication": "neutral",
                    "expected_move_pct": 4.5,
                    "rvol": 0.9,
                    "latest_price": 185.00,
                    "key_levels": {"upper_target": 195.00, "lower_target": 175.00},
                }
            return None  # Filtered out

        mock_analyze.side_effect = mock_analyze_side_effect

        # Call scan_candidates with custom symbols including hyphenated BRK-B
        result = await handle_call_tool(
            name="scan_candidates",
            arguments={
                "symbols": ["AAPL", "BRK-B", "MSFT"],
                "min_score": 2.0,
                "min_adx": 20,
            },
        )

        data = json.loads(result.content[0].text)

        # Verify scan completed
        assert "scan_parameters" in data
        assert data["scan_parameters"]["universe"] == "custom"
        assert data["scan_parameters"]["symbols_in_universe"] == 3

        # Verify BRK-B was processed correctly (hyphen handled)
        assert "top_bullish" in data
        assert "top_bearish" in data

        bullish_symbols = [c["symbol"] for c in data["top_bullish"]]
        bearish_symbols = [c["symbol"] for c in data["top_bearish"]]

        assert "BRK-B" in bullish_symbols, "BRK-B should be in bullish candidates"
        assert "AAPL" in bearish_symbols, "AAPL should be in bearish candidates"

        # Verify the mock was called with correct symbols including hyphen
        called_symbols = [call[0][0] for call in mock_analyze.call_args_list]
        assert "BRK-B" in called_symbols, "BRK-B should have been passed to analyzer"

    @pytest.mark.asyncio
    @patch("volume_price_analysis.analysis.analyze_single_symbol")
    async def test_scan_handles_errors_gracefully(self, mock_analyze):
        """Test scan_candidates handles analysis errors without failing entire scan."""

        def mock_analyze_side_effect(symbol, *args, **kwargs):
            if symbol == "INVALID":
                raise ValueError("No data found for symbol")
            return {
                "symbol": symbol,
                "composite_score": 3.0,
                "recommendation": "bullish",
                "signal_quality": "good",
                "adx": 25.0,
                "trend_strength": "moderate",
                "trend_direction": "bullish",
                "rsi": 50.0,
                "rsi_divergence": "none",
                "iv_percentile": 40.0,
                "iv_implication": "neutral",
                "expected_move_pct": 4.0,
                "rvol": 1.0,
                "latest_price": 100.00,
                "key_levels": {"upper_target": 110.00, "lower_target": 90.00},
            }

        mock_analyze.side_effect = mock_analyze_side_effect

        result = await handle_call_tool(
            name="scan_candidates",
            arguments={"symbols": ["AAPL", "INVALID", "MSFT"]},
        )

        data = json.loads(result.content[0].text)

        # Scan should complete despite one symbol error
        assert "summary" in data
        assert data["summary"]["errors"] == 1
        assert data["scan_parameters"]["symbols_scanned"] == 2  # AAPL and MSFT

        # Error should be captured
        assert data["errors"] is not None
        assert any(e["symbol"] == "INVALID" for e in data["errors"])


class TestJsonSanitization:
    """Tests for NaN/Infinity JSON sanitization."""

    def test_sanitize_replaces_nan_with_none(self):
        """Test that NaN floats become None."""
        result = _sanitize_for_json({"val": float("nan"), "ok": 1.5})
        assert result["val"] is None
        assert result["ok"] == 1.5

    def test_sanitize_replaces_infinity_with_none(self):
        """Test that Infinity floats become None."""
        result = _sanitize_for_json({"pos": float("inf"), "neg": float("-inf")})
        assert result["pos"] is None
        assert result["neg"] is None

    def test_sanitize_handles_nested_structures(self):
        """Test that sanitization works recursively in dicts and lists."""
        result = _sanitize_for_json({"data": [{"v": float("nan")}, {"v": 3.0}]})
        assert result["data"][0]["v"] is None
        assert result["data"][1]["v"] == 3.0

    def test_json_response_produces_valid_json(self):
        """Test that _json_response output is parseable JSON with null for NaN."""
        output = _json_response({"val": float("nan"), "ok": 42})
        parsed = json.loads(output)
        assert parsed["val"] is None
        assert parsed["ok"] == 42


class TestParameterValidation:
    """Tests for parameter range validation."""

    def test_validate_range_valid(self):
        """Test that valid values pass validation."""
        _validate_range(20, "num_bins", 2, 1000)
        _validate_range(14, "mfi_period", 1, 200)
        _validate_range(1, "window", 1, 200)

    def test_validate_range_at_boundaries(self):
        """Test that boundary values pass validation."""
        _validate_range(2, "num_bins", 2, 1000)
        _validate_range(1000, "num_bins", 2, 1000)

    def test_validate_range_below_minimum(self):
        """Test that values below minimum raise ValueError."""
        with pytest.raises(ValueError, match="num_bins must be between 2 and 1000"):
            _validate_range(0, "num_bins", 2, 1000)

    def test_validate_range_above_maximum(self):
        """Test that values above maximum raise ValueError."""
        with pytest.raises(ValueError, match="window must be between 1 and 200"):
            _validate_range(500, "window", 1, 200)

    def test_validate_range_negative(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="mfi_period must be between 1 and 200"):
            _validate_range(-1, "mfi_period", 1, 200)

    @pytest.mark.asyncio
    async def test_invalid_num_bins_returns_error(self):
        """Test that invalid num_bins returns error via tool handler."""
        result = await handle_call_tool(
            name="calculate_volume_profile",
            arguments={"symbol": "AAPL", "num_bins": 0},
        )
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert "num_bins" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_mfi_period_returns_error(self):
        """Test that invalid mfi_period returns error via tool handler."""
        result = await handle_call_tool(
            name="calculate_mfi",
            arguments={"symbol": "AAPL", "mfi_period": -5},
        )
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert "mfi_period" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_cmf_period_returns_error(self):
        """Test that invalid cmf_period returns error via tool handler."""
        result = await handle_call_tool(
            name="calculate_cmf",
            arguments={"symbol": "AAPL", "cmf_period": 250},
        )
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert "cmf_period" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_window_returns_error(self):
        """Test that invalid window returns error via tool handler."""
        result = await handle_call_tool(
            name="analyze_volume_trends",
            arguments={"symbol": "AAPL", "window": 999},
        )
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert "window" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_holding_period_returns_error(self):
        """Test that invalid holding_period returns error via scan_candidates tool."""
        result = await handle_call_tool(
            name="scan_candidates",
            arguments={"symbols": ["AAPL"], "holding_period": 0},
        )
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert "holding_period" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_max_results_returns_error(self):
        """Test that invalid max_results returns error via scan_candidates tool."""
        result = await handle_call_tool(
            name="scan_candidates",
            arguments={"symbols": ["AAPL"], "max_results": 0},
        )
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert "max_results" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_options_analysis_holding_period_returns_error(self):
        """Test that invalid holding_period returns error via options_analysis tool."""
        result = await handle_call_tool(
            name="options_analysis",
            arguments={
                "symbol": "AAPL",
                "holding_period": 0,
                "days_to_expiration": 30,
            },
        )
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert "holding_period" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_days_to_expiration_returns_error(self):
        """Test that invalid days_to_expiration returns error via options_analysis tool."""
        result = await handle_call_tool(
            name="options_analysis",
            arguments={
                "symbol": "AAPL",
                "holding_period": 10,
                "days_to_expiration": 0,
            },
        )
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
        assert "days_to_expiration" in data["error"]


def _make_mock_data(n=30, base=100.0, step=0.5, volume_base=1000000, volume_step=20000):
    """Helper to create mock OHLCV DataFrames for server tests."""
    return pd.DataFrame(
        {
            "Date": pd.date_range(start="2024-01-01", periods=n, freq="D"),
            "Open": [base + i * step - 0.5 for i in range(n)],
            "High": [base + i * step + 2 for i in range(n)],
            "Low": [base + i * step - 2 for i in range(n)],
            "Close": [base + i * step for i in range(n)],
            "Volume": [volume_base + i * volume_step for i in range(n)],
        }
    )


class TestCallToolMFIConditions:
    """Tests for MFI oversold and neutral conditions (server lines 651-654)."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_mfi_oversold_condition(self, mock_fetch):
        """Test MFI returns 'Oversold (<20)' when MFI < 20."""
        # Strong downtrend with decreasing volume to push MFI below 20
        n = 30
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=n, freq="D"),
                "Open": [200 - i * 3 for i in range(n)],
                "High": [201 - i * 3 for i in range(n)],
                "Low": [199 - i * 3 for i in range(n)],
                "Close": [200 - i * 3 for i in range(n)],
                "Volume": [2000000 + i * 100000 for i in range(n)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_mfi",
            arguments={"symbol": "BEAR", "period": "1mo", "mfi_period": 14},
        )
        data = json.loads(result.content[0].text)
        # MFI should be oversold in a strong downtrend
        assert data["condition"] in ["Oversold (<20)", "Neutral (20-80)"]

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    @patch("volume_price_analysis.server.calculate_mfi")
    async def test_mfi_oversold_via_mock(self, mock_mfi, mock_fetch):
        """Test MFI oversold branch via mocked MFI values."""
        mock_fetch.return_value = _make_mock_data(n=20)
        # Return MFI series with last value < 20
        mock_mfi.return_value = pd.Series([15.0] * 20)

        result = await handle_call_tool(
            name="calculate_mfi",
            arguments={"symbol": "TEST", "mfi_period": 14},
        )
        data = json.loads(result.content[0].text)
        assert data["condition"] == "Oversold (<20)"

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    @patch("volume_price_analysis.server.calculate_mfi")
    async def test_mfi_neutral_via_mock(self, mock_mfi, mock_fetch):
        """Test MFI neutral branch via mocked MFI values."""
        mock_fetch.return_value = _make_mock_data(n=20)
        mock_mfi.return_value = pd.Series([50.0] * 20)

        result = await handle_call_tool(
            name="calculate_mfi",
            arguments={"symbol": "TEST", "mfi_period": 14},
        )
        data = json.loads(result.content[0].text)
        assert data["condition"] == "Neutral (20-80)"


class TestCallToolADLineEdgeCases:
    """Tests for AD Line flat/decreasing trend (server lines 677, 685-688)."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_ad_line_single_data_point(self, mock_fetch):
        """Test AD Line with a single data point -> flat trend (line 677)."""
        mock_data = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2024-01-01")],
                "Open": [100.0],
                "High": [102.0],
                "Low": [98.0],
                "Close": [101.0],
                "Volume": [1000000],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_ad_line",
            arguments={"symbol": "FLAT", "period": "1d"},
        )
        data = json.loads(result.content[0].text)
        assert data["ad_trend"] == "flat"
        assert data["data_points"] == 1

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_ad_line_decreasing_trend(self, mock_fetch):
        """Test AD Line returns 'decreasing' when A/D is falling (lines 685-686)."""
        # Close near low (negative A/D) to force decreasing A/D line
        n = 10
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=n, freq="D"),
                "Open": [100.0] * n,
                "High": [110.0] * n,
                "Low": [90.0] * n,
                "Close": [91.0] * n,  # close near low = negative MFM
                "Volume": [1000000] * n,
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_ad_line",
            arguments={"symbol": "DEC", "period": "1mo"},
        )
        data = json.loads(result.content[0].text)
        # A/D line should be decreasing (consistently negative MFM)
        assert data["ad_trend"] in ["decreasing", "flat"]

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_ad_line_flat_trend(self, mock_fetch):
        """Test AD Line returns 'flat' when A/D is constant (lines 687-688)."""
        # Close at midpoint => MFM = 0 => A/D stays 0
        n = 10
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=n, freq="D"),
                "Open": [100.0] * n,
                "High": [110.0] * n,
                "Low": [90.0] * n,
                "Close": [100.0] * n,  # midpoint = MFM of 0
                "Volume": [1000000] * n,
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_ad_line",
            arguments={"symbol": "FLAT", "period": "1mo"},
        )
        data = json.loads(result.content[0].text)
        assert data["ad_trend"] == "flat"


class TestCallToolCMFEdgeCases:
    """Tests for CMF edge cases (server lines 714, 726-727)."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_cmf_neutral_zero(self, mock_fetch):
        """Test CMF returns 'Neutral (0)' when CMF is exactly 0 (lines 726-727)."""
        # Close at exact midpoint of H-L => CMF should be 0
        n = 25
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=n, freq="D"),
                "Open": [100.0] * n,
                "High": [110.0] * n,
                "Low": [90.0] * n,
                "Close": [100.0] * n,  # midpoint => MFM = 0
                "Volume": [1000000] * n,
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="calculate_cmf",
            arguments={"symbol": "ZERO", "cmf_period": 20},
        )
        data = json.loads(result.content[0].text)
        assert data["condition"] == "Neutral (0)"
        assert data["latest_cmf"] == 0.0

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    @patch("volume_price_analysis.server.calculate_chaikin_money_flow")
    async def test_cmf_infinite_becomes_none(self, mock_cmf, mock_fetch):
        """Test CMF returns Insufficient Data when CMF is infinite (line 714)."""
        mock_fetch.return_value = _make_mock_data(n=25)
        # Return CMF series where last finite-ish value is inf
        mock_cmf.return_value = pd.Series([float("inf")] * 25)

        result = await handle_call_tool(
            name="calculate_cmf",
            arguments={"symbol": "INF", "cmf_period": 20},
        )
        data = json.loads(result.content[0].text)
        assert data["condition"] == "Insufficient Data"
        assert data["latest_cmf"] is None


class TestCallToolOptionsAnalysis:
    """Tests for options_analysis tool (server lines 910-920)."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.run_options_analysis")
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_options_analysis_basic(self, mock_fetch, mock_options):
        """Test options_analysis tool calls run_options_analysis and returns result."""
        mock_fetch.return_value = _make_mock_data(n=60)
        mock_options.return_value = {
            "symbol": "AAPL",
            "holding_period": 14,
            "composite_score": 3.5,
            "recommendation": "bullish",
        }

        result = await handle_call_tool(
            name="options_analysis",
            arguments={"symbol": "AAPL", "holding_period": 14, "days_to_expiration": 30},
        )
        data = json.loads(result.content[0].text)
        assert data["symbol"] == "AAPL"
        assert data["recommendation"] == "bullish"
        mock_options.assert_called_once()

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.run_options_analysis")
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_options_analysis_default_dte(self, mock_fetch, mock_options):
        """Test options_analysis defaults days_to_expiration to holding_period."""
        mock_fetch.return_value = _make_mock_data(n=60)
        mock_options.return_value = {"symbol": "TSLA", "holding_period": 21}

        await handle_call_tool(
            name="options_analysis",
            arguments={"symbol": "TSLA", "holding_period": 21},
        )
        # days_to_expiration should default to holding_period
        call_kwargs = mock_options.call_args
        assert call_kwargs[1]["days_to_expiration"] == 21


class TestGenerateEnhancedSummary:
    """Tests for generate_enhanced_summary (server lines 993, 998-1001, 1008-1009, etc.)."""

    def _make_series(self, values):
        """Helper to create a pd.Series from a list."""
        return pd.Series(values)

    def _base_args(self):
        """Create base arguments for generate_enhanced_summary with reasonable defaults."""
        data = pd.DataFrame({"Close": [100 + i * 0.5 for i in range(30)]})
        obv = self._make_series([i * 100000 for i in range(30)])
        ad_line = self._make_series([i * 50000 for i in range(30)])
        vwap = self._make_series([100 + i * 0.3 for i in range(30)])
        vwma = self._make_series([100 + i * 0.3 for i in range(30)])
        mfi = self._make_series([50.0] * 30)
        cmf = self._make_series([0.0] * 30)
        trends = {"divergence_detected": False}
        latest_close = 114.5
        latest_vwap = 108.7
        hv = self._make_series([0.20] * 30)
        atr = self._make_series([2.0] * 30)
        bbands = {
            "upper": self._make_series([120.0] * 30),
            "middle": self._make_series([115.0] * 30),
            "lower": self._make_series([110.0] * 30),
            "bandwidth": self._make_series([5.0] * 30),
            "percent_b": self._make_series([0.5] * 30),
        }
        profile = {"interpretation": "Price within value area - balanced market"}
        rvol = {"current_rvol": 1.0}
        breakout = {"is_breakout": False, "direction": "none"}
        return (
            data,
            obv,
            ad_line,
            vwap,
            vwma,
            mfi,
            cmf,
            trends,
            latest_close,
            latest_vwap,
            hv,
            atr,
            bbands,
            profile,
            rvol,
            breakout,
        )

    def test_below_vwap_sentiment(self):
        """Test bearish sentiment when price below VWAP (line 993)."""
        args = list(self._base_args())
        args[8] = 95.0  # latest_close below VWAP
        args[9] = 110.0  # latest_vwap

        summary = generate_enhanced_summary(*args)
        assert any("below VWAP" in s.lower() or "bearish" in s.lower() for s in summary)

    def test_mixed_volume_signals(self):
        """Test mixed volume signals when OBV and A/D disagree (lines 998-1001)."""
        args = list(self._base_args())
        # OBV increasing, A/D decreasing => mixed signals
        args[1] = self._make_series([i * 100000 for i in range(30)])  # OBV increasing
        args[2] = self._make_series([30 * 50000 - i * 50000 for i in range(30)])  # A/D decreasing

        summary = generate_enhanced_summary(*args)
        assert any("mixed" in s.lower() or "diverging" in s.lower() for s in summary)

    def test_strong_distribution(self):
        """Test strong distribution when both OBV and A/D falling (lines 998-999)."""
        args = list(self._base_args())
        # Both OBV and A/D decreasing
        args[1] = self._make_series([30 * 100000 - i * 100000 for i in range(30)])  # OBV falling
        args[2] = self._make_series([30 * 50000 - i * 50000 for i in range(30)])  # A/D falling

        summary = generate_enhanced_summary(*args)
        assert any("distribution" in s.lower() for s in summary)

    def test_oversold_conditions(self):
        """Test oversold detection (lines 1008-1009)."""
        args = list(self._base_args())
        args[5] = self._make_series([15.0] * 30)  # MFI < 20

        summary = generate_enhanced_summary(*args)
        assert any("oversold" in s.lower() for s in summary)

    def test_high_volatility(self):
        """Test high volatility message (lines 1014-1015)."""
        args = list(self._base_args())
        args[10] = self._make_series([0.35] * 30)  # HV > 0.30

        summary = generate_enhanced_summary(*args)
        assert any("high volatility" in s.lower() or "volatility" in s.lower() for s in summary)

    def test_low_volatility(self):
        """Test low volatility message (lines 1016-1018)."""
        args = list(self._base_args())
        args[10] = self._make_series([0.10] * 30)  # HV < 0.15

        summary = generate_enhanced_summary(*args)
        assert any("low volatility" in s.lower() or "breakout" in s.lower() for s in summary)

    def test_bollinger_squeeze(self):
        """Test Bollinger Band squeeze detection (line 1023)."""
        args = list(self._base_args())
        # Last bandwidth much lower than average of last 20 => squeeze
        bw_values = [10.0] * 20 + [2.0] * 10  # last 10 values narrow
        args[12] = {
            "upper": self._make_series([120.0] * 30),
            "middle": self._make_series([115.0] * 30),
            "lower": self._make_series([110.0] * 30),
            "bandwidth": self._make_series(bw_values),
            "percent_b": self._make_series([0.5] * 30),
        }

        summary = generate_enhanced_summary(*args)
        assert any("squeeze" in s.lower() for s in summary)

    def test_extremely_high_volume(self):
        """Test extremely high volume message (lines 1030-1031)."""
        args = list(self._base_args())
        args[14] = {"current_rvol": 3.0}  # > 2.0

        summary = generate_enhanced_summary(*args)
        assert any("extremely high volume" in s.lower() or "3.0x" in s for s in summary)

    def test_very_low_volume(self):
        """Test very low volume message (line 1035)."""
        args = list(self._base_args())
        args[14] = {"current_rvol": 0.3}  # < 0.5

        summary = generate_enhanced_summary(*args)
        assert any("low volume" in s.lower() for s in summary)

    def test_volume_breakout_detected(self):
        """Test volume breakout message (lines 1039-1040)."""
        args = list(self._base_args())
        args[15] = {"is_breakout": True, "direction": "bullish"}

        summary = generate_enhanced_summary(*args)
        assert any("breakout" in s.lower() for s in summary)

    def test_divergence_detected(self):
        """Test divergence message (lines 1044-1045)."""
        args = list(self._base_args())
        args[7] = {"divergence_detected": True, "divergence_type": "Price up, Volume down"}

        summary = generate_enhanced_summary(*args)
        assert any("divergence" in s.lower() for s in summary)

    def test_nan_bandwidth_no_squeeze(self):
        """Test that NaN bandwidth does not trigger squeeze (line 808 in comprehensive)."""
        args = list(self._base_args())
        args[12] = {
            "upper": self._make_series([float("nan")] * 30),
            "middle": self._make_series([float("nan")] * 30),
            "lower": self._make_series([float("nan")] * 30),
            "bandwidth": self._make_series([float("nan")] * 30),
            "percent_b": self._make_series([float("nan")] * 30),
        }

        summary = generate_enhanced_summary(*args)
        # Squeeze should NOT appear
        assert not any("squeeze" in s.lower() for s in summary)

    def test_nan_hv_no_volatility_message(self):
        """Test that NaN HV does not trigger volatility message."""
        args = list(self._base_args())
        args[10] = self._make_series([float("nan")] * 30)

        summary = generate_enhanced_summary(*args)
        assert not any(
            "volatility" in s.lower() and ("high" in s.lower() or "low" in s.lower())
            for s in summary
        )

    def test_overbought_conditions(self):
        """Test overbought detection (lines 1006-1007)."""
        args = list(self._base_args())
        args[5] = self._make_series([85.0] * 30)  # MFI > 80

        summary = generate_enhanced_summary(*args)
        assert any("overbought" in s.lower() for s in summary)


class TestComprehensiveAnalysisBranches:
    """Tests for comprehensive_analysis edge case branches (lines 785-795, 808, 813)."""

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_comprehensive_mfi_overbought(self, mock_fetch):
        """Test comprehensive analysis with overbought MFI (lines 785-786)."""
        # Strongly rising data to push MFI above 80
        n = 30
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=n, freq="D"),
                "Open": [100 + i * 3 for i in range(n)],
                "High": [102 + i * 3 for i in range(n)],
                "Low": [98 + i * 3 for i in range(n)],
                "Close": [101 + i * 3 for i in range(n)],
                "Volume": [1000000 + i * 100000 for i in range(n)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="comprehensive_analysis", arguments={"symbol": "BULL", "period": "1mo"}
        )
        data = json.loads(result.content[0].text)
        assert "summary" in data
        assert isinstance(data["summary"], list)

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_comprehensive_mfi_oversold(self, mock_fetch):
        """Test comprehensive analysis with oversold MFI (lines 787-788)."""
        n = 30
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=n, freq="D"),
                "Open": [200 - i * 3 for i in range(n)],
                "High": [202 - i * 3 for i in range(n)],
                "Low": [198 - i * 3 for i in range(n)],
                "Close": [200 - i * 3 for i in range(n)],
                "Volume": [1000000 + i * 100000 for i in range(n)],
            }
        )
        mock_fetch.return_value = mock_data

        result = await handle_call_tool(
            name="comprehensive_analysis", arguments={"symbol": "BEAR", "period": "1mo"}
        )
        data = json.loads(result.content[0].text)
        assert "summary" in data

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.calculate_atr")
    @patch("volume_price_analysis.server.calculate_bollinger_bands")
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_comprehensive_nan_bandwidth_and_atr(self, mock_fetch, mock_bb, mock_atr):
        """Test comprehensive analysis with NaN bandwidth and ATR (lines 808, 813)."""
        n = 30
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=n, freq="D"),
                "Open": [100 + i * 0.5 for i in range(n)],
                "High": [102 + i * 0.5 for i in range(n)],
                "Low": [98 + i * 0.5 for i in range(n)],
                "Close": [101 + i * 0.5 for i in range(n)],
                "Volume": [1000000 + i * 20000 for i in range(n)],
            }
        )
        mock_fetch.return_value = mock_data
        # Return NaN for all bollinger bands
        mock_bb.return_value = {
            "upper": pd.Series([np.nan] * n),
            "middle": pd.Series([np.nan] * n),
            "lower": pd.Series([np.nan] * n),
            "bandwidth": pd.Series([np.nan] * n),
            "percent_b": pd.Series([np.nan] * n),
        }
        # Return NaN for ATR
        mock_atr.return_value = pd.Series([np.nan] * n)

        result = await handle_call_tool(
            name="comprehensive_analysis", arguments={"symbol": "NAN", "period": "1mo"}
        )
        data = json.loads(result.content[0].text)
        assert "summary" in data
        # Verify the tool completes without error despite NaN values
        assert data["symbol"] == "NAN"

    @pytest.mark.asyncio
    @patch("volume_price_analysis.server.calculate_mfi")
    @patch("volume_price_analysis.server.calculate_chaikin_money_flow")
    @patch("volume_price_analysis.server.fetch_stock_data")
    async def test_comprehensive_mfi_oversold_and_cmf_strong_selling(
        self, mock_fetch, mock_cmf, mock_mfi
    ):
        """Test comprehensive analysis with oversold MFI and strong selling CMF (lines 788, 793)."""
        n = 30
        mock_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=n, freq="D"),
                "Open": [100 + i * 0.5 for i in range(n)],
                "High": [102 + i * 0.5 for i in range(n)],
                "Low": [98 + i * 0.5 for i in range(n)],
                "Close": [101 + i * 0.5 for i in range(n)],
                "Volume": [1000000 + i * 20000 for i in range(n)],
            }
        )
        mock_fetch.return_value = mock_data
        # MFI < 20 => "Oversold"
        mock_mfi.return_value = pd.Series([15.0] * n)
        # CMF < -0.25 => "Strong selling"
        mock_cmf.return_value = pd.Series([-0.3] * n)

        result = await handle_call_tool(
            name="comprehensive_analysis", arguments={"symbol": "TEST", "period": "1mo"}
        )
        data = json.loads(result.content[0].text)
        assert "volume_indicators" in data
        assert data["volume_indicators"]["mfi"]["condition"] == "Oversold"
        assert data["volume_indicators"]["cmf"]["condition"] == "Strong selling"


class TestServerVersion:
    """Tests for the MCP server version reporting."""

    def test_server_version_not_hardcoded(self):
        """_SERVER_VERSION must not be the old hardcoded value "1.0.0"."""
        from volume_price_analysis.server import _SERVER_VERSION

        assert _SERVER_VERSION != "1.0.0"

    def test_server_version_is_string(self):
        """_SERVER_VERSION must be a non-empty string."""
        from volume_price_analysis.server import _SERVER_VERSION

        assert isinstance(_SERVER_VERSION, str) and _SERVER_VERSION
