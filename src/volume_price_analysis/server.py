"""MCP Server for Volume-Price Analysis."""

import asyncio
import json
import logging
import math
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
from mcp.server import NotificationOptions, Server, ServerRequestContext
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from .analysis import build_headline, run_options_analysis, run_scan
from .data_fetcher import fetch_stock_data
from .indicators import (
    analyze_volume_trends,
    calculate_accumulation_distribution,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_chaikin_money_flow,
    calculate_composite_score,
    calculate_enhanced_volume_profile,
    calculate_historical_volatility,
    calculate_mfi,
    calculate_obv,
    calculate_price_roc,
    calculate_relative_volume,
    calculate_rsi_divergence,
    calculate_volume_profile,
    calculate_vpt,
    calculate_vwap,
    calculate_vwma,
    detect_bollinger_squeeze,
    detect_volume_breakout,
)

logger = logging.getLogger(__name__)

try:
    _SERVER_VERSION = version("volume-price-analysis-mcp")
except PackageNotFoundError:
    _SERVER_VERSION = "0.0.0"


def _sanitize_for_json(obj: object) -> object:
    """Recursively replace NaN/Infinity floats with None for RFC 8259 compliance."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _json_response(result: dict) -> str:
    """Serialize result dict to JSON, converting NaN/Infinity to null."""
    return json.dumps(_sanitize_for_json(result), indent=2, default=str)


async def _handle_scan_candidates(arguments: dict) -> CallToolResult:
    """Handle scan_candidates tool - delegates to analysis.run_scan."""
    holding_period = arguments.get("holding_period", 14)
    _validate_range(holding_period, "holding_period", 1, 90)
    max_results = arguments.get("max_results", 15)
    _validate_range(max_results, "max_results", 1, 100)

    result = await run_scan(
        symbols=arguments.get("symbols", []),
        universe=arguments.get("universe", "full_market"),
        period=arguments.get("period", "3mo"),
        holding_period=holding_period,
        min_score=arguments.get("min_score", 2.0),
        min_adx=arguments.get("min_adx", 20),
        max_iv_percentile=arguments.get("max_iv_percentile", 100),
        direction=arguments.get("direction", "any"),
        max_results=max_results,
    )

    return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])


async def handle_list_tools() -> list[Tool]:
    """List available volume-price analysis tools."""
    return [
        Tool(
            name="get_stock_data",
            description="Fetch historical stock data for a given symbol and time period",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA')",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional if using period)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional if using period)",
                    },
                    "period": {
                        "type": "string",
                        "description": (
                            "Period to fetch if dates not specified "
                            "(e.g., '1mo', '3mo', '6mo', '1y', '5y')"
                        ),
                        "default": "1mo",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="calculate_obv",
            description=(
                "Calculate On-Balance Volume (OBV) - cumulative volume "
                "indicator that adds volume on up days and subtracts on down days"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)",
                    },
                    "period": {
                        "type": "string",
                        "description": "Period if dates not specified (default: '1mo')",
                        "default": "1mo",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="calculate_vwap",
            description=(
                "Calculate Volume Weighted Average Price (VWAP) - average "
                "price weighted by volume, used as a trading benchmark"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)",
                    },
                    "period": {
                        "type": "string",
                        "description": "Period if dates not specified (default: '1mo')",
                        "default": "1mo",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="calculate_volume_profile",
            description=(
                "Calculate Volume Profile - distribution of volume at "
                "different price levels, useful for identifying support/resistance"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)",
                    },
                    "period": {
                        "type": "string",
                        "description": "Period if dates not specified (default: '1mo')",
                        "default": "1mo",
                    },
                    "num_bins": {
                        "type": "integer",
                        "description": "Number of price levels to analyze (default: 20)",
                        "default": 20,
                        "minimum": 2,
                        "maximum": 1000,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="calculate_mfi",
            description=(
                "Calculate Money Flow Index (MFI) - volume-weighted RSI "
                "that oscillates 0-100, >80 overbought, <20 oversold"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)",
                    },
                    "period": {
                        "type": "string",
                        "description": "Period if dates not specified (default: '1mo')",
                        "default": "1mo",
                    },
                    "mfi_period": {
                        "type": "integer",
                        "description": "Lookback period for MFI calculation (default: 14)",
                        "default": 14,
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="calculate_ad_line",
            description=(
                "Calculate Accumulation/Distribution Line (A/D Line) - measures "
                "cumulative flow of money into and out of a security"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)",
                    },
                    "period": {
                        "type": "string",
                        "description": "Period if dates not specified (default: '1mo')",
                        "default": "1mo",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="calculate_cmf",
            description=(
                "Calculate Chaikin Money Flow (CMF) - measures buying and selling "
                "pressure over a set period (ranges -1 to +1). "
                "> 0 indicates buying, < 0 indicates selling"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)",
                    },
                    "period": {
                        "type": "string",
                        "description": "Period if dates not specified (default: '1mo')",
                        "default": "1mo",
                    },
                    "cmf_period": {
                        "type": "integer",
                        "description": "Lookback period for CMF calculation (default: 20)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="analyze_volume_trends",
            description="Analyze volume trends and detect price-volume divergences",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)",
                    },
                    "period": {
                        "type": "string",
                        "description": "Period if dates not specified (default: '1mo')",
                        "default": "1mo",
                    },
                    "window": {
                        "type": "integer",
                        "description": "Rolling window for trend analysis (default: 20)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="comprehensive_analysis",
            description=(
                "Perform comprehensive volume-price analysis including "
                "OBV, VWAP, MFI, and volume trends"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)",
                    },
                    "period": {
                        "type": "string",
                        "description": "Period if dates not specified (default: '1mo')",
                        "default": "1mo",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="options_analysis",
            description=(
                "Perform comprehensive volume-price analysis optimized for "
                "options trading with 14-30 day holding periods. Includes "
                "ADX trend strength, RSI divergence detection, IV percentile, "
                "expected move calculations, and composite signal scoring. "
                "Automatically adapts indicator periods based on holding_period."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)",
                    },
                    "period": {
                        "type": "string",
                        "description": (
                            "Period if dates not specified "
                            "(default: '3mo' for sufficient historical context)"
                        ),
                        "default": "3mo",
                    },
                    "holding_period": {
                        "type": "integer",
                        "description": (
                            "Expected options holding period in days (14-30). "
                            "Indicator periods automatically adjust: "
                            "14 days = fast (7-10 day indicators), "
                            "21 days = medium (10-14 day indicators), "
                            "30 days = standard (14-20 day indicators)"
                        ),
                        "default": 14,
                        "minimum": 1,
                        "maximum": 90,
                    },
                    "days_to_expiration": {
                        "type": "integer",
                        "description": (
                            "Days until options expiration for expected move "
                            "calculation (default: same as holding_period)"
                        ),
                        "minimum": 1,
                        "maximum": 365,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="scan_candidates",
            description=(
                "Scan the market to find the best options trading candidates. "
                "Use 'universe' for market-wide scans: 'full_market' (~550 S&P 500 + ETFs), "
                "'sp500' (~503 constituents via pytickersymbols), 'etfs' (50 ETFs). "
                "Or provide custom 'symbols' list. Returns ranked results with composite scores."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Custom list of ticker symbols. "
                            "Leave empty to use 'universe' parameter instead."
                        ),
                    },
                    "universe": {
                        "type": "string",
                        "description": (
                            "Pre-built symbol universe: 'full_market' (~550, S&P 500 + ETFs), "
                            "'sp500' (~503 constituents), 'etfs' (50). "
                            "Default: 'full_market'. Ignored if symbols provided."
                        ),
                        "default": "full_market",
                    },
                    "period": {
                        "type": "string",
                        "description": "Period for analysis (default: '3mo')",
                        "default": "3mo",
                    },
                    "holding_period": {
                        "type": "integer",
                        "description": "Options holding period in days (14-30)",
                        "default": 14,
                        "minimum": 1,
                        "maximum": 90,
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum |score| to include (default: 2.0)",
                        "default": 2.0,
                    },
                    "min_adx": {
                        "type": "number",
                        "description": "Minimum ADX for trend strength (default: 20)",
                        "default": 20,
                    },
                    "max_iv_percentile": {
                        "type": "number",
                        "description": (
                            "Max volatility percentile (default: 100, use 50 for cheap "
                            "options). Note: this is a historical-volatility (HV) proxy, "
                            "not options-implied volatility; results expose both "
                            "iv_percentile (compat) and hv_percentile."
                        ),
                        "default": 100,
                    },
                    "direction": {
                        "type": "string",
                        "description": "'bullish', 'bearish', or 'any' (default: 'any')",
                        "default": "any",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results per direction (default: 15)",
                        "default": 15,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="calculate_rsi_divergence",
            description=(
                "Detect causal RSI divergence at the latest bar using pivot-based analysis. "
                "Bullish divergence: price makes a lower low while RSI makes a higher low "
                "(potential reversal up). Bearish divergence: price makes a higher high while "
                "RSI makes a lower high (potential reversal down). Uses confirmed swing pivots "
                "only — no lookahead. Default period '3mo' to ensure enough pivots."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)",
                    },
                    "period": {
                        "type": "string",
                        "description": (
                            "Period if dates not specified (default: '3mo'). "
                            "At least 3mo recommended — 1mo rarely holds two confirmed pivots."
                        ),
                        "default": "3mo",
                    },
                    "rsi_period": {
                        "type": "integer",
                        "description": "RSI calculation period (default: 14)",
                        "default": 14,
                        "minimum": 1,
                        "maximum": 200,
                    },
                    "divergence_lookback": {
                        "type": "integer",
                        "description": "Minimum bars for divergence history gate (default: 10)",
                        "default": 10,
                        "minimum": 5,
                        "maximum": 200,
                    },
                },
                "required": ["symbol"],
            },
        ),
    ]


def _validate_range(
    value: int | float, param_name: str, min_val: int | float, max_val: int | float
) -> None:
    """Validate that a parameter value is within the allowed range."""
    if value < min_val or value > max_val:
        msg = f"{param_name} must be between {min_val} and {max_val}, got {value}"
        raise ValueError(msg)


async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool execution requests."""
    logger.info("Tool called: %s", name)
    logger.debug("Tool arguments: %s", arguments)

    try:
        # scan_candidates handles its own data fetching per symbol
        if name == "scan_candidates":
            return await _handle_scan_candidates(arguments)

        # Validate tool name before extracting parameters
        single_symbol_tools = {
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
            "calculate_rsi_divergence",
        }
        if name not in single_symbol_tools:
            raise ValueError(f"Unknown tool: {name}")

        # Extract common parameters for single-symbol tools
        symbol = arguments.get("symbol", "").upper()
        if not symbol.strip():
            raise ValueError("symbol parameter is required")
        start_date = arguments.get("start_date")
        end_date = arguments.get("end_date")

        # Set default period based on tool type
        long_period_tools = {"options_analysis", "calculate_rsi_divergence"}
        default_period = "3mo" if name in long_period_tools else "1mo"
        period = arguments.get("period", default_period)

        # Validate tool-specific integer parameters before fetching data (fail fast)
        if name == "calculate_volume_profile":
            _validate_range(arguments.get("num_bins", 20), "num_bins", 2, 1000)
        elif name == "calculate_mfi":
            _validate_range(arguments.get("mfi_period", 14), "mfi_period", 1, 200)
        elif name == "calculate_cmf":
            _validate_range(arguments.get("cmf_period", 20), "cmf_period", 1, 200)
        elif name == "analyze_volume_trends":
            _validate_range(arguments.get("window", 20), "window", 1, 200)
        elif name == "options_analysis":
            _validate_range(arguments.get("holding_period", 14), "holding_period", 1, 90)
            _validate_range(
                arguments.get("days_to_expiration", arguments.get("holding_period", 14)),
                "days_to_expiration",
                1,
                365,
            )
        elif name == "calculate_rsi_divergence":
            _validate_range(arguments.get("rsi_period", 14), "rsi_period", 1, 200)
            _validate_range(arguments.get("divergence_lookback", 10), "divergence_lookback", 5, 200)

        # Fetch stock data
        data = fetch_stock_data(symbol, start_date, end_date, period)
        logger.debug("Data fetched for %s: %d rows", symbol, len(data))

        if name == "get_stock_data":
            start_dt = data["Date"].iloc[0].strftime("%Y-%m-%d")
            end_dt = data["Date"].iloc[-1].strftime("%Y-%m-%d")
            result = {
                "symbol": symbol,
                "period": f"{start_date} to {end_date}" if start_date and end_date else period,
                "data_points": len(data),
                "date_range": f"{start_dt} to {end_dt}",
                "latest_close": float(data["Close"].iloc[-1]),
                "latest_volume": int(data["Volume"].iloc[-1]),
                "sample_data": data.tail(5).to_dict(orient="records"),
            }

            return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

        elif name == "calculate_obv":
            obv = calculate_obv(data)
            data["OBV"] = obv

            lookback = min(5, len(obv))
            cols = ["Date", "Close", "Volume", "OBV"]
            result = {
                "symbol": symbol,
                "indicator": "On-Balance Volume (OBV)",
                "latest_obv": float(obv.iloc[-1]),
                "obv_trend": "increasing" if obv.iloc[-1] > obv.iloc[-lookback] else "decreasing",
                "data_points": len(obv),
                "recent_values": data[cols].tail(10).to_dict(orient="records"),  # type: ignore[call-overload]
            }

            return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

        elif name == "calculate_vwap":
            vwap = calculate_vwap(data)
            data["VWAP"] = vwap

            latest_close = data["Close"].iloc[-1]
            latest_vwap = vwap.iloc[-1]
            position = "above" if latest_close > latest_vwap else "below"

            result = {
                "symbol": symbol,
                "indicator": "Volume Weighted Average Price (VWAP)",
                "latest_vwap": float(latest_vwap),
                "latest_close": float(latest_close),
                "price_vs_vwap": f"{((latest_close / latest_vwap - 1) * 100):.2f}%",
                "position": f"Price is {position} VWAP",
                "recent_values": data[["Date", "Close", "VWAP"]].tail(10).to_dict(orient="records"),  # type: ignore[call-overload]
            }

            return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

        elif name == "calculate_volume_profile":
            num_bins = arguments.get("num_bins", 20)
            profile = calculate_volume_profile(data, num_bins)

            # Find the price level with highest volume (Point of Control)
            max_volume_idx = profile["volumes"].index(max(profile["volumes"]))
            poc = profile["price_levels"][max_volume_idx]

            price_min = min(profile["price_levels"])
            price_max = max(profile["price_levels"])
            result = {
                "symbol": symbol,
                "indicator": "Volume Profile",
                "num_price_levels": num_bins,
                "point_of_control": float(poc),
                "poc_volume": float(profile["volumes"][max_volume_idx]),
                "price_range": f"${price_min:.2f} - ${price_max:.2f}",
                "profile_data": [
                    {"price_level": float(p), "volume": float(v)}
                    for p, v in zip(profile["price_levels"], profile["volumes"], strict=True)
                ],
            }

            return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

        elif name == "calculate_mfi":
            mfi_period = arguments.get("mfi_period", 14)
            mfi = calculate_mfi(data, mfi_period)
            data["MFI"] = mfi

            latest_mfi = mfi.iloc[-1]

            if latest_mfi > 80:
                condition = "Overbought (>80)"
            elif latest_mfi < 20:
                condition = "Oversold (<20)"
            else:
                condition = "Neutral (20-80)"

            result = {
                "symbol": symbol,
                "indicator": f"Money Flow Index (MFI-{mfi_period})",
                "latest_mfi": float(latest_mfi),
                "condition": condition,
                "recent_values": data[["Date", "Close", "MFI"]].tail(10).to_dict(orient="records"),  # type: ignore[call-overload]
            }

            return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

        elif name == "calculate_ad_line":
            ad_line = calculate_accumulation_distribution(data)
            data["AD_Line"] = ad_line

            data_points = len(ad_line)
            latest_value = ad_line.iloc[-1] if data_points > 0 else None
            latest_ad_line = (
                None if latest_value is None or pd.isna(latest_value) else float(latest_value)
            )

            if data_points <= 1 or latest_ad_line is None:
                ad_trend = "flat"
            else:
                # Compare to an earlier value (up to 5 data points back) to capture recent momentum
                lookback = min(5, data_points)
                past_value = ad_line.iloc[-lookback]

                if latest_value > past_value:
                    ad_trend = "increasing"
                elif latest_value < past_value:
                    ad_trend = "decreasing"
                else:
                    ad_trend = "flat"

            result = {
                "symbol": symbol,
                "indicator": "Accumulation/Distribution Line (A/D Line)",
                "latest_ad_line": latest_ad_line,
                "ad_trend": ad_trend,
                "data_points": data_points,
                "recent_values": data[["Date", "Close", "Volume", "AD_Line"]]
                .tail(10)
                .to_dict(orient="records"),  # type: ignore[call-overload]
            }

            return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

        elif name == "calculate_cmf":
            cmf_period = arguments.get("cmf_period", 20)
            cmf = calculate_chaikin_money_flow(data, cmf_period)
            data["CMF"] = cmf

            # CMF uses a rolling window, so the first (period-1) values are NaN.
            # Extract the last finite value, falling back to None if all values are
            # NaN/inf (NaN from insufficient data, inf from zero rolling volume sum).
            valid_cmf = cmf.dropna()
            latest_valid_cmf = valid_cmf.iloc[-1] if not valid_cmf.empty else None
            if latest_valid_cmf is not None and not math.isfinite(latest_valid_cmf):
                latest_valid_cmf = None

            if latest_valid_cmf is None:
                condition = "Insufficient Data"
                latest_cmf_val = None
            elif latest_valid_cmf > 0:
                condition = "Buying Pressure (>0)"
                latest_cmf_val = float(latest_valid_cmf)
            elif latest_valid_cmf < 0:
                condition = "Selling Pressure (<0)"
                latest_cmf_val = float(latest_valid_cmf)
            else:
                condition = "Neutral (0)"
                latest_cmf_val = float(latest_valid_cmf)

            result = {
                "symbol": symbol,
                "indicator": f"Chaikin Money Flow (CMF-{cmf_period})",
                "latest_cmf": latest_cmf_val,
                "condition": condition,
                "recent_values": data[["Date", "Close", "CMF"]].tail(10).to_dict(orient="records"),  # type: ignore[call-overload]
            }

            return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

        elif name == "analyze_volume_trends":
            window = arguments.get("window", 20)
            trends = analyze_volume_trends(data, window)

            result = {"symbol": symbol, "analysis": "Volume Trend Analysis", **trends}

            return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

        elif name == "comprehensive_analysis":
            # Calculate all volume indicators
            obv = calculate_obv(data)
            vwap = calculate_vwap(data)
            mfi = calculate_mfi(data)
            vpt = calculate_vpt(data)
            trends = analyze_volume_trends(data)
            ad_line = calculate_accumulation_distribution(data)
            cmf = calculate_chaikin_money_flow(data)
            rvol = calculate_relative_volume(data)
            breakout = detect_volume_breakout(data)
            vwma = calculate_vwma(data)
            roc = calculate_price_roc(data)

            # Calculate volatility indicators
            hv = calculate_historical_volatility(data)
            atr = calculate_atr(data)
            bbands = calculate_bollinger_bands(data)

            # Enhanced volume profile with VAH/VAL
            profile = calculate_enhanced_volume_profile(data)

            latest_close = data["Close"].iloc[-1]
            latest_vwap = vwap.iloc[-1]
            latest_vwma = vwma.iloc[-1]
            start_dt = data["Date"].iloc[0].strftime("%Y-%m-%d")
            end_dt = data["Date"].iloc[-1].strftime("%Y-%m-%d")

            # Pre-calculate values for clarity
            lookback = min(5, len(data))
            obv_increasing = obv.iloc[-1] > obv.iloc[-lookback]
            ad_increasing = ad_line.iloc[-1] > ad_line.iloc[-lookback]
            obv_flow = "into" if obv_increasing else "out of"
            ad_action = "buying" if ad_increasing else "selling"
            mfi_val = mfi.iloc[-1]
            cmf_val = cmf.iloc[-1]

            if mfi_val > 80:
                mfi_condition = "Overbought"
            elif mfi_val < 20:
                mfi_condition = "Oversold"
            else:
                mfi_condition = "Neutral"

            if cmf_val > 0.25:
                cmf_condition = "Strong buying"
            elif cmf_val < -0.25:
                cmf_condition = "Strong selling"
            else:
                cmf_condition = "Neutral"

            # Pre-calculate bollinger band values
            bb_upper = bbands["upper"].iloc[-1]
            bb_middle = bbands["middle"].iloc[-1]
            bb_lower = bbands["lower"].iloc[-1]
            bb_pct_b = bbands["percent_b"].iloc[-1]
            bb_bw = bbands["bandwidth"].iloc[-1]
            atr_val = atr.iloc[-1]

            is_squeeze = detect_bollinger_squeeze(data)

            if not pd.isna(atr_val):
                atr_interp = f"Expected daily range: ±${atr_val:.2f}"
            else:
                atr_interp = "N/A"

            # Additive top-line headline (recommendation/score/1-line rationale).
            # The detailed `summary` list below is left untouched.
            headline = build_headline(calculate_composite_score(data))

            result = {
                "symbol": symbol,
                "analysis_type": "Comprehensive Volume-Price Analysis",
                "period": f"{start_dt} to {end_dt}",
                "latest_price": float(latest_close),
                "headline": headline,
                "volume_indicators": {
                    "obv": {
                        "value": float(obv.iloc[-1]),
                        "trend": "increasing" if obv_increasing else "decreasing",
                        "interpretation": f"Money flowing {obv_flow} the security",
                    },
                    "accumulation_distribution": {
                        "value": float(ad_line.iloc[-1]),
                        "trend": "increasing" if ad_increasing else "decreasing",
                        "interpretation": f"Institutional {ad_action} pressure",
                    },
                    "vpt": {
                        "value": float(vpt.iloc[-1]),
                        "trend": (
                            "increasing" if vpt.iloc[-1] > vpt.iloc[-lookback] else "decreasing"
                        ),
                    },
                    "mfi": {"value": float(mfi_val), "condition": mfi_condition},
                    "cmf": {
                        "value": float(cmf_val),
                        "condition": cmf_condition,
                        "interpretation": "Positive = buying pressure, Negative = selling pressure",
                    },
                    # Project scalar fields only: rvol also carries "rvol_series",
                    # a pd.Series that would serialize as a truncated repr string.
                    "relative_volume": {
                        "current_rvol": rvol["current_rvol"],
                        "average_volume": rvol["average_volume"],
                        "current_volume": rvol["current_volume"],
                        "significance": rvol["significance"],
                    },
                    "volume_breakout": breakout,
                },
                "price_indicators": {
                    "vwap": {
                        "value": float(latest_vwap),
                        "price_vs_vwap": f"{((latest_close / latest_vwap - 1) * 100):.2f}%",
                        "position": "above" if latest_close > latest_vwap else "below",
                    },
                    "vwma_20": {
                        "value": float(latest_vwma),
                        "price_vs_vwma": f"{((latest_close / latest_vwma - 1) * 100):.2f}%",
                        "position": "above" if latest_close > latest_vwma else "below",
                    },
                    # Same projection: roc carries "roc_series" (pd.Series).
                    "price_roc": {
                        "current_roc": roc["current_roc"],
                        "direction": roc["direction"],
                        "strength": roc["strength"],
                        "volume_confirmed": roc["volume_confirmed"],
                        "signal": roc["signal"],
                    },
                },
                "volatility_indicators": {
                    "historical_volatility_20d": {
                        "value": float(hv.iloc[-1]) if not pd.isna(hv.iloc[-1]) else 0.0,
                        "annualized": True,
                        "interpretation": "Higher HV = more expensive options",
                    },
                    "atr_14d": {
                        "value": float(atr_val) if not pd.isna(atr_val) else 0.0,
                        "interpretation": atr_interp,
                    },
                    "bollinger_bands": {
                        "upper": float(bb_upper) if not pd.isna(bb_upper) else 0.0,
                        "middle": float(bb_middle) if not pd.isna(bb_middle) else 0.0,
                        "lower": float(bb_lower) if not pd.isna(bb_lower) else 0.0,
                        "percent_b": float(bb_pct_b) if not pd.isna(bb_pct_b) else 0.0,
                        "bandwidth": float(bb_bw) if not pd.isna(bb_bw) else 0.0,
                        "squeeze_status": "Yes" if is_squeeze else "No",
                    },
                },
                "volume_profile": {
                    "point_of_control": profile["poc"],
                    "value_area_high": profile["vah"],
                    "value_area_low": profile["val"],
                    "current_position": profile["position"],
                    "interpretation": profile["interpretation"],
                    "poc_distance": f"{profile['poc_distance_pct']:.2f}%",
                    "vah_distance": f"{profile['vah_distance_pct']:.2f}%",
                    "val_distance": f"{profile['val_distance_pct']:.2f}%",
                },
                "volume_trends": trends,
                "summary": generate_enhanced_summary(
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
                    profile,
                    rvol,
                    breakout,
                ),
            }

            return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

        elif name == "options_analysis":
            holding_period = arguments.get("holding_period", 14)
            days_to_expiration = arguments.get("days_to_expiration", holding_period)

            result = run_options_analysis(
                symbol=symbol,
                data=data,
                holding_period=holding_period,
                days_to_expiration=days_to_expiration,
            )

            return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

        elif name == "calculate_rsi_divergence":
            rsi_period = arguments.get("rsi_period", 14)
            divergence_lookback = arguments.get("divergence_lookback", 10)
            divergence_result = calculate_rsi_divergence(data, rsi_period, divergence_lookback)
            result = {"symbol": symbol, **divergence_result}
            return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

        raise AssertionError(f"Unhandled tool: {name}")  # pragma: no cover

    except ValueError as e:
        logger.warning("Tool %s validation error: %s", name, str(e))
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))],
            is_error=True,
        )
    except Exception as e:
        logger.error("Tool %s failed: %s", name, str(e), exc_info=True)
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps({"error": "An internal error occurred"}, indent=2),
                )
            ],
            is_error=True,
        )


async def _on_list_tools(
    _ctx: ServerRequestContext, _params: PaginatedRequestParams | None
) -> ListToolsResult:
    """MCP v2 list_tools handler wrapping handle_list_tools()."""
    return ListToolsResult(tools=await handle_list_tools())


async def _on_call_tool(
    _ctx: ServerRequestContext, params: CallToolRequestParams
) -> CallToolResult:
    """MCP v2 call_tool handler wrapping handle_call_tool()."""
    return await handle_call_tool(params.name, params.arguments or {})


server = Server(
    "volume-price-analysis",
    version=_SERVER_VERSION,
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
)


def generate_enhanced_summary(
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
    profile,
    rvol,
    breakout,
):
    """Generate enhanced human-readable summary of the analysis."""
    summary = []

    # Price vs VWAP
    if latest_close > latest_vwap:
        summary.append("✓ Price trading above VWAP - Bullish institutional sentiment")
    else:
        summary.append("⚠️  Price trading below VWAP - Bearish institutional sentiment")

    # Volume Flow Analysis
    lookback = min(5, len(data))
    if obv.iloc[-1] > obv.iloc[-lookback] and ad_line.iloc[-1] > ad_line.iloc[-lookback]:
        summary.append("✓ Strong accumulation - Both OBV and A/D Line rising")
    elif obv.iloc[-1] < obv.iloc[-lookback] and ad_line.iloc[-1] < ad_line.iloc[-lookback]:
        summary.append("⚠️  Strong distribution - Both OBV and A/D Line falling")
    else:
        summary.append("⚠️  Mixed volume signals - OBV and A/D Line diverging")

    # Money Flow
    latest_mfi = mfi.iloc[-1]
    latest_cmf = cmf.iloc[-1]
    if latest_mfi > 80 or latest_cmf > 0.25:
        summary.append("⚠️  Overbought conditions detected - Potential reversal risk")
    elif latest_mfi < 20 or latest_cmf < -0.25:
        summary.append("✓ Oversold conditions detected - Potential bounce opportunity")

    # Volatility Assessment
    if not pd.isna(hv.iloc[-1]):
        if hv.iloc[-1] > 0.30:
            hv_pct = f"{hv.iloc[-1]:.1%}"
            summary.append(f"⚠️  High volatility ({hv_pct}) - Options expensive, wider stops needed")
        elif hv.iloc[-1] < 0.15:
            hv_pct = f"{hv.iloc[-1]:.1%}"
            summary.append(f"✓ Low volatility ({hv_pct}) - Potential breakout setup")

    # Bollinger Band Squeeze
    if detect_bollinger_squeeze(data):
        summary.append("✓ Bollinger Band squeeze detected - Breakout likely imminent")

    # Volume Profile Position
    summary.append(f"Volume Profile: {profile['interpretation']}")

    # Relative Volume
    if rvol["current_rvol"] > 2.0:
        rvol_val = rvol["current_rvol"]
        summary.append(
            f"⚠️  Extremely high volume ({rvol_val:.1f}x average) - Major catalyst or news"
        )
    elif rvol["current_rvol"] < 0.5:
        summary.append("⚠️  Very low volume - Moves may be unreliable")

    # Volume Breakout
    if breakout["is_breakout"]:
        direction = breakout["direction"].capitalize()
        summary.append(f"✓ Volume breakout detected - {direction} momentum confirmed")

    # Divergence
    if trends["divergence_detected"]:
        div_type = trends["divergence_type"]
        summary.append(f"⚠️  Price-volume divergence: {div_type} - Trend may be weakening")

    return summary


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="volume-price-analysis",
                server_version=_SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def cli() -> None:
    """Synchronous entry point for the console script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
