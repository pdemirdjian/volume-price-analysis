"""MCP Server for Volume-Price Analysis."""

import asyncio
import json
import logging

import pandas as pd
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

from .analysis import run_options_analysis, run_scan
from .data_fetcher import fetch_stock_data
from .indicators import (
    analyze_volume_trends,
    calculate_accumulation_distribution,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_chaikin_money_flow,
    calculate_enhanced_volume_profile,
    calculate_historical_volatility,
    calculate_mfi,
    calculate_obv,
    calculate_price_roc,
    calculate_relative_volume,
    calculate_volume_profile,
    calculate_vpt,
    calculate_vwap,
    calculate_vwma,
    detect_volume_breakout,
)

logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("volume-price-analysis")


async def _handle_scan_candidates(arguments: dict) -> list[TextContent]:
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

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available volume-price analysis tools."""
    return [
        Tool(
            name="get_stock_data",
            description="Fetch historical stock data for a given symbol and time period",
            inputSchema={
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
            inputSchema={
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
            inputSchema={
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
            inputSchema={
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
            inputSchema={
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
            name="analyze_volume_trends",
            description="Analyze volume trends and detect price-volume divergences",
            inputSchema={
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
            inputSchema={
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
            inputSchema={
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
                "Use 'universe' for market-wide scans: 'full_market' (~200 liquid stocks+ETFs), "
                "'mega_caps' (30 largest), 'tech' (50 tech stocks), 'etfs' (40 ETFs). "
                "Or provide custom 'symbols' list. Returns ranked results with composite scores."
            ),
            inputSchema={
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
                            "Pre-built symbol universe: 'full_market' (~200), "
                            "'mega_caps' (30), 'tech' (50), 'financials' (20), "
                            "'healthcare' (20), 'consumer' (20), 'energy' (20), "
                            "'etfs' (40). Default: 'full_market'. Ignored if symbols provided."
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
                        "description": "Max IV percentile (default: 100, use 50 for cheap options)",
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
    ]


def _validate_range(
    value: int | float, param_name: str, min_val: int | float, max_val: int | float
) -> None:
    """Validate that a parameter value is within the allowed range."""
    if value < min_val or value > max_val:
        msg = f"{param_name} must be between {min_val} and {max_val}, got {value}"
        raise ValueError(msg)


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool execution requests."""
    logger.info("Tool called: %s", name)
    logger.debug("Tool arguments: %s", arguments)

    try:
        # scan_candidates handles its own data fetching per symbol
        if name == "scan_candidates":
            return await _handle_scan_candidates(arguments)

        # Extract common parameters for single-symbol tools
        symbol = arguments.get("symbol", "").upper()
        start_date = arguments.get("start_date")
        end_date = arguments.get("end_date")

        # Set default period based on tool type
        default_period = "3mo" if name == "options_analysis" else "1mo"
        period = arguments.get("period", default_period)

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

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "calculate_obv":
            obv = calculate_obv(data)
            data["OBV"] = obv

            cols = ["Date", "Close", "Volume", "OBV"]
            result = {
                "symbol": symbol,
                "indicator": "On-Balance Volume (OBV)",
                "latest_obv": float(obv.iloc[-1]),
                "obv_trend": "increasing" if obv.iloc[-1] > obv.iloc[-5] else "decreasing",
                "data_points": len(obv),
                "recent_values": data[cols].tail(10).to_dict(orient="records"),  # type: ignore[call-overload]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

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

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "calculate_volume_profile":
            num_bins = arguments.get("num_bins", 20)
            _validate_range(num_bins, "num_bins", 2, 1000)
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

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "calculate_mfi":
            mfi_period = arguments.get("mfi_period", 14)
            _validate_range(mfi_period, "mfi_period", 1, 200)
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

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "analyze_volume_trends":
            window = arguments.get("window", 20)
            _validate_range(window, "window", 1, 200)
            trends = analyze_volume_trends(data, window)

            result = {"symbol": symbol, "analysis": "Volume Trend Analysis", **trends}

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

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
            obv_increasing = obv.iloc[-1] > obv.iloc[-5]
            ad_increasing = ad_line.iloc[-1] > ad_line.iloc[-5]
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

            if not pd.isna(bb_bw):
                is_squeeze = bb_bw < bbands["bandwidth"].iloc[-20:].mean() * 0.7
            else:
                is_squeeze = False

            if not pd.isna(atr_val):
                atr_interp = f"Expected daily range: ±${atr_val:.2f}"
            else:
                atr_interp = "N/A"

            result = {
                "symbol": symbol,
                "analysis_type": "Comprehensive Volume-Price Analysis",
                "period": f"{start_dt} to {end_dt}",
                "latest_price": float(latest_close),
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
                        "trend": "increasing" if vpt.iloc[-1] > vpt.iloc[-5] else "decreasing",
                    },
                    "mfi": {"value": float(mfi_val), "condition": mfi_condition},
                    "cmf": {
                        "value": float(cmf_val),
                        "condition": cmf_condition,
                        "interpretation": "Positive = buying pressure, Negative = selling pressure",
                    },
                    "relative_volume": rvol,
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
                    "price_roc": roc,
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
                    bbands,
                    profile,
                    rvol,
                    breakout,
                ),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "options_analysis":
            holding_period = arguments.get("holding_period", 14)
            _validate_range(holding_period, "holding_period", 1, 90)
            days_to_expiration = arguments.get("days_to_expiration", holding_period)
            _validate_range(days_to_expiration, "days_to_expiration", 1, 365)

            result = run_options_analysis(
                symbol=symbol,
                data=data,
                holding_period=holding_period,
                days_to_expiration=days_to_expiration,
            )

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error("Tool %s failed: %s", name, str(e), exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


def generate_summary(data, obv, vwap, mfi, trends, latest_close, latest_vwap):
    """Generate a human-readable summary of the analysis."""
    summary = []

    # Price vs VWAP
    if latest_close > latest_vwap:
        summary.append("Price is trading above VWAP, indicating bullish sentiment")
    else:
        summary.append("Price is trading below VWAP, indicating bearish sentiment")

    # OBV Trend
    if obv.iloc[-1] > obv.iloc[-5]:
        summary.append("OBV is increasing, suggesting accumulation")
    else:
        summary.append("OBV is decreasing, suggesting distribution")

    # MFI Condition
    latest_mfi = mfi.iloc[-1]
    if latest_mfi > 80:
        summary.append("MFI indicates overbought conditions")
    elif latest_mfi < 20:
        summary.append("MFI indicates oversold conditions")

    # Divergence
    if trends["divergence_detected"]:
        summary.append(f"⚠️  Price-volume divergence detected: {trends['divergence_type']}")

    return summary


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
    bbands,
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
    if obv.iloc[-1] > obv.iloc[-5] and ad_line.iloc[-1] > ad_line.iloc[-5]:
        summary.append("✓ Strong accumulation - Both OBV and A/D Line rising")
    elif obv.iloc[-1] < obv.iloc[-5] and ad_line.iloc[-1] < ad_line.iloc[-5]:
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
    if not pd.isna(bbands["bandwidth"].iloc[-1]):
        if bbands["bandwidth"].iloc[-1] < bbands["bandwidth"].iloc[-20:].mean() * 0.7:
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
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
