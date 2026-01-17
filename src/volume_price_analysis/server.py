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

from .data_fetcher import fetch_stock_data
from .indicators import (
    analyze_volume_trends,
    # Advanced volume indicators
    calculate_accumulation_distribution,
    # New indicators for enhanced options analysis
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_chaikin_money_flow,
    calculate_composite_score,
    calculate_enhanced_volume_profile,
    calculate_expected_move,
    # Volatility indicators
    calculate_historical_volatility,
    calculate_iv_percentile,
    calculate_mfi,
    calculate_obv,
    calculate_price_roc,
    calculate_relative_volume,
    calculate_rsi_with_divergence,
    calculate_volume_profile,
    calculate_vpt,
    calculate_vwap,
    calculate_vwma,
    detect_volume_breakout,
)

logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("volume-price-analysis")

# Concurrency limit for parallel scanning
MAX_CONCURRENT_SCANS = 10

# Minimum data points required for meaningful Bollinger Band squeeze detection
MIN_SQUEEZE_DETECTION_PERIODS = 5


def _analyze_single_symbol(
    symbol: str,
    period: str,
    holding_period: int,
    min_score: float,
    min_adx: float,
    max_iv: float,
    direction: str,
) -> dict | None:
    """
    Analyze a single symbol for scan_candidates.

    Returns a candidate dict if it passes filters, None otherwise.
    Raises exception on error.
    """
    sym_data = fetch_stock_data(symbol, None, None, period)
    if len(sym_data) < 30:
        return None

    # Calculate composite score and key indicators
    composite = calculate_composite_score(sym_data, holding_period)
    adx_data = calculate_adx(sym_data, 14)
    iv_pct_data = calculate_iv_percentile(sym_data, 20)
    expected_move = calculate_expected_move(sym_data, holding_period, 20)
    rsi_data = calculate_rsi_with_divergence(sym_data, 14, 10)
    rvol = calculate_relative_volume(sym_data, 20)

    score = composite["composite_score"]
    adx = adx_data["adx"]
    iv_pct = iv_pct_data["iv_percentile"]

    # Apply filters
    passes_score = abs(score) >= min_score
    passes_adx = adx >= min_adx
    passes_iv = iv_pct <= max_iv

    if direction == "bullish":
        passes_direction = score > 0
    elif direction == "bearish":
        passes_direction = score < 0
    else:
        passes_direction = True

    if not (passes_score and passes_adx and passes_direction and passes_iv):
        return None

    return {
        "symbol": symbol,
        "composite_score": round(score, 2),
        "recommendation": composite["recommendation"],
        "signal_quality": composite["signal_quality"],
        "adx": round(adx, 1),
        "trend_strength": adx_data["trend_strength"],
        "trend_direction": adx_data["trend_direction"],
        "rsi": round(rsi_data["rsi"], 1),
        "rsi_divergence": rsi_data["divergence_type"],
        "iv_percentile": round(iv_pct, 1),
        "iv_implication": iv_pct_data["options_implication"],
        "expected_move_pct": round(expected_move["expected_move_percent"], 2),
        "rvol": round(rvol["current_rvol"], 2),
        "latest_price": round(float(sym_data["Close"].iloc[-1]), 2),
        "key_levels": {
            "upper_target": round(expected_move["upper_target_1std"], 2),
            "lower_target": round(expected_move["lower_target_1std"], 2),
        },
    }


async def _analyze_symbol_async(
    symbol: str,
    period: str,
    holding_period: int,
    min_score: float,
    min_adx: float,
    max_iv: float,
    direction: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict | None, str | None]:
    """
    Async wrapper for symbol analysis with concurrency limiting.

    Returns (symbol, candidate_or_none, error_or_none).
    """
    async with semaphore:
        try:
            result = await asyncio.to_thread(
                _analyze_single_symbol,
                symbol,
                period,
                holding_period,
                min_score,
                min_adx,
                max_iv,
                direction,
            )
            return (symbol, result, None)
        except Exception as e:
            return (symbol, None, str(e))


async def _handle_scan_candidates(arguments: dict) -> list[TextContent]:
    """Handle scan_candidates tool - scans multiple symbols in parallel."""
    # Pre-built symbol universes
    universes = {
        "mega_caps": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
            "ABBV", "LLY", "AVGO", "PEP", "KO", "COST", "TMO", "MCD", "WMT",
            "CSCO", "ACN", "CRM",
        ],
        "tech": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
            "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT",
            "LRCX", "KLAC", "MRVL", "NOW", "SNOW", "PLTR", "PANW", "CRWD",
            "ZS", "DDOG", "NET", "MDB", "TEAM", "SNPS", "CDNS", "ADI", "INTU",
            "FTNT", "WDAY", "ANSS", "CPRT", "COIN", "SHOP", "MELI", "SE",
            "RBLX", "U", "DOCU", "ZM", "OKTA", "SPLK", "VEEV", "TTD",
        ],
        "financials": [
            "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB",
            "PNC", "TFC", "COF", "AIG", "MET", "PRU", "ALL", "TRV", "CB", "CME",
        ],
        "healthcare": [
            "UNH", "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD",
            "MRNA", "REGN", "VRTX", "ISRG", "DXCM", "IDXX", "ZTS", "CI", "ELV",
            "HUM", "BIIB",
        ],
        "consumer": [
            "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "CMG",
            "DPZ", "YUM", "LULU", "ULTA", "ROST", "DG", "DLTR", "EL", "CL",
            "KMB", "TJX",
        ],
        "energy": [
            "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX",
            "DVN", "HAL", "BKR", "FANG", "HES", "MRO", "APA", "OVV", "CTRA",
            "EQT", "AR",
        ],
        "etfs": [
            "SPY", "QQQ", "IWM", "DIA", "EEM", "VTI", "VOO", "VEA", "VWO",
            "GLD", "SLV", "USO", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
            "XLY", "XLB", "XLU", "XLRE", "XLC", "VNQ", "HYG", "LQD", "TLT",
            "IEF", "SHY", "BND", "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ",
            "SMH", "SOXX", "IBB", "XBI", "KRE",
        ],
    }
    universes["liquid"] = list(
        set(
            universes["mega_caps"]
            + universes["tech"]
            + universes["financials"]
            + universes["healthcare"]
            + universes["consumer"]
            + universes["energy"]
        )
    )
    universes["full_market"] = list(set(universes["liquid"] + universes["etfs"]))

    # Get scanning parameters
    custom_symbols = arguments.get("symbols", [])
    universe = arguments.get("universe", "full_market").lower()
    scan_period = arguments.get("period", "3mo")
    holding_period = arguments.get("holding_period", 14)
    min_score = arguments.get("min_score", 2.0)
    min_adx = arguments.get("min_adx", 20)
    max_iv = arguments.get("max_iv_percentile", 100)
    direction = arguments.get("direction", "any").lower()
    max_results = arguments.get("max_results", 15)

    # Determine symbols to scan
    if custom_symbols and len(custom_symbols) > 0:
        symbols = [s.upper() for s in custom_symbols]
        universe_used = "custom"
    elif universe in universes:
        symbols = universes[universe]
        universe_used = universe
    else:
        symbols = universes["full_market"]
        universe_used = "full_market"

    # Parallel scanning with concurrency limit
    logger.info(
        "Starting parallel scan of %d symbols (max concurrent: %d)",
        len(symbols),
        MAX_CONCURRENT_SCANS,
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCANS)

    tasks = [
        _analyze_symbol_async(
            sym,
            scan_period,
            holding_period,
            min_score,
            min_adx,
            max_iv,
            direction,
            semaphore,
        )
        for sym in symbols
    ]

    results = await asyncio.gather(*tasks)

    # Process results
    candidates = []
    errors = []
    scanned = 0

    for sym, candidate, error in results:
        if error:
            errors.append({"symbol": sym, "error": error})
        else:
            scanned += 1
            if candidate is not None:
                candidates.append(candidate)

    logger.info("Scan complete: %d candidates from %d scanned", len(candidates), scanned)

    # Sort by absolute composite score (highest first)
    candidates.sort(key=lambda x: abs(x["composite_score"]), reverse=True)

    # Separate into bullish and bearish
    bullish = [c for c in candidates if c["composite_score"] > 0]
    bearish = [c for c in candidates if c["composite_score"] < 0]

    # Find highest conviction setups
    high_conviction = [
        c
        for c in candidates
        if abs(c["composite_score"]) >= 4 and c["adx"] >= 28 and c["iv_percentile"] <= 50
    ]

    result = {
        "scan_parameters": {
            "universe": universe_used,
            "symbols_in_universe": len(symbols),
            "symbols_scanned": scanned,
            "holding_period": holding_period,
            "min_score": min_score,
            "min_adx": min_adx,
            "max_iv_percentile": max_iv,
            "direction_filter": direction,
        },
        "summary": {
            "total_candidates": len(candidates),
            "bullish_setups": len(bullish),
            "bearish_setups": len(bearish),
            "high_conviction": len(high_conviction),
            "errors": len(errors),
        },
        "high_conviction_setups": high_conviction[:5] if high_conviction else [],
        "top_bullish": bullish[:max_results] if bullish else [],
        "top_bearish": bearish[:max_results] if bearish else [],
        "errors": errors[:10] if errors else None,
    }

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
                    },
                    "days_to_expiration": {
                        "type": "integer",
                        "description": (
                            "Days until options expiration for expected move "
                            "calculation (default: same as holding_period)"
                        ),
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
                    },
                },
                "required": [],
            },
        ),
    ]


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
        period = arguments.get("period", "1mo")

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
                "recent_values": data[cols].tail(10).to_dict(
                    orient="records"
                ),  # type: ignore[call-overload]
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
                "recent_values": data[["Date", "Close", "VWAP"]].tail(10).to_dict(
                    orient="records"
                ),  # type: ignore[call-overload]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

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

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

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
                "recent_values": data[["Date", "Close", "MFI"]].tail(10).to_dict(
                    orient="records"
                ),  # type: ignore[call-overload]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "analyze_volume_trends":
            window = arguments.get("window", 20)
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
            # Get options-specific parameters with adaptive defaults
            holding_period = arguments.get("holding_period", 14)
            days_to_expiration = arguments.get("days_to_expiration", holding_period)

            # Adaptive indicator periods based on holding period
            if holding_period <= 14:
                mfi_period = 7
                volume_window = 10
                rsi_period = 7
                adx_period = 10
                hv_window = 10
            elif holding_period <= 21:
                mfi_period = 10
                volume_window = 14
                rsi_period = 10
                adx_period = 14
                hv_window = 14
            else:  # 22-30 days
                mfi_period = 14
                volume_window = 20
                rsi_period = 14
                adx_period = 14
                hv_window = 20

            # Calculate all indicators with adaptive parameters
            obv = calculate_obv(data)
            vwap = calculate_vwap(data)
            mfi = calculate_mfi(data, mfi_period)
            vpt = calculate_vpt(data)
            trends = analyze_volume_trends(data, volume_window)
            ad_line = calculate_accumulation_distribution(data)
            cmf = calculate_chaikin_money_flow(data, volume_window)
            rvol = calculate_relative_volume(data, volume_window)
            breakout = detect_volume_breakout(data, 2.0, volume_window)
            vwma = calculate_vwma(data, volume_window)
            roc = calculate_price_roc(data, volume_window)

            # NEW: Enhanced indicators
            adx_data = calculate_adx(data, adx_period)
            rsi_data = calculate_rsi_with_divergence(data, rsi_period, volume_window)
            iv_percentile = calculate_iv_percentile(data, hv_window)
            expected_move = calculate_expected_move(data, days_to_expiration, hv_window)
            composite = calculate_composite_score(data, holding_period)

            # Volatility indicators
            hv = calculate_historical_volatility(data, hv_window)
            atr = calculate_atr(data, volume_window)
            bbands = calculate_bollinger_bands(data, volume_window)

            # Enhanced volume profile with VAH/VAL
            profile = calculate_enhanced_volume_profile(data)

            latest_close = data["Close"].iloc[-1]
            latest_vwap = vwap.iloc[-1]
            latest_vwma = vwma.iloc[-1]
            start_dt = data["Date"].iloc[0].strftime("%Y-%m-%d")
            end_dt = data["Date"].iloc[-1].strftime("%Y-%m-%d")

            # Pre-calculate values for options analysis (with bounds checking)
            if len(obv) >= 4:
                obv_up = obv.iloc[-1] > obv.iloc[-3]
            else:
                obv_up = False
            if len(ad_line) >= 4:
                ad_up = ad_line.iloc[-1] > ad_line.iloc[-3]
            else:
                ad_up = False
            if len(vpt) >= 4:
                vpt_diff = abs(vpt.iloc[-1] - vpt.iloc[-3])
                vpt_conviction = vpt_diff > abs(vpt.iloc[-3] * 0.1) if vpt.iloc[-3] != 0 else False
            else:
                vpt_diff = 0.0
                vpt_conviction = False
            mfi_val = mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50.0
            cmf_val = cmf.iloc[-1] if not pd.isna(cmf.iloc[-1]) else 0.0

            if mfi_val > 80:
                mfi_cond = "Overbought"
            elif mfi_val < 20:
                mfi_cond = "Oversold"
            else:
                mfi_cond = "Neutral"

            if mfi_val > 75:
                mfi_signal = "consider_puts"
            elif mfi_val < 25:
                mfi_signal = "consider_calls"
            else:
                mfi_signal = "neutral"

            if cmf_val > 0.25:
                cmf_signal = "strong_buying"
            elif cmf_val < -0.25:
                cmf_signal = "strong_selling"
            else:
                cmf_signal = "neutral"

            # Pre-calculate bollinger band values
            bb_upper = bbands["upper"].iloc[-1]
            bb_middle = bbands["middle"].iloc[-1]
            bb_lower = bbands["lower"].iloc[-1]
            bb_pct_b = bbands["percent_b"].iloc[-1]
            bb_bw = bbands["bandwidth"].iloc[-1]
            atr_val = atr.iloc[-1]

            if not pd.isna(bb_bw):
                bw_series = bbands["bandwidth"].dropna()
                if len(bw_series) >= volume_window:
                    bb_mean = bw_series.iloc[-volume_window:].mean()
                    is_squeeze = bb_bw < bb_mean * 0.7
                elif len(bw_series) >= MIN_SQUEEZE_DETECTION_PERIODS:
                    bb_mean = bw_series.mean()
                    is_squeeze = bb_bw < bb_mean * 0.7
                else:
                    is_squeeze = False  # Insufficient data for squeeze detection
            else:
                is_squeeze = False

            if not pd.isna(bb_pct_b):
                if bb_pct_b > 0.8:
                    bb_position = "overbought"
                elif bb_pct_b < 0.2:
                    bb_position = "oversold"
                else:
                    bb_position = "neutral"
            else:
                bb_position = "neutral"

            if not pd.isna(atr_val):
                daily_range = f"±${atr_val:.2f}"
                stop_low = latest_close - (2 * atr_val)
                stop_high = latest_close - (1.5 * atr_val)
                stop_loss = f"${stop_low:.2f} to ${stop_high:.2f}"
            else:
                daily_range = "N/A"
                stop_loss = "N/A"

            # Time decay risk assessment
            if days_to_expiration > 21:
                theta_risk = "low"
                theta_note = "Comfortable theta decay - can hold through minor pullbacks"
            elif days_to_expiration > 14:
                theta_risk = "moderate"
                theta_note = "Monitor daily - theta acceleration begins"
            elif days_to_expiration > 7:
                theta_risk = "elevated"
                theta_note = "Active management required - theta decay significant"
            else:
                theta_risk = "critical"
                theta_note = "Urgent - close or roll positions to avoid rapid decay"

            result = {
                "symbol": symbol,
                "analysis_type": f"Options Trading ({holding_period}-Day Optimized)",
                "period": f"{start_dt} to {end_dt}",
                "latest_price": float(latest_close),
                "parameters": {
                    "holding_period": holding_period,
                    "days_to_expiration": days_to_expiration,
                    "mfi_period": mfi_period,
                    "volume_window": volume_window,
                    "rsi_period": rsi_period,
                    "adx_period": adx_period,
                    "hv_window": hv_window,
                    "optimization": f"Adaptive for {holding_period}-day options",
                },
                "composite_signal": {
                    "score": composite["composite_score"],
                    "recommendation": composite["recommendation"],
                    "action": composite["action"],
                    "signal_quality": composite["signal_quality"],
                    "quality_note": composite["quality_note"],
                    "score_breakdown": composite["score_breakdown"],
                },
                "trend_analysis": {
                    "adx": {
                        "value": adx_data["adx"],
                        "plus_di": adx_data["plus_di"],
                        "minus_di": adx_data["minus_di"],
                        "trend_strength": adx_data["trend_strength"],
                        "trend_direction": adx_data["trend_direction"],
                        "adx_slope": adx_data["adx_slope"],
                        "interpretation": adx_data["interpretation"],
                    },
                    "rsi": {
                        "value": rsi_data["rsi"],
                        "condition": rsi_data["condition"],
                        "divergence_type": rsi_data["divergence_type"],
                        "divergence_signal": rsi_data["signal"],
                        "interpretation": rsi_data["interpretation"],
                    },
                },
                "volume_indicators": {
                    "obv": {
                        "value": float(obv.iloc[-1]),
                        "trend": "increasing" if obv_up else "decreasing",
                        "short_term_momentum": "bullish" if obv_up else "bearish",
                    },
                    "accumulation_distribution": {
                        "value": float(ad_line.iloc[-1]),
                        "trend": "increasing" if ad_up else "decreasing",
                        "signal": "institutional_buying" if ad_up else "institutional_selling",
                    },
                    "vpt": {
                        "value": float(vpt.iloc[-1]),
                        "trend": "increasing" if vpt.iloc[-1] > vpt.iloc[-3] else "decreasing",
                        "volume_conviction": "strong" if vpt_conviction else "weak",
                    },
                    "mfi": {
                        "value": float(mfi_val),
                        "condition": mfi_cond,
                        "options_signal": mfi_signal,
                    },
                    "cmf": {"value": float(cmf_val), "signal": cmf_signal},
                    "relative_volume": {
                        "current_rvol": rvol["current_rvol"],
                        "significance": rvol["significance"],
                    },
                    "volume_breakout": breakout,
                },
                "price_indicators": {
                    "vwap": {
                        "value": float(latest_vwap),
                        "price_vs_vwap": f"{((latest_close / latest_vwap - 1) * 100):.2f}%",
                        "position": "above" if latest_close > latest_vwap else "below",
                        "signal": "bullish_entry"
                        if latest_close > latest_vwap
                        else "bearish_entry",
                    },
                    "vwma": {
                        "value": float(latest_vwma),
                        "price_vs_vwma": f"{((latest_close / latest_vwma - 1) * 100):.2f}%",
                        "trend": "bullish" if latest_close > latest_vwma else "bearish",
                    },
                    "price_roc": {
                        "current_roc": roc["current_roc"],
                        "direction": roc["direction"],
                        "strength": roc["strength"],
                        "volume_confirmed": roc["volume_confirmed"],
                    },
                },
                "volatility_analysis": {
                    "iv_percentile_proxy": {
                        "percentile": iv_percentile["iv_percentile"],
                        "current_hv": iv_percentile["current_hv"],
                        "hv_range": f"{iv_percentile['hv_min']:.1%} - {iv_percentile['hv_max']:.1%}",  # noqa: E501
                        "interpretation": iv_percentile["interpretation"],
                        "options_implication": iv_percentile["options_implication"],
                        "strategy_suggestion": iv_percentile["strategy_suggestion"],
                    },
                    "expected_move": {
                        "dollars": expected_move["expected_move_dollars"],
                        "percent": expected_move["expected_move_percent"],
                        "upper_target": expected_move["upper_target_1std"],
                        "lower_target": expected_move["lower_target_1std"],
                        "targets": expected_move["targets"],
                        "strike_guidance": expected_move["strike_guidance"],
                        "interpretation": expected_move["interpretation"],
                    },
                    "atr": {
                        "value": float(atr_val) if not pd.isna(atr_val) else 0.0,
                        "daily_range": daily_range,
                        "stop_loss_suggestion": stop_loss,
                    },
                    "bollinger_bands": {
                        "upper": float(bb_upper) if not pd.isna(bb_upper) else 0.0,
                        "middle": float(bb_middle) if not pd.isna(bb_middle) else 0.0,
                        "lower": float(bb_lower) if not pd.isna(bb_lower) else 0.0,
                        "percent_b": float(bb_pct_b) if not pd.isna(bb_pct_b) else 0.0,
                        "bandwidth": float(bb_bw) if not pd.isna(bb_bw) else 0.0,
                        "squeeze_detected": is_squeeze,
                        "position": bb_position,
                    },
                },
                "volume_profile": {
                    "point_of_control": profile["poc"],
                    "value_area_high": profile["vah"],
                    "value_area_low": profile["val"],
                    "current_position": profile["position"],
                    "interpretation": profile["interpretation"],
                    "strike_selection_guidance": {
                        "poc_strike": f"${profile['poc']:.2f} - Highest probability",
                        "vah_strike": f"${profile['vah']:.2f} - Resistance level",
                        "val_strike": f"${profile['val']:.2f} - Support level",
                        "current_vs_poc": f"{profile['poc_distance_pct']:.2f}%",
                    },
                },
                "time_decay": {
                    "days_to_expiration": days_to_expiration,
                    "theta_risk": theta_risk,
                    "theta_note": theta_note,
                },
                "volume_trends": trends,
                "options_insights": generate_options_insights(
                    composite,
                    adx_data,
                    rsi_data,
                    iv_percentile,
                    expected_move,
                    profile,
                    rvol,
                    breakout,
                    trends,
                    mfi_val,
                    cmf_val,
                    is_squeeze,
                    bb_pct_b,
                    holding_period,
                    latest_close,
                ),
            }

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


def generate_options_summary(
    data, obv, vwap, mfi, trends, latest_close, latest_vwap, poc, mfi_period, volume_window
):
    """Generate options-specific trading insights (legacy function for compatibility)."""
    summary = []

    # Entry Signal Assessment
    price_above_vwap = latest_close > latest_vwap
    obv_momentum = obv.iloc[-1] > obv.iloc[-3]
    latest_mfi = mfi.iloc[-1]

    # Overall bias
    if price_above_vwap and obv_momentum and latest_mfi < 70:
        summary.append(
            "✓ BULLISH SETUP: Price above VWAP with positive OBV momentum - Consider call options"
        )
    elif not price_above_vwap and not obv_momentum and latest_mfi > 30:
        summary.append(
            "✓ BEARISH SETUP: Price below VWAP with negative OBV momentum - Consider put options"
        )
    else:
        summary.append("⚠️  MIXED SIGNALS: Wait for clearer directional confirmation")

    # Strike Selection Guidance
    poc_distance = (latest_close / poc - 1) * 100
    if abs(poc_distance) < 3:
        summary.append(
            f"Strike Selection: Price near POC (${poc:.2f}) - "
            "High probability support/resistance zone"
        )
    elif poc_distance > 3:
        summary.append(
            f"Strike Selection: Price {poc_distance:.1f}% above POC (${poc:.2f}) - "
            "POC may act as support on pullback"
        )
    else:
        summary.append(
            f"Strike Selection: Price {abs(poc_distance):.1f}% below POC (${poc:.2f}) - "
            "POC may act as resistance"
        )

    # Volume Conviction
    if trends["divergence_detected"]:
        div_type = trends["divergence_type"]
        summary.append(
            f"⚠️  VOLUME DIVERGENCE: {div_type} - Momentum may be weakening, use tight stops"
        )
    elif trends["volume_vs_average"].startswith("-"):
        summary.append(
            "⚠️  Below-average volume - Moves may lack conviction, consider smaller position sizing"
        )
    else:
        summary.append("✓ Above-average volume - Price moves have strong participation")

    # MFI-specific options guidance
    if latest_mfi > 75:
        summary.append(
            f"MFI Warning: At {latest_mfi:.1f}, approaching overbought - "
            "Call buyers beware, put sellers consider premium"
        )
    elif latest_mfi < 25:
        summary.append(
            f"MFI Opportunity: At {latest_mfi:.1f}, approaching oversold - "
            "Put buyers beware, call sellers consider premium"
        )

    # Time decay consideration
    summary.append(
        f"Indicator Responsiveness: Using {mfi_period}d MFI and "
        f"{volume_window}d volume window for faster signals"
    )

    return summary


def generate_enhanced_options_summary(
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
    mfi_period,
    volume_window,
):
    """Generate comprehensive options-specific trading insights."""
    summary = []

    # Entry Signal Assessment with multiple confirmations
    price_above_vwap = latest_close > latest_vwap
    price_above_vwma = latest_close > vwma.iloc[-1]
    obv_momentum = obv.iloc[-1] > obv.iloc[-3]
    ad_momentum = ad_line.iloc[-1] > ad_line.iloc[-3]
    latest_mfi = mfi.iloc[-1]
    latest_cmf = cmf.iloc[-1]

    # Overall bias with multiple confirmations
    bullish_signals = sum(
        [
            price_above_vwap,
            price_above_vwma,
            obv_momentum,
            ad_momentum,
            latest_mfi < 70,
            latest_cmf > 0,
        ]
    )
    bearish_signals = sum(
        [
            not price_above_vwap,
            not price_above_vwma,
            not obv_momentum,
            not ad_momentum,
            latest_mfi > 30,
            latest_cmf < 0,
        ]
    )

    if bullish_signals >= 5:
        summary.append(
            "✓✓ STRONG BULLISH SETUP: Multiple confirmations - High probability call options"
        )
    elif bullish_signals >= 4:
        summary.append(
            "✓ BULLISH SETUP: Price above key averages with volume support - Consider call options"
        )
    elif bearish_signals >= 5:
        summary.append(
            "✓✓ STRONG BEARISH SETUP: Multiple confirmations - High probability put options"
        )
    elif bearish_signals >= 4:
        summary.append(
            "✓ BEARISH SETUP: Price below key averages with distribution - Consider put options"
        )
    else:
        summary.append(
            "⚠️  MIXED SIGNALS: Wait for clearer directional confirmation before entering"
        )

    # Volatility-based Strategy Selection
    if not pd.isna(hv.iloc[-1]):
        hv_val = hv.iloc[-1]
        if hv_val > 0.40:
            summary.append(
                f"⚠️  EXTREME VOLATILITY ({hv_val:.1%}): Consider selling "
                f"premium (credit spreads, iron condors)"
            )
        elif hv_val > 0.30:
            summary.append(
                f"High Volatility ({hv_val:.1%}): Options expensive - "
                f"favor spreads over naked long options"
            )
        elif hv_val < 0.15:
            summary.append(
                f"✓ Low Volatility ({hv_val:.1%}): Cheap options - "
                f"good time for directional long plays or straddles"
            )

    # Strike Selection with VAH/VAL
    poc = profile["poc"]
    vah = profile["vah"]
    val = profile["val"]

    if profile["position"] == "above_value_area":
        summary.append(
            f"Strike Selection: Price ${latest_close:.2f} above value area - "
            f"VAH ${vah:.2f} is key resistance, POC ${poc:.2f} is support"
        )
    elif profile["position"] == "below_value_area":
        summary.append(
            f"Strike Selection: Price ${latest_close:.2f} below value area - "
            f"VAL ${val:.2f} is key support, POC ${poc:.2f} is resistance"
        )
    else:
        summary.append(
            f"Strike Selection: Price in value area - POC ${poc:.2f} is magnet, "
            f"VAH ${vah:.2f}/VAL ${val:.2f} are boundaries"
        )

    # ATR-based Position Sizing
    if not pd.isna(atr.iloc[-1]):
        atr_val = atr.iloc[-1]
        atr_pct = (atr_val / latest_close) * 100
        summary.append(
            f"Position Sizing: Daily ATR is ${atr_val:.2f} ({atr_pct:.1f}% of "
            f"price) - Use for stop placement"
        )

    # Bollinger Band Strategy
    if not pd.isna(bbands["percent_b"].iloc[-1]):
        percent_b = bbands["percent_b"].iloc[-1]
        bw_mean = bbands["bandwidth"].iloc[-volume_window:].mean() * 0.7
        squeeze = bbands["bandwidth"].iloc[-1] < bw_mean

        if squeeze:
            summary.append(
                "⚠️  BOLLINGER SQUEEZE: Breakout imminent - consider straddle/strangle strategies"
            )
        elif percent_b > 0.9:
            summary.append(
                f"Bollinger %B at {percent_b:.2f} - Near upper band, "
                f"consider put spreads or profit-taking on calls"
            )
        elif percent_b < 0.1:
            summary.append(
                f"Bollinger %B at {percent_b:.2f} - Near lower band, "
                f"consider call spreads or profit-taking on puts"
            )

    # Volume Conviction Assessment
    if breakout["is_breakout"]:
        mult = breakout["multiplier_above_avg"]
        direction = breakout["direction"]
        summary.append(
            f"✓ VOLUME BREAKOUT: {mult:.1f}x average volume, {direction} - Strong conviction"
        )
    elif rvol["current_rvol"] > 1.5:
        summary.append(
            f"✓ High relative volume ({rvol['current_rvol']:.1f}x) - "
            f"Moves have institutional participation"
        )
    elif trends["divergence_detected"]:
        summary.append(
            f"⚠️  VOLUME DIVERGENCE: {trends['divergence_type']} - "
            f"Trend weakening, tighten stops on existing positions"
        )
    elif rvol["current_rvol"] < 0.7:
        summary.append(
            "⚠️  Below-average volume - Wait for confirmation before entering new positions"
        )

    # MFI/CMF Combined Signal
    if latest_mfi > 80 and latest_cmf > 0.25:
        summary.append(
            f"⚠️  EXTREME OVERBOUGHT: MFI {latest_mfi:.1f}, CMF {latest_cmf:.2f} "
            f"- High reversal risk, consider puts or exit calls"
        )
    elif latest_mfi < 20 and latest_cmf < -0.25:
        summary.append(
            f"✓ EXTREME OVERSOLD: MFI {latest_mfi:.1f}, CMF {latest_cmf:.2f} - "
            f"Strong bounce potential, consider calls"
        )
    elif latest_mfi > 75:
        summary.append(
            f"MFI {latest_mfi:.1f} - Approaching overbought, consider "
            f"profit-taking on calls or put spreads"
        )
    elif latest_mfi < 25:
        summary.append(
            f"MFI {latest_mfi:.1f} - Approaching oversold, consider "
            f"profit-taking on puts or call spreads"
        )

    # Institutional Activity (A/D Line)
    if ad_momentum and latest_cmf > 0.1:
        summary.append(
            "✓ Institutional accumulation detected - Smart money buying, bullish for options"
        )
    elif not ad_momentum and latest_cmf < -0.1:
        summary.append(
            "⚠️  Institutional distribution detected - Smart money selling, bearish for options"
        )

    # Time Decay Reminder
    summary.append(
        f"⏱️  Short-term setup: {mfi_period}d MFI, {volume_window}d windows - "
        f"Optimized for 10-14 day options"
    )
    summary.append(
        "⏱️  Monitor theta decay closely on long positions - "
        "Consider spreads to reduce time decay impact"
    )

    return summary


def generate_options_insights(
    composite,
    adx_data,
    rsi_data,
    iv_percentile,
    expected_move,
    profile,
    rvol,
    breakout,
    trends,
    mfi_val,
    cmf_val,
    is_squeeze,
    bb_pct_b,
    holding_period,
    latest_close,
):
    """Generate comprehensive options trading insights for 14-30 day plays."""
    insights = []

    # 1. Primary Signal - Composite Score
    score = composite["composite_score"]
    if score >= 5:
        insights.append(
            f"✓✓ STRONG BULLISH: Composite score {score:.1f}/10 - High conviction call setup"
        )
    elif score >= 2:
        insights.append(
            f"✓ BULLISH: Composite score {score:.1f}/10 - Consider call options or bull spreads"
        )
    elif score <= -5:
        insights.append(
            f"✓✓ STRONG BEARISH: Composite score {score:.1f}/10 - High conviction put setup"
        )
    elif score <= -2:
        insights.append(
            f"✓ BEARISH: Composite score {score:.1f}/10 - Consider put options or bear spreads"
        )
    else:
        insights.append(
            f"⚠️  NEUTRAL: Composite score {score:.1f}/10 - "
            f"No clear directional edge, consider iron condors or wait"
        )

    # 2. Trend Quality Assessment
    adx = adx_data["adx"]
    trend_dir = adx_data["trend_direction"]
    if adx > 30:
        insights.append(
            f"✓ STRONG TREND: ADX at {adx:.1f} ({trend_dir}) - "
            f"Directional plays have wind at their back"
        )
    elif adx > 25:
        insights.append(
            f"✓ Moderate Trend: ADX at {adx:.1f} ({trend_dir}) - "
            f"Decent setup for directional options"
        )
    elif adx > 20:
        insights.append(
            f"⚠️  Weak Trend: ADX at {adx:.1f} - "
            f"Consider reduced position size or neutral strategies"
        )
    else:
        insights.append(
            f"⚠️  NO TREND: ADX at {adx:.1f} - Premium selling (iron condors, strangles) preferred"
        )

    # 3. RSI Divergence Alert
    if rsi_data["divergence_type"] == "bullish":
        insights.append(
            "✓✓ RSI BULLISH DIVERGENCE: Price weakness not confirmed by momentum - "
            "Potential reversal up, favor calls"
        )
    elif rsi_data["divergence_type"] == "bearish":
        insights.append(
            "✓✓ RSI BEARISH DIVERGENCE: Price strength not confirmed by momentum - "
            "Potential reversal down, favor puts"
        )

    # 4. IV Percentile / Volatility Edge
    iv_pct = iv_percentile["iv_percentile"]
    if iv_pct > 80:
        insights.append(
            f"⚠️  HIGH IV PERCENTILE ({iv_pct:.0f}%): Options are EXPENSIVE - "
            f"Favor selling premium (credit spreads, iron condors)"
        )
    elif iv_pct > 60:
        insights.append(
            f"IV slightly elevated ({iv_pct:.0f}%) - Consider debit spreads to reduce vega risk"
        )
    elif iv_pct < 20:
        insights.append(
            f"✓ LOW IV PERCENTILE ({iv_pct:.0f}%): Options are CHEAP - "
            f"Great time for long options, straddles, or strangles"
        )
    elif iv_pct < 40:
        insights.append(
            f"✓ Below-average IV ({iv_pct:.0f}%) - Long directional plays are reasonably priced"
        )

    # 5. Expected Move Guidance
    em_pct = expected_move["expected_move_percent"]
    em_upper = expected_move["upper_target_1std"]
    em_lower = expected_move["lower_target_1std"]
    insights.append(
        f"Expected Move: ±{em_pct:.1f}% by expiration - "
        f"Target range ${em_lower:.2f} to ${em_upper:.2f} (68% probability)"
    )

    # 6. Strike Selection from Volume Profile
    poc = profile["poc"]
    vah = profile["vah"]
    val = profile["val"]
    position = profile["position"]

    if position == "above_value_area":
        insights.append(
            f"Strike Guidance: Price above value area - "
            f"VAH ${vah:.2f} is key support if going long calls, "
            f"POC ${poc:.2f} is downside target for puts"
        )
    elif position == "below_value_area":
        insights.append(
            f"Strike Guidance: Price below value area - "
            f"VAL ${val:.2f} is key resistance if going long puts, "
            f"POC ${poc:.2f} is upside target for calls"
        )
    else:
        insights.append(
            f"Strike Guidance: Price in value area - "
            f"POC ${poc:.2f} acts as magnet, "
            f"VAH ${vah:.2f}/VAL ${val:.2f} are boundary targets"
        )

    # 7. Bollinger Squeeze Alert
    if is_squeeze:
        insights.append(
            "✓✓ BOLLINGER SQUEEZE DETECTED: Volatility compressed - "
            "Breakout imminent! Consider straddles or wait for direction"
        )

    if bb_pct_b is not None and not pd.isna(bb_pct_b):
        if bb_pct_b > 0.95:
            insights.append(
                f"⚠️  Price at upper Bollinger Band ({bb_pct_b:.0%}) - "
                f"Extended, consider puts or profit-taking on calls"
            )
        elif bb_pct_b < 0.05:
            insights.append(
                f"✓ Price at lower Bollinger Band ({bb_pct_b:.0%}) - "
                f"Oversold, consider calls or profit-taking on puts"
            )

    # 8. Volume Conviction
    if breakout["is_breakout"]:
        direction = breakout["direction"]
        mult = breakout["multiplier_above_avg"]
        insights.append(
            f"✓ VOLUME BREAKOUT: {mult:.1f}x average ({direction}) - "
            f"Strong conviction behind the move"
        )
    elif rvol["current_rvol"] > 1.5:
        insights.append(
            f"✓ High volume ({rvol['current_rvol']:.1f}x average) - "
            f"Institutional participation detected"
        )
    elif rvol["current_rvol"] < 0.7:
        insights.append("⚠️  Low volume - Wait for volume confirmation before entry")

    # 9. Divergence Warning
    if trends["divergence_detected"]:
        insights.append(
            f"⚠️  PRICE-VOLUME DIVERGENCE: {trends['divergence_type']} - "
            f"Current trend may be weakening"
        )

    # 10. MFI/CMF Extremes
    if mfi_val > 80 and cmf_val > 0.2:
        insights.append(
            "⚠️  EXTREME OVERBOUGHT: MFI + CMF both elevated - "
            "High reversal risk, protect call profits or consider puts"
        )
    elif mfi_val < 20 and cmf_val < -0.2:
        insights.append(
            "✓ EXTREME OVERSOLD: MFI + CMF both depressed - "
            "Bounce potential high, consider calls for mean reversion"
        )

    # 11. Holding Period Reminder
    if holding_period <= 14:
        insights.append(
            f"⏱️  {holding_period}-day holding period: Using fast indicators - "
            f"Execute quickly, manage theta aggressively"
        )
    elif holding_period <= 21:
        insights.append(
            f"⏱️  {holding_period}-day holding period: Balanced approach - "
            f"Monitor daily, theta decay moderate"
        )
    else:
        insights.append(
            f"⏱️  {holding_period}-day holding period: Standard indicators - "
            f"Can weather short-term volatility, theta decay manageable"
        )

    return insights


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
