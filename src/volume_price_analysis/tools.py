"""The MCP tool registry: one record per tool, name + schema + run callable.

``server.py`` is only the MCP adapter and a single dispatcher; everything a
tool actually *does* lives here, in its :class:`ToolSpec`. A tool's ``run``
receives a :class:`ToolContext` carrying the parsed arguments, the
:class:`~volume_price_analysis.data_fetcher.DataSource` to read market data
through, and a lazy ``fetch()`` for the standard symbol/period/start/end
arguments. ``scan_candidates`` is the one tool that never calls ``fetch()`` --
it fetches per symbol inside ``run_scan``.

Adding a tool means appending one :class:`ToolSpec` to :data:`TOOLS`; the
list-tools response and the dispatcher both derive from it.
"""

import functools
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from .analysis import run_comprehensive_analysis, run_options_analysis, run_scan
from .data_fetcher import DataSource
from .indicators import (
    TREND_LOOKBACK,
    _parse_anchor_date,
    analyze_volume_trends,
    calculate_accumulation_distribution,
    calculate_anchored_vwap,
    calculate_chaikin_money_flow,
    calculate_mfi,
    calculate_obv,
    calculate_rsi_divergence,
    calculate_volume_profile,
    calculate_vwap,
    trend_verdict,
)
from .interpretation import (
    CMF_PRESSURE_BANDS,
    MFI_TOOL_CONDITION_BANDS,
    threshold_verdict,
)
from .output_schemas import (
    COMPREHENSIVE_ANALYSIS_SCHEMA,
    OPTIONS_ANALYSIS_SCHEMA,
    SCAN_RESULT_SCHEMA,
)
from .payloads import ComprehensiveAnalysis, OptionsAnalysis, ScanResult
from .sessions import SessionState, session_state

logger = logging.getLogger(__name__)


def validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate supplied schema types and coerce numeric values before range checks."""
    required = schema.get("required", [])
    for name, prop in schema.get("properties", {}).items():
        if name not in arguments:
            continue
        value = arguments[name]
        if value is None and name not in required:
            del arguments[name]
            continue
        kind = prop.get("type")
        if kind in ("integer", "number"):
            message = f"{name} must be a finite {kind}"
            if isinstance(value, bool) or not isinstance(value, int | float | str):
                raise ValueError(message)
            try:
                number = float(value)
            except ValueError, OverflowError:
                raise ValueError(message) from None
            if not math.isfinite(number) or (kind == "integer" and not number.is_integer()):
                raise ValueError(message)
            value = int(value) if isinstance(value, int) else number
            if kind == "integer":
                value = int(value)
            arguments[name] = value
        elif kind == "string":
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            if "enum" in prop and value not in prop["enum"]:
                raise ValueError(f"{name} must be one of {prop['enum']}")
            continue
        elif kind == "array":
            if not isinstance(value, list):
                raise ValueError(f"{name} must be an array")
            if prop.get("items", {}).get("type") == "string":
                for index, item in enumerate(value):
                    if not isinstance(item, str) or not item.strip():
                        raise ValueError(f"{name}[{index}] must be a non-empty string")
            continue
        elif kind == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            continue
        else:
            continue
        low, high = prop.get("minimum"), prop.get("maximum")
        too_low = low is not None and value < low
        too_high = high is not None and value > high
        if not (too_low or too_high):
            continue
        if low is not None and high is not None:
            msg = f"{name} must be between {low} and {high}, got {value}"
        elif low is not None:
            msg = f"{name} must be at least {low}, got {value}"
        else:
            msg = f"{name} must be at most {high}, got {value}"
        raise ValueError(msg)


@dataclass
class ToolContext:
    """Everything one tool invocation is allowed to touch.

    ``fetch()`` is lazy (tools that never need prices never trigger a fetch),
    memoised (one invocation is one fetch), and hands out a copy each call so
    no tool can mutate the frame the data source owns.
    """

    args: dict[str, Any]
    data_source: DataSource
    default_period: str = "1mo"
    now: datetime | None = None
    fetched_frame: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _session_state: SessionState | None = field(default=None, init=False, repr=False)

    def require_symbol(self) -> str:
        """Return the upper-cased ``symbol`` argument, or raise if absent/blank."""
        symbol: str = self.args.get("symbol", "").upper()
        if not symbol.strip():
            raise ValueError("symbol parameter is required")
        return symbol

    @property
    def period(self) -> str:
        """The requested period, falling back to this tool's default."""
        period: str = self.args.get("period", self.default_period)
        return period

    @property
    def include_partial_bar(self) -> bool:
        """Whether the caller opted in to scoring today's in-progress daily bar."""
        return bool(self.args.get("include_partial_bar", False))

    @property
    def session(self) -> SessionState:
        """Resolved NYSE session state for this invocation (evaluated once)."""
        if self._session_state is None:
            self._session_state = session_state(self.now)
        return self._session_state

    def fetch(self) -> pd.DataFrame:
        """Return the OHLCV frame for the standard symbol/period/start/end args.

        Drops today's in-progress daily bar unless ``include_partial_bar`` is
        True or today's NYSE session has already settled.
        """
        if self.fetched_frame is None:
            symbol = self.require_symbol()
            frame = self.data_source.fetch(
                symbol,
                period=self.period,
                start=self.args.get("start_date"),
                end=self.args.get("end_date"),
            )
            cutoff = None if self.include_partial_bar else self.session.partial_bar_cutoff
            if cutoff is not None:
                dates = pd.to_datetime(frame["Date"]).dt.date
                frame = frame[dates < cutoff].reset_index(drop=True)
                if frame.empty:
                    raise ValueError(
                        f"No completed daily bars available for {symbol} "
                        f"before {cutoff.isoformat()}"
                    )
            self.fetched_frame = frame
            logger.debug("Data fetched for %s: %d rows", symbol, len(self.fetched_frame))
        return self.fetched_frame.copy()

    def session_metadata(self) -> dict[str, Any]:
        """Session completeness fields attached to every bar-scoring tool result."""
        state = self.session
        if self.fetched_frame is not None and not self.fetched_frame.empty:
            dates = pd.to_datetime(self.fetched_frame["Date"]).dt.date
            last_date = dates.iloc[-1]
            complete = state.is_bar_complete(last_date)
            if complete:
                as_of_session = last_date
            else:
                completed = dates[dates < state.today]
                as_of_session = (
                    completed.iloc[-1] if not completed.empty else state.last_complete_session
                )
            return {
                "as_of_session": as_of_session.isoformat(),
                "last_bar_complete": complete,
            }
        return {
            "as_of_session": state.last_complete_session.isoformat(),
            "last_bar_complete": state.session_complete or not self.include_partial_bar,
        }


@dataclass(frozen=True)
class ToolSpec:
    """One MCP tool: how it is advertised and what it does."""

    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[ToolContext], Awaitable[Any]]
    default_period: str = "1mo"
    # JSON Schema of the result dict, advertised as the MCP ``outputSchema``. A tool
    # that declares one also returns its result as ``structuredContent``.
    output_schema: dict[str, Any] | None = None


# --- Shared schema fragments -------------------------------------------------

_SYMBOL_PROPERTY = {
    "type": "string",
    "description": "Stock ticker symbol",
}
_START_DATE_PROPERTY = {
    "type": "string",
    "description": "Start date in YYYY-MM-DD format (optional)",
}
_END_DATE_PROPERTY = {
    "type": "string",
    "description": "End date in YYYY-MM-DD format (optional)",
}
_INCLUDE_PARTIAL_BAR_PROPERTY = {
    "type": "boolean",
    "description": (
        "Include today's in-progress daily bar before the NYSE session settles "
        "(16:15 ET regular, 13:15 ET half-day). Default: false"
    ),
    "default": False,
}


def _single_symbol_schema(
    period_description: str = "Period if dates not specified (default: '1mo')",
    period_default: str = "1mo",
    **extra_properties: dict[str, Any],
) -> dict[str, Any]:
    """Schema for the standard symbol/start_date/end_date/period argument set."""
    return {
        "type": "object",
        "properties": {
            "symbol": dict(_SYMBOL_PROPERTY),
            "start_date": dict(_START_DATE_PROPERTY),
            "end_date": dict(_END_DATE_PROPERTY),
            "period": {
                "type": "string",
                "description": period_description,
                "default": period_default,
            },
            "include_partial_bar": dict(_INCLUDE_PARTIAL_BAR_PROPERTY),
            **extra_properties,
        },
        "required": ["symbol"],
    }


def _with_session_metadata(
    fn: Callable[[ToolContext], Awaitable[Any]],
) -> Callable[[ToolContext], Awaitable[Any]]:
    """Attach ``as_of_session`` and ``last_bar_complete`` to a bar-scoring tool's result."""

    @functools.wraps(fn)
    async def wrapped(ctx: ToolContext) -> Any:
        result = await fn(ctx)
        if isinstance(result, dict):
            return {**result, **ctx.session_metadata()}
        return result

    return wrapped


# --- Tool bodies -------------------------------------------------------------


@_with_session_metadata
async def _run_get_stock_data(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    start_date = ctx.args.get("start_date")
    end_date = ctx.args.get("end_date")
    data = ctx.fetch()

    start_dt = data["Date"].iloc[0].strftime("%Y-%m-%d")
    end_dt = data["Date"].iloc[-1].strftime("%Y-%m-%d")
    return {
        "symbol": symbol,
        "period": f"{start_date} to {end_date}" if start_date and end_date else ctx.period,
        "data_points": len(data),
        "date_range": f"{start_dt} to {end_dt}",
        "latest_close": float(data["Close"].iloc[-1]),
        "latest_volume": int(data["Volume"].iloc[-1]),
        "sample_data": data.tail(5).to_dict(orient="records"),
    }


@_with_session_metadata
async def _run_calculate_obv(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    data = ctx.fetch()

    obv = calculate_obv(data)
    data["OBV"] = obv

    cols = ["Date", "Close", "Volume", "OBV"]
    return {
        "symbol": symbol,
        "indicator": "On-Balance Volume (OBV)",
        "latest_obv": float(obv.iloc[-1]),
        "obv_trend": trend_verdict(obv, TREND_LOOKBACK),
        "data_points": len(obv),
        "recent_values": data[cols].tail(10).to_dict(orient="records"),  # type: ignore[call-overload]
    }


@_with_session_metadata
async def _run_calculate_vwap(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    data = ctx.fetch()

    vwap = calculate_vwap(data)
    data["VWAP"] = vwap

    latest_close = data["Close"].iloc[-1]
    latest_vwap = vwap.iloc[-1]
    position = "above" if latest_close > latest_vwap else "below"

    return {
        "symbol": symbol,
        "indicator": "Volume Weighted Average Price (VWAP)",
        "latest_vwap": float(latest_vwap),
        "latest_close": float(latest_close),
        "price_vs_vwap": f"{((latest_close / latest_vwap - 1) * 100):.2f}%",
        "position": f"Price is {position} VWAP",
        "recent_values": data[["Date", "Close", "VWAP"]].tail(10).to_dict(orient="records"),  # type: ignore[call-overload]
    }


@_with_session_metadata
async def _run_calculate_anchored_vwap(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    if ctx.args.get("anchor", "swing_low") == "date":
        _parse_anchor_date(ctx.args.get("date"))
    data = ctx.fetch()
    result = calculate_anchored_vwap(
        data, ctx.args.get("anchor", "swing_low"), ctx.args.get("date")
    )
    series = result.pop("avwap_series")
    # Slice positionally: fetched frames need not have a unique index.
    tail = data.iloc[len(data) - len(series) :].copy()
    tail["AVWAP"] = series.to_numpy()
    return {
        "symbol": symbol,
        **result,
        "recent_values": tail[["Date", "Close", "AVWAP"]].tail(10).to_dict(orient="records"),
    }


@_with_session_metadata
async def _run_calculate_volume_profile(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    num_bins = ctx.args.get("num_bins", 20)
    data = ctx.fetch()

    profile = calculate_volume_profile(data, num_bins)

    # Find the price level with highest volume (Point of Control)
    max_volume_idx = profile["volumes"].index(max(profile["volumes"]))
    poc = profile["price_levels"][max_volume_idx]

    price_min = min(profile["price_levels"])
    price_max = max(profile["price_levels"])
    return {
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


@_with_session_metadata
async def _run_calculate_mfi(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    mfi_period = ctx.args.get("mfi_period", 14)
    data = ctx.fetch()

    mfi = calculate_mfi(data, mfi_period)
    data["MFI"] = mfi

    latest_mfi = mfi.iloc[-1]
    condition = threshold_verdict(latest_mfi, MFI_TOOL_CONDITION_BANDS, "Neutral (20-80)")

    return {
        "symbol": symbol,
        "indicator": f"Money Flow Index (MFI-{mfi_period})",
        "latest_mfi": float(latest_mfi),
        "condition": condition,
        "recent_values": data[["Date", "Close", "MFI"]].tail(10).to_dict(orient="records"),  # type: ignore[call-overload]
    }


@_with_session_metadata
async def _run_calculate_ad_line(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    data = ctx.fetch()

    ad_line = calculate_accumulation_distribution(data)
    data["AD_Line"] = ad_line

    data_points = len(ad_line)
    latest_value = ad_line.iloc[-1] if data_points > 0 else None
    latest_ad_line = None if latest_value is None or pd.isna(latest_value) else float(latest_value)

    ad_trend = trend_verdict(ad_line, TREND_LOOKBACK)

    return {
        "symbol": symbol,
        "indicator": "Accumulation/Distribution Line (A/D Line)",
        "latest_ad_line": latest_ad_line,
        "ad_trend": ad_trend,
        "data_points": data_points,
        "recent_values": data[["Date", "Close", "Volume", "AD_Line"]]
        .tail(10)
        .to_dict(orient="records"),  # type: ignore[call-overload]
    }


@_with_session_metadata
async def _run_calculate_cmf(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    cmf_period = ctx.args.get("cmf_period", 20)
    data = ctx.fetch()

    cmf = calculate_chaikin_money_flow(data, cmf_period)
    data["CMF"] = cmf

    # CMF uses a rolling window, so the first (period-1) values are NaN.
    # Extract the last finite value, falling back to None if all values are
    # NaN/inf (NaN from insufficient data, inf from zero rolling volume sum).
    valid_cmf = cmf.dropna()
    latest_valid_cmf = valid_cmf.iloc[-1] if not valid_cmf.empty else None
    if latest_valid_cmf is not None and not math.isfinite(latest_valid_cmf):
        latest_valid_cmf = None

    condition = threshold_verdict(
        latest_valid_cmf,
        CMF_PRESSURE_BANDS,
        "Neutral (0)",
        missing="Insufficient Data",
    )
    latest_cmf_val = None if latest_valid_cmf is None else float(latest_valid_cmf)

    return {
        "symbol": symbol,
        "indicator": f"Chaikin Money Flow (CMF-{cmf_period})",
        "latest_cmf": latest_cmf_val,
        "condition": condition,
        "recent_values": data[["Date", "Close", "CMF"]].tail(10).to_dict(orient="records"),  # type: ignore[call-overload]
    }


@_with_session_metadata
async def _run_analyze_volume_trends(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    window = ctx.args.get("window", 20)
    data = ctx.fetch()

    trends = analyze_volume_trends(data, window)

    return {"symbol": symbol, "analysis": "Volume Trend Analysis", **trends}


@_with_session_metadata
async def _run_comprehensive_analysis(ctx: ToolContext) -> ComprehensiveAnalysis:
    return run_comprehensive_analysis(symbol=ctx.require_symbol(), data=ctx.fetch())


@_with_session_metadata
async def _run_options_analysis(ctx: ToolContext) -> OptionsAnalysis:
    symbol = ctx.require_symbol()
    holding_period = ctx.args.get("holding_period", 14)
    days_to_expiration = ctx.args.get("days_to_expiration", holding_period)
    data = ctx.fetch()

    return run_options_analysis(
        symbol=symbol,
        data=data,
        holding_period=holding_period,
        days_to_expiration=days_to_expiration,
    )


@_with_session_metadata
async def _run_scan_candidates(ctx: ToolContext) -> ScanResult:
    """scan_candidates fetches per symbol inside run_scan -- never via ctx.fetch()."""
    arguments = ctx.args
    holding_period = arguments.get("holding_period", 14)
    max_results = arguments.get("max_results", 15)
    cutoff = None if ctx.include_partial_bar else ctx.session.partial_bar_cutoff

    return await run_scan(
        symbols=arguments.get("symbols", []),
        universe=arguments.get("universe", "full_market"),
        period=arguments.get("period", "3mo"),
        holding_period=holding_period,
        min_score=arguments.get("min_score", 2.0),
        min_adx=arguments.get("min_adx", 20),
        max_iv_percentile=arguments.get("max_iv_percentile", 100),
        min_avg_daily_volume=arguments.get("min_avg_daily_volume", 0),
        direction=arguments.get("direction", "any"),
        max_results=max_results,
        data_source=ctx.data_source,
        as_of=cutoff,
    )


@_with_session_metadata
async def _run_calculate_rsi_divergence(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    rsi_period = ctx.args.get("rsi_period", 14)
    divergence_lookback = ctx.args.get("divergence_lookback", 10)
    data = ctx.fetch()

    divergence_result = calculate_rsi_divergence(data, rsi_period, divergence_lookback)
    return {"symbol": symbol, **divergence_result}


# --- The registry ------------------------------------------------------------

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
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
                "include_partial_bar": dict(_INCLUDE_PARTIAL_BAR_PROPERTY),
            },
            "required": ["symbol"],
        },
        run=_run_get_stock_data,
    ),
    ToolSpec(
        name="calculate_obv",
        description=(
            "Calculate On-Balance Volume (OBV) - cumulative volume "
            "indicator that adds volume on up days and subtracts on down days"
        ),
        input_schema=_single_symbol_schema(),
        run=_run_calculate_obv,
    ),
    ToolSpec(
        name="calculate_vwap",
        description=(
            "Calculate Volume Weighted Average Price (VWAP) - average "
            "price weighted by volume, used as a trading benchmark. "
            "It is anchored at the start of the fetched period."
        ),
        input_schema=_single_symbol_schema(),
        run=_run_calculate_vwap,
    ),
    ToolSpec(
        name="calculate_anchored_vwap",
        description=(
            "Calculate anchored VWAP from a confirmed swing pivot (three bars per side), "
            "maximum-volume bar, exact date, or period start. Returns the retrospective "
            "tail, current value, signed close distance in percent and ATR(14) units, "
            "and anchor date/type/typical price. Missing pivots or dates raise errors; "
            "undefined distances are null. Informational only."
        ),
        input_schema=_single_symbol_schema(
            anchor={
                "type": "string",
                "description": "Anchor used as the starting bar for VWAP",
                "enum": ["date", "swing_low", "swing_high", "max_volume", "period_start"],
                "default": "swing_low",
            },
            date={
                "type": "string",
                "description": "Exact bar date YYYY-MM-DD; required for date anchor",
            },
        ),
        run=_run_calculate_anchored_vwap,
    ),
    ToolSpec(
        name="calculate_volume_profile",
        description=(
            "Calculate Volume Profile - distribution of volume at "
            "different price levels, useful for identifying support/resistance"
        ),
        input_schema=_single_symbol_schema(
            num_bins={
                "type": "integer",
                "description": "Number of price levels to analyze (default: 20)",
                "default": 20,
                "minimum": 2,
                "maximum": 1000,
            },
        ),
        run=_run_calculate_volume_profile,
    ),
    ToolSpec(
        name="calculate_mfi",
        description=(
            "Calculate Money Flow Index (MFI) - volume-weighted RSI "
            "that oscillates 0-100, >80 overbought, <20 oversold"
        ),
        input_schema=_single_symbol_schema(
            mfi_period={
                "type": "integer",
                "description": "Lookback period for MFI calculation (default: 14)",
                "default": 14,
                "minimum": 1,
                "maximum": 200,
            },
        ),
        run=_run_calculate_mfi,
    ),
    ToolSpec(
        name="calculate_ad_line",
        description=(
            "Calculate Accumulation/Distribution Line (A/D Line) - measures "
            "cumulative flow of money into and out of a security"
        ),
        input_schema=_single_symbol_schema(),
        run=_run_calculate_ad_line,
    ),
    ToolSpec(
        name="calculate_cmf",
        description=(
            "Calculate Chaikin Money Flow (CMF) - measures buying and selling "
            "pressure over a set period (ranges -1 to +1). "
            "> 0 indicates buying, < 0 indicates selling"
        ),
        input_schema=_single_symbol_schema(
            cmf_period={
                "type": "integer",
                "description": "Lookback period for CMF calculation (default: 20)",
                "default": 20,
                "minimum": 1,
                "maximum": 200,
            },
        ),
        run=_run_calculate_cmf,
    ),
    ToolSpec(
        name="analyze_volume_trends",
        description="Analyze volume trends and detect price-volume divergences",
        input_schema=_single_symbol_schema(
            window={
                "type": "integer",
                "description": "Rolling window for trend analysis (default: 20)",
                "default": 20,
                "minimum": 1,
                "maximum": 200,
            },
        ),
        run=_run_analyze_volume_trends,
    ),
    ToolSpec(
        name="comprehensive_analysis",
        description=(
            "Perform comprehensive volume-price analysis including "
            "OBV, VWAP, MFI, and volume trends"
        ),
        input_schema=_single_symbol_schema(),
        run=_run_comprehensive_analysis,
        output_schema=COMPREHENSIVE_ANALYSIS_SCHEMA,
    ),
    ToolSpec(
        name="options_analysis",
        description=(
            "Perform comprehensive volume-price analysis optimized for "
            "options trading with 14-30 day holding periods. Includes "
            "ADX trend strength, RSI divergence detection, IV percentile, "
            "expected move calculations, and composite signal scoring. "
            "Automatically adapts indicator periods based on holding_period."
        ),
        input_schema=_single_symbol_schema(
            period_description=(
                "Period if dates not specified (default: '3mo' for sufficient historical context)"
            ),
            period_default="3mo",
            holding_period={
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
            days_to_expiration={
                "type": "integer",
                "description": (
                    "Days until options expiration for expected move "
                    "calculation (default: same as holding_period)"
                ),
                "minimum": 1,
                "maximum": 365,
            },
        ),
        run=_run_options_analysis,
        output_schema=OPTIONS_ANALYSIS_SCHEMA,
        default_period="3mo",
    ),
    ToolSpec(
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
                    "minimum": 0,
                    "default": 2.0,
                },
                "min_adx": {
                    "type": "number",
                    "description": "Minimum ADX for trend strength (default: 20)",
                    "minimum": 0,
                    "default": 20,
                },
                "max_iv_percentile": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": (
                        "Max volatility percentile (default: 100, use 50 for cheap "
                        "options). Note: this is a historical-volatility (HV) proxy, "
                        "not options-implied volatility; results expose both "
                        "iv_percentile (compat) and hv_percentile."
                    ),
                    "default": 100,
                },
                "min_avg_daily_volume": {
                    "type": "number",
                    "description": "Minimum average daily share volume (0 = no filter)",
                    "minimum": 0,
                    "default": 0,
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
                "include_partial_bar": dict(_INCLUDE_PARTIAL_BAR_PROPERTY),
            },
            "required": [],
        },
        run=_run_scan_candidates,
        output_schema=SCAN_RESULT_SCHEMA,
        default_period="3mo",
    ),
    ToolSpec(
        name="calculate_rsi_divergence",
        description=(
            "Detect causal RSI divergence at the latest bar using pivot-based analysis. "
            "Bullish divergence: price makes a lower low while RSI makes a higher low "
            "(potential reversal up). Bearish divergence: price makes a higher high while "
            "RSI makes a lower high (potential reversal down). Uses confirmed swing pivots "
            "only — no lookahead. Default period '3mo' to ensure enough pivots."
        ),
        input_schema=_single_symbol_schema(
            period_description=(
                "Period if dates not specified (default: '3mo'). "
                "At least 3mo recommended — 1mo rarely holds two confirmed pivots."
            ),
            period_default="3mo",
            rsi_period={
                "type": "integer",
                "description": "RSI calculation period (default: 14)",
                "default": 14,
                "minimum": 1,
                "maximum": 200,
            },
            divergence_lookback={
                "type": "integer",
                "description": "Minimum bars for divergence history gate (default: 10)",
                "default": 10,
                "minimum": 5,
                "maximum": 200,
            },
        ),
        run=_run_calculate_rsi_divergence,
        default_period="3mo",
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOLS}
