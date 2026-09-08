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

import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .analysis import build_headline, run_options_analysis, run_scan
from .data_fetcher import DataSource
from .indicators import (
    CMF_CONDITION_BANDS,
    CMF_PRESSURE_BANDS,
    MFI_CONDITION_BANDS,
    MFI_TOOL_CONDITION_BANDS,
    TREND_LOOKBACK,
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
    threshold_verdict,
    trend_verdict,
)

logger = logging.getLogger(__name__)


def validate_range(
    value: int | float, param_name: str, min_val: int | float, max_val: int | float
) -> None:
    """Validate that a parameter value is within the allowed range."""
    if value < min_val or value > max_val:
        msg = f"{param_name} must be between {min_val} and {max_val}, got {value}"
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
    fetched_frame: pd.DataFrame | None = field(default=None, init=False, repr=False)

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

    def fetch(self) -> pd.DataFrame:
        """Return the OHLCV frame for the standard symbol/period/start/end args."""
        if self.fetched_frame is None:
            symbol = self.require_symbol()
            self.fetched_frame = self.data_source.fetch(
                symbol,
                period=self.period,
                start=self.args.get("start_date"),
                end=self.args.get("end_date"),
            )
            logger.debug("Data fetched for %s: %d rows", symbol, len(self.fetched_frame))
        return self.fetched_frame.copy()


@dataclass(frozen=True)
class ToolSpec:
    """One MCP tool: how it is advertised and what it does."""

    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[ToolContext], Awaitable[Any]]
    default_period: str = "1mo"


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
            **extra_properties,
        },
        "required": ["symbol"],
    }


# --- Tool bodies -------------------------------------------------------------


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


async def _run_calculate_volume_profile(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    num_bins = ctx.args.get("num_bins", 20)
    validate_range(num_bins, "num_bins", 2, 1000)
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


async def _run_calculate_mfi(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    mfi_period = ctx.args.get("mfi_period", 14)
    validate_range(mfi_period, "mfi_period", 1, 200)
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


async def _run_calculate_cmf(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    cmf_period = ctx.args.get("cmf_period", 20)
    validate_range(cmf_period, "cmf_period", 1, 200)
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


async def _run_analyze_volume_trends(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    window = ctx.args.get("window", 20)
    validate_range(window, "window", 1, 200)
    data = ctx.fetch()

    trends = analyze_volume_trends(data, window)

    return {"symbol": symbol, "analysis": "Volume Trend Analysis", **trends}


async def _run_comprehensive_analysis(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    data = ctx.fetch()

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
    obv_trend = trend_verdict(obv, TREND_LOOKBACK)
    ad_trend = trend_verdict(ad_line, TREND_LOOKBACK)
    obv_flow = "into" if obv_trend == "increasing" else "out of"
    ad_action = "buying" if ad_trend == "increasing" else "selling"
    mfi_val = mfi.iloc[-1]
    cmf_val = cmf.iloc[-1]

    mfi_condition = threshold_verdict(mfi_val, MFI_CONDITION_BANDS, "Neutral")
    cmf_condition = threshold_verdict(cmf_val, CMF_CONDITION_BANDS, "Neutral")

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

    return {
        "symbol": symbol,
        "analysis_type": "Comprehensive Volume-Price Analysis",
        "period": f"{start_dt} to {end_dt}",
        "latest_price": float(latest_close),
        "headline": headline,
        "volume_indicators": {
            "obv": {
                "value": float(obv.iloc[-1]),
                "trend": obv_trend,
                "interpretation": f"Money flowing {obv_flow} the security",
            },
            "accumulation_distribution": {
                "value": float(ad_line.iloc[-1]),
                "trend": ad_trend,
                "interpretation": f"Institutional {ad_action} pressure",
            },
            "vpt": {
                "value": float(vpt.iloc[-1]),
                "trend": trend_verdict(vpt, TREND_LOOKBACK),
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


async def _run_options_analysis(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    holding_period = ctx.args.get("holding_period", 14)
    validate_range(holding_period, "holding_period", 1, 90)
    days_to_expiration = ctx.args.get("days_to_expiration", holding_period)
    validate_range(days_to_expiration, "days_to_expiration", 1, 365)
    data = ctx.fetch()

    return run_options_analysis(
        symbol=symbol,
        data=data,
        holding_period=holding_period,
        days_to_expiration=days_to_expiration,
    )


async def _run_scan_candidates(ctx: ToolContext) -> dict:
    """scan_candidates fetches per symbol inside run_scan -- never via ctx.fetch()."""
    arguments = ctx.args
    holding_period = arguments.get("holding_period", 14)
    validate_range(holding_period, "holding_period", 1, 90)
    max_results = arguments.get("max_results", 15)
    validate_range(max_results, "max_results", 1, 100)

    return await run_scan(
        symbols=arguments.get("symbols", []),
        universe=arguments.get("universe", "full_market"),
        period=arguments.get("period", "3mo"),
        holding_period=holding_period,
        min_score=arguments.get("min_score", 2.0),
        min_adx=arguments.get("min_adx", 20),
        max_iv_percentile=arguments.get("max_iv_percentile", 100),
        direction=arguments.get("direction", "any"),
        max_results=max_results,
        data_source=ctx.data_source,
    )


async def _run_calculate_rsi_divergence(ctx: ToolContext) -> dict:
    symbol = ctx.require_symbol()
    rsi_period = ctx.args.get("rsi_period", 14)
    validate_range(rsi_period, "rsi_period", 1, 200)
    divergence_lookback = ctx.args.get("divergence_lookback", 10)
    validate_range(divergence_lookback, "divergence_lookback", 5, 200)
    data = ctx.fetch()

    divergence_result = calculate_rsi_divergence(data, rsi_period, divergence_lookback)
    return {"symbol": symbol, **divergence_result}


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
    obv_trend = trend_verdict(obv, TREND_LOOKBACK)
    ad_trend = trend_verdict(ad_line, TREND_LOOKBACK)
    if obv_trend == "increasing" and ad_trend == "increasing":
        summary.append("✓ Strong accumulation - Both OBV and A/D Line rising")
    elif obv_trend == "decreasing" and ad_trend == "decreasing":
        summary.append("⚠️  Strong distribution - Both OBV and A/D Line falling")
    else:
        summary.append("⚠️  Mixed volume signals - OBV and A/D Line diverging")

    # Money Flow
    mfi_condition = threshold_verdict(mfi.iloc[-1], MFI_CONDITION_BANDS, "Neutral")
    cmf_condition = threshold_verdict(cmf.iloc[-1], CMF_CONDITION_BANDS, "Neutral")
    if mfi_condition == "Overbought" or cmf_condition == "Strong buying":
        summary.append("⚠️  Overbought conditions detected - Potential reversal risk")
    elif mfi_condition == "Oversold" or cmf_condition == "Strong selling":
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
            "price weighted by volume, used as a trading benchmark"
        ),
        input_schema=_single_symbol_schema(),
        run=_run_calculate_vwap,
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
        run=_run_scan_candidates,
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
