"""Reusable analysis functions for scanning and options analysis.

This module extracts the core analysis logic from server.py so it can be
used by both the MCP server and the morning agent without duplication.
"""

import asyncio
import logging

import pandas as pd
from pytickersymbols import PyTickerSymbols

from .data_fetcher import fetch_stock_data
from .indicators import (
    SQUEEZE_WINDOW,
    analyze_volume_trends,
    calculate_accumulation_distribution,
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_chaikin_money_flow,
    calculate_composite_score,
    calculate_enhanced_volume_profile,
    calculate_expected_move,
    calculate_iv_percentile,
    calculate_mfi,
    calculate_obv,
    calculate_price_roc,
    calculate_relative_volume,
    calculate_rsi_with_divergence,
    calculate_vpt,
    calculate_vwap,
    calculate_vwma,
    composite_adx_period,
    detect_bollinger_squeeze,
    detect_volume_breakout,
)

logger = logging.getLogger(__name__)

# Concurrency limit for parallel scanning
MAX_CONCURRENT_SCANS = 10

# Minimum bars of history required to analyze a symbol in a scan.
MIN_SCAN_HISTORY = 30


class InsufficientDataError(Exception):
    """Raised when a symbol has too little history to analyze.

    This is a *skip* signal, not a failure: the symbol is excluded from the scan
    and counted separately from genuine fetch/calculation errors so the scan's
    diagnostics stay honest about coverage gaps.
    """


def _build_sp500_symbols() -> list[str]:
    """Build S&P 500 symbol list dynamically from pytickersymbols (bundled data, no network)."""
    try:
        pts = PyTickerSymbols()
        symbols: list[str] = []
        for stock in pts.get_stocks_by_index("S&P 500"):
            # Prefer the USD-denominated Yahoo symbol (US exchange)
            yahoo_sym = None
            for sym_info in stock.get("symbols", []):
                if sym_info.get("yahoo") and sym_info.get("currency") == "USD":
                    yahoo_sym = sym_info["yahoo"]
                    break
            # Fallback to first available Yahoo symbol
            if not yahoo_sym:
                for sym_info in stock.get("symbols", []):
                    if sym_info.get("yahoo"):
                        yahoo_sym = sym_info["yahoo"]
                        break
            if yahoo_sym:
                symbols.append(yahoo_sym)
        return sorted(set(symbols))
    except Exception:
        logger.warning(
            "Failed to load S&P 500 symbols from pytickersymbols, using fallback symbols",
            exc_info=True,
        )
        return ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]


_ETF_LIST: list[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "EEM",
    "VTI",
    "VOO",
    "VEA",
    "VWO",
    "GLD",
    "SLV",
    "USO",
    "XLF",
    "XLE",
    "XLK",
    "XLV",
    "XLI",
    "XLP",
    "XLY",
    "XLB",
    "XLU",
    "XLRE",
    "XLC",
    "VNQ",
    "HYG",
    "LQD",
    "TLT",
    "IEF",
    "SHY",
    "BND",
    "ARKK",
    "ARKG",
    "ARKW",
    "ARKF",
    "ARKQ",
    "SMH",
    "SOXX",
    "IBB",
    "XBI",
    "KRE",
    "TQQQ",
    "SQQQ",
    "KWEB",
    "FXI",
    "CIBR",
    "TAN",
    "PAVE",
    "SCHD",
    "VIG",
    "UVXY",
]

# Pre-built symbol universes for scanning
UNIVERSES: dict[str, list[str]] = {
    "sp500": _build_sp500_symbols(),
    "etfs": _ETF_LIST,
}
UNIVERSES["full_market"] = sorted(set(UNIVERSES["sp500"] + UNIVERSES["etfs"]))

# Human-readable labels for composite-score recommendation values.
_RECOMMENDATION_LABELS = {
    "strong_bullish": "Strong bullish",
    "bullish": "Bullish",
    "neutral": "Neutral",
    "bearish": "Bearish",
    "strong_bearish": "Strong bearish",
}

# Maps composite score_breakdown keys to short labels used in the headline
# rationale. Only signals present here are named, so the rationale never invents
# a driver that the composite did not actually score.
_DRIVER_LABELS = {
    "price_vs_vwap": "price vs VWAP",
    "price_vs_vwma": "price vs VWMA",
    "obv_momentum": "OBV momentum",
    "ad_momentum": "A/D line",
    "mfi": "MFI",
    "cmf": "CMF",
    "rsi": "RSI",
    "rsi_divergence": "RSI divergence",
    "adx_direction": "ADX trend",
    "volume_breakout": "volume breakout",
}

# Below this absolute composite score the call is treated as directionally
# neutral (matches the +/-2 recommendation boundary in calculate_composite_score).
_NEUTRAL_SCORE_BAND = 2.0


def _build_rationale(score: float, label: str, signal_quality: str, breakdown: dict) -> str:
    """Compose a one-line, data-grounded rationale for the headline.

    Names the strongest score_breakdown components that point the same direction
    as the overall score. Only signals actually present in the breakdown are
    cited, so the sentence stays faithful to the underlying scoring.
    """
    if abs(score) < _NEUTRAL_SCORE_BAND:
        return f"{label} (score {score:+.1f}/10): mixed signals, no clear directional edge."

    direction = "bullish" if score > 0 else "bearish"
    sign = 1 if score > 0 else -1
    aligned = [
        (key, value)
        for key, value in breakdown.items()
        if key in _DRIVER_LABELS and value * sign > 0
    ]
    # Strongest contributors first; stable label-order tiebreak for determinism.
    aligned.sort(key=lambda kv: (-abs(kv[1]), kv[0]))
    drivers = [_DRIVER_LABELS[key] for key, _ in aligned[:3]]

    if drivers:
        return (
            f"{label} (score {score:+.1f}/10, {signal_quality} conviction): "
            f"{', '.join(drivers)} aligned {direction}."
        )
    return (
        f"{label} (score {score:+.1f}/10, {signal_quality} conviction): "
        f"driven by aggregate volume-price signals."
    )


def build_headline(composite: dict) -> dict:
    """Build a compact top-line headline from a composite-score result.

    Additive summary for MCP tool responses (comprehensive_analysis,
    options_analysis) and the briefing projection. Surfaces the recommendation,
    score, signal quality, and a single grounded rationale sentence so a consumer
    gets the bottom line without parsing the full nested analysis.

    Args:
        composite: The dict returned by ``calculate_composite_score``.

    Returns:
        ``{recommendation, composite_score, signal_quality, rationale}``.
    """
    score = float(composite.get("composite_score", 0.0))
    recommendation = composite.get("recommendation", "neutral")
    signal_quality = composite.get("signal_quality", "low")
    label = _RECOMMENDATION_LABELS.get(recommendation, recommendation.replace("_", " ").title())
    breakdown = composite.get("score_breakdown") or {}

    return {
        "recommendation": recommendation,
        "composite_score": round(score, 2),
        "signal_quality": signal_quality,
        "rationale": _build_rationale(score, label, signal_quality, breakdown),
    }


def analyze_single_symbol(
    symbol: str,
    period: str,
    holding_period: int,
    min_score: float,
    min_adx: float,
    max_iv: float,
    direction: str,
    min_avg_volume: float = 0,
) -> dict | None:
    """
    Analyze a single symbol for scan_candidates.

    Returns a candidate dict if it passes filters, None if it was analyzed but
    did not qualify. Raises ``InsufficientDataError`` when there is too little
    history to analyze (a skip), and propagates other exceptions as errors.
    """
    sym_data = fetch_stock_data(symbol, None, None, period)
    if len(sym_data) < MIN_SCAN_HISTORY:
        raise InsufficientDataError(f"{symbol}: {len(sym_data)} bars < {MIN_SCAN_HISTORY} required")

    # Calculate composite score and key indicators
    composite = calculate_composite_score(sym_data, holding_period)
    # Reuse the ADX the composite already computed (adaptive to holding_period) so the
    # reported adx, the min_adx filter, and signal_quality all reference the same value
    # instead of a separate, fixed-period ADX(14). See HOM-48.
    adx_summary = composite["adx_summary"]
    iv_pct_data = calculate_iv_percentile(sym_data, 20)
    expected_move = calculate_expected_move(sym_data, holding_period, 20)
    rsi_data = calculate_rsi_with_divergence(sym_data, 14, 10)
    rvol = calculate_relative_volume(sym_data, 20)

    score = composite["composite_score"]
    adx = adx_summary["adx"]
    iv_pct = iv_pct_data["iv_percentile"]
    avg_volume = float(sym_data["Volume"].mean())

    # Apply filters
    passes_score = abs(score) >= min_score
    passes_adx = adx >= min_adx
    passes_iv = iv_pct <= max_iv
    passes_volume = avg_volume >= min_avg_volume if min_avg_volume > 0 else True

    if direction == "bullish":
        passes_direction = score > 0
    elif direction == "bearish":
        passes_direction = score < 0
    else:
        passes_direction = True

    if not (passes_score and passes_adx and passes_direction and passes_iv and passes_volume):
        return None

    return {
        "symbol": symbol,
        "composite_score": round(score, 2),
        "recommendation": composite["recommendation"],
        "signal_quality": composite["signal_quality"],
        "adx": round(adx, 1),
        "adx_period": adx_summary["period"],
        "trend_strength": adx_summary["trend_strength"],
        "trend_direction": adx_summary["trend_direction"],
        "rsi": round(rsi_data["rsi"], 1),
        "rsi_divergence": rsi_data["divergence_type"],
        # iv_percentile kept for backward compat; hv_percentile is the honest name
        # (the value is a historical-volatility proxy, not options-implied vol).
        "iv_percentile": round(iv_pct, 1),
        "hv_percentile": round(iv_pct, 1),
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
    min_avg_volume: float = 0,
) -> tuple[str, dict | None, str | None, bool]:
    """
    Async wrapper for symbol analysis with concurrency limiting.

    Returns (symbol, candidate_or_none, error_or_none, skipped). ``skipped`` is
    True when the symbol had insufficient history (a non-error skip).
    """
    async with semaphore:
        try:
            result = await asyncio.to_thread(
                analyze_single_symbol,
                symbol,
                period,
                holding_period,
                min_score,
                min_adx,
                max_iv,
                direction,
                min_avg_volume,
            )
            return (symbol, result, None, False)
        except InsufficientDataError:
            return (symbol, None, None, True)
        except Exception as e:
            return (symbol, None, str(e), False)


async def run_scan(
    symbols: list[str] | None = None,
    universe: str = "full_market",
    period: str = "3mo",
    holding_period: int = 14,
    min_score: float = 2.0,
    min_adx: float = 20,
    max_iv_percentile: float = 100,
    direction: str = "any",
    max_results: int = 15,
    max_concurrent: int = MAX_CONCURRENT_SCANS,
    min_avg_daily_volume: float = 0,
) -> dict:
    """
    Scan the market for options trading candidates.

    Args:
        symbols: Custom list of symbols (overrides universe if provided).
        universe: Pre-built universe name. Valid values: "sp500", "etfs", "full_market".
        period: Historical data period for analysis.
        holding_period: Expected options holding period in days.
        min_score: Minimum |composite_score| to include.
        min_adx: Minimum ADX for trend strength.
        max_iv_percentile: Maximum IV percentile filter.
        direction: "bullish", "bearish", or "any".
        max_results: Maximum results per direction.
        max_concurrent: Maximum concurrent symbol analyses.
        min_avg_daily_volume: Minimum average daily share volume (0 = no filter).

    Returns:
        Dictionary with scan results including candidates, summary, and errors.
    """
    direction = direction.lower()
    valid_directions = ("bullish", "bearish", "any")
    if direction not in valid_directions:
        raise ValueError(
            f"Invalid direction '{direction}'. Must be one of: {', '.join(valid_directions)}"
        )

    # Determine symbols to scan
    if symbols and len(symbols) > 0:
        if len(symbols) > 500:
            raise ValueError(f"Too many symbols ({len(symbols)}). Maximum is 500.")
        scan_symbols = [s.upper() for s in symbols]
        universe_used = "custom"
    elif universe.lower() in UNIVERSES:
        scan_symbols = UNIVERSES[universe.lower()]
        universe_used = universe.lower()
    else:
        logger.warning(
            "Unknown universe %r; falling back to 'full_market'. Valid values: %s",
            universe,
            ", ".join(sorted(UNIVERSES)),
        )
        scan_symbols = UNIVERSES["full_market"]
        universe_used = "full_market"

    # Parallel scanning with concurrency limit
    logger.info(
        "Starting parallel scan of %d symbols (max concurrent: %d)",
        len(scan_symbols),
        max_concurrent,
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        _analyze_symbol_async(
            sym,
            period,
            holding_period,
            min_score,
            min_adx,
            max_iv_percentile,
            direction,
            semaphore,
            min_avg_daily_volume,
        )
        for sym in scan_symbols
    ]

    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=600)
    except TimeoutError:
        logger.error("Scan timed out after 600 seconds")
        raise ValueError("Scan timed out after 10 minutes") from None

    # Process results. Each symbol falls into exactly one bucket:
    #   - error: a genuine fetch/calculation failure
    #   - skipped: insufficient history to analyze (not a failure)
    #   - scanned: successfully analyzed (may or may not become a candidate)
    candidates = []
    errors = []
    scanned = 0
    skipped = 0

    for sym, candidate, error, was_skipped in results:
        if error:
            errors.append({"symbol": sym, "error": error})
        elif was_skipped:
            skipped += 1
        else:
            scanned += 1
            if candidate is not None:
                candidates.append(candidate)

    logger.info(
        "Scan complete: %d candidates from %d scanned (%d skipped, %d errors)",
        len(candidates),
        scanned,
        skipped,
        len(errors),
    )

    # Sort by absolute composite score (highest first)
    candidates.sort(key=lambda x: abs(x["composite_score"]), reverse=True)

    # Separate into bullish and bearish (zero-score goes to bullish)
    bullish = [c for c in candidates if c["composite_score"] >= 0]
    bearish = [c for c in candidates if c["composite_score"] < 0]

    # Find highest conviction setups. NOTE: c["adx"] is the composite's adaptive-period
    # ADX (ADX(10) for holding_period<=14, else ADX(14)) -- coherent with min_adx and
    # signal_quality. The 28 gate is read against that period; adx_period is reported in
    # scan_parameters so clients can interpret it. See HOM-48.
    high_conviction = [
        c
        for c in candidates
        if abs(c["composite_score"]) >= 4 and c["adx"] >= 28 and c["iv_percentile"] <= 50
    ]

    return {
        "scan_parameters": {
            "universe": universe_used,
            "symbols_in_universe": len(scan_symbols),
            "symbols_scanned": scanned,
            "holding_period": holding_period,
            "min_score": min_score,
            "min_adx": min_adx,
            "max_iv_percentile": max_iv_percentile,
            "min_avg_daily_volume": min_avg_daily_volume,
            "direction_filter": direction,
            # ADX lookback backing the reported `adx`, the min_adx filter, and the
            # high_conviction gate -- adaptive to holding_period (HOM-48).
            "adx_period": composite_adx_period(holding_period),
            # iv_percentile / hv_percentile are an HV-based proxy, not options
            # implied volatility. See indicators.calculate_iv_percentile.
            "volatility_basis": "historical_volatility",
        },
        "summary": {
            "total_candidates": len(candidates),
            "bullish_setups": len(bullish),
            "bearish_setups": len(bearish),
            "high_conviction": len(high_conviction),
            # Symbols skipped for insufficient history, distinct from errors.
            "skipped": skipped,
            "errors": len(errors),
        },
        "high_conviction_setups": high_conviction[:5] if high_conviction else [],
        "top_bullish": bullish[:max_results] if bullish else [],
        "top_bearish": bearish[:max_results] if bearish else [],
        # Always a list (capped at 10) so consumers never special-case None.
        "errors": errors[:10],
    }


def run_options_analysis(
    symbol: str,
    data: pd.DataFrame,
    holding_period: int = 14,
    days_to_expiration: int | None = None,
) -> dict:
    """
    Run comprehensive options analysis on a single symbol.

    Args:
        symbol: Stock ticker symbol.
        data: DataFrame with OHLCV data (must already be fetched).
        holding_period: Expected options holding period in days.
        days_to_expiration: Days until options expiration (defaults to holding_period).

    Returns:
        Dictionary with full options analysis results.
    """
    if data.empty:
        raise ValueError(f"Empty DataFrame provided for symbol '{symbol}'")
    required_columns = {"Open", "High", "Low", "Close", "Volume", "Date"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {', '.join(sorted(missing))}")

    if days_to_expiration is None:
        days_to_expiration = holding_period

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

    # Enhanced indicators
    adx_data = calculate_adx(data, adx_period)
    rsi_data = calculate_rsi_with_divergence(data, rsi_period, volume_window)
    iv_percentile = calculate_iv_percentile(data, hv_window)
    expected_move = calculate_expected_move(data, days_to_expiration, hv_window)
    composite = calculate_composite_score(data, holding_period)

    # Volatility indicators
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

    # The squeeze verdict deliberately ignores the holding-period-adaptive
    # `bbands` (which still drive the displayed band levels); see
    # detect_bollinger_squeeze.
    is_squeeze = detect_bollinger_squeeze(data)

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

    return {
        "symbol": symbol,
        "analysis_type": f"Options Trading ({holding_period}-Day Optimized)",
        "period": f"{start_dt} to {end_dt}",
        "latest_price": float(latest_close),
        # Additive top-line summary (recommendation/score/1-line rationale).
        "headline": build_headline(composite),
        "parameters": {
            "holding_period": holding_period,
            "days_to_expiration": days_to_expiration,
            "mfi_period": mfi_period,
            "volume_window": volume_window,
            "rsi_period": rsi_period,
            "adx_period": adx_period,
            "hv_window": hv_window,
            "squeeze_window": SQUEEZE_WINDOW,
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
                "trend": (
                    "unknown"
                    if len(vpt) < 4
                    else ("increasing" if vpt.iloc[-1] > vpt.iloc[-3] else "decreasing")
                ),
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
                "signal": "bullish_entry" if latest_close > latest_vwap else "bearish_entry",
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
                "hv_percentile": iv_percentile["hv_percentile"],
                "basis": iv_percentile["basis"],
                "is_proxy": iv_percentile["is_proxy"],
                "current_hv": iv_percentile["current_hv"],
                "hv_range": f"{iv_percentile['hv_min']:.1%} - {iv_percentile['hv_max']:.1%}",
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
        "options_insights": _generate_options_insights(
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


def _generate_options_insights(
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
            f"STRONG BULLISH: Composite score {score:.1f}/10 - High conviction call setup"
        )
    elif score >= 2:
        insights.append(
            f"BULLISH: Composite score {score:.1f}/10 - Consider call options or bull spreads"
        )
    elif score <= -5:
        insights.append(
            f"STRONG BEARISH: Composite score {score:.1f}/10 - High conviction put setup"
        )
    elif score <= -2:
        insights.append(
            f"BEARISH: Composite score {score:.1f}/10 - Consider put options or bear spreads"
        )
    else:
        insights.append(
            f"NEUTRAL: Composite score {score:.1f}/10 - "
            f"No clear directional edge, consider iron condors or wait"
        )

    # 2. Trend Quality Assessment
    adx = adx_data["adx"]
    trend_dir = adx_data["trend_direction"]
    if adx > 30:
        insights.append(
            f"STRONG TREND: ADX at {adx:.1f} ({trend_dir}) - "
            f"Directional plays have wind at their back"
        )
    elif adx > 25:
        insights.append(
            f"Moderate Trend: ADX at {adx:.1f} ({trend_dir}) - Decent setup for directional options"
        )
    elif adx > 20:
        insights.append(
            f"Weak Trend: ADX at {adx:.1f} - Consider reduced position size or neutral strategies"
        )
    else:
        insights.append(
            f"NO TREND: ADX at {adx:.1f} - Premium selling (iron condors, strangles) preferred"
        )

    # 3. RSI Divergence Alert
    if rsi_data["divergence_type"] == "bullish":
        insights.append(
            "RSI BULLISH DIVERGENCE: Price weakness not confirmed by momentum - "
            "Potential reversal up, favor calls"
        )
    elif rsi_data["divergence_type"] == "bearish":
        insights.append(
            "RSI BEARISH DIVERGENCE: Price strength not confirmed by momentum - "
            "Potential reversal down, favor puts"
        )

    # 4. HV Percentile / Volatility Edge (HV proxy, not options-implied vol)
    hv_pct = iv_percentile["hv_percentile"]
    if hv_pct > 80:
        insights.append(
            f"HIGH HV PERCENTILE ({hv_pct:.0f}%, HV proxy): Options likely EXPENSIVE - "
            f"Favor selling premium (credit spreads, iron condors)"
        )
    elif hv_pct > 60:
        insights.append(
            f"HV slightly elevated ({hv_pct:.0f}%, HV proxy) - "
            f"Consider debit spreads to reduce vega risk"
        )
    elif hv_pct < 20:
        insights.append(
            f"LOW HV PERCENTILE ({hv_pct:.0f}%, HV proxy): Options likely CHEAP - "
            f"Great time for long options, straddles, or strangles"
        )
    elif hv_pct < 40:
        insights.append(
            f"Below-average HV ({hv_pct:.0f}%, HV proxy) - "
            f"Long directional plays are reasonably priced"
        )

    # 5. Expected Move Guidance
    em_pct = expected_move["expected_move_percent"]
    em_upper = expected_move["upper_target_1std"]
    em_lower = expected_move["lower_target_1std"]
    insights.append(
        f"Expected Move: +/-{em_pct:.1f}% by expiration - "
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
            "BOLLINGER SQUEEZE DETECTED: Volatility compressed - "
            "Breakout imminent! Consider straddles or wait for direction"
        )

    if bb_pct_b is not None and not pd.isna(bb_pct_b):
        if bb_pct_b > 0.95:
            insights.append(
                f"Price at upper Bollinger Band ({bb_pct_b:.0%}) - "
                f"Extended, consider puts or profit-taking on calls"
            )
        elif bb_pct_b < 0.05:
            insights.append(
                f"Price at lower Bollinger Band ({bb_pct_b:.0%}) - "
                f"Oversold, consider calls or profit-taking on puts"
            )

    # 8. Volume Conviction
    if breakout["is_breakout"]:
        direction = breakout["direction"]
        mult = breakout["multiplier_above_avg"]
        insights.append(
            f"VOLUME BREAKOUT: {mult:.1f}x average ({direction}) - "
            f"Strong conviction behind the move"
        )
    elif rvol["current_rvol"] > 1.5:
        insights.append(
            f"High volume ({rvol['current_rvol']:.1f}x average) - "
            f"Institutional participation detected"
        )
    elif rvol["current_rvol"] < 0.7:
        insights.append("Low volume - Wait for volume confirmation before entry")

    # 9. Divergence Warning
    if trends["divergence_detected"]:
        insights.append(
            f"PRICE-VOLUME DIVERGENCE: {trends['divergence_type']} - Current trend may be weakening"
        )

    # 10. MFI/CMF Extremes
    if mfi_val > 80 and cmf_val > 0.2:
        insights.append(
            "EXTREME OVERBOUGHT: MFI + CMF both elevated - "
            "High reversal risk, protect call profits or consider puts"
        )
    elif mfi_val < 20 and cmf_val < -0.2:
        insights.append(
            "EXTREME OVERSOLD: MFI + CMF both depressed - "
            "Bounce potential high, consider calls for mean reversion"
        )

    # 11. Holding Period Reminder
    if holding_period <= 14:
        insights.append(
            f"{holding_period}-day holding period: Using fast indicators - "
            f"Execute quickly, manage theta aggressively"
        )
    elif holding_period <= 21:
        insights.append(
            f"{holding_period}-day holding period: Balanced approach - "
            f"Monitor daily, theta decay moderate"
        )
    else:
        insights.append(
            f"{holding_period}-day holding period: Standard indicators - "
            f"Can weather short-term volatility, theta decay manageable"
        )

    return insights
