"""Volume-Price analysis indicators and calculations."""

from typing import Any

import numpy as np
import pandas as pd


def calculate_obv(data: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    OBV is a cumulative indicator that adds volume on up days and subtracts
    volume on down days. It helps identify whether volume is flowing into
    or out of a security.

    Args:
        data: DataFrame with 'Close' and 'Volume' columns

    Returns:
        Series containing OBV values
    """
    obv = [0]
    for i in range(1, len(data)):
        if data["Close"].iloc[i] > data["Close"].iloc[i - 1]:
            obv.append(obv[-1] + data["Volume"].iloc[i])
        elif data["Close"].iloc[i] < data["Close"].iloc[i - 1]:
            obv.append(obv[-1] - data["Volume"].iloc[i])
        else:
            obv.append(obv[-1])

    return pd.Series(obv, index=data.index)


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).

    VWAP is the average price weighted by volume. It's used to identify
    the true average price and is often used as a trading benchmark.

    Args:
        data: DataFrame with 'High', 'Low', 'Close', and 'Volume' columns

    Returns:
        Series containing VWAP values
    """
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    vwap = (typical_price * data["Volume"]).cumsum() / data["Volume"].cumsum()
    return vwap


def calculate_volume_profile(data: pd.DataFrame, num_bins: int = 20) -> dict[str, list]:
    """
    Calculate Volume Profile - distribution of volume at different price levels.

    This shows which price levels had the most trading activity, helping identify
    support/resistance levels and areas of high liquidity.

    Args:
        data: DataFrame with 'Close' and 'Volume' columns
        num_bins: Number of price bins to create

    Returns:
        Dictionary with 'price_levels' and 'volumes' lists
    """
    min_price = data["Low"].min()
    max_price = data["High"].max()

    # Create price bins
    bins = np.linspace(min_price, max_price, num_bins + 1)
    price_levels = (bins[:-1] + bins[1:]) / 2  # Midpoint of each bin

    # Vectorized volume profile calculation
    lows = data["Low"].values
    highs = data["High"].values
    vols = data["Volume"].values

    # Get bin indices for lows and highs (clip to valid range)
    low_bins = np.clip(np.digitize(lows, bins) - 1, 0, num_bins - 1)
    high_bins = np.clip(np.digitize(highs, bins) - 1, 0, num_bins - 1)

    # Calculate bins covered and volume per bin for each candle
    bins_covered = np.maximum(1, high_bins - low_bins + 1)
    volume_per_bin = vols / bins_covered

    # Aggregate volume for each price bin using vectorized accumulation
    volumes = np.zeros(num_bins)

    # For candles spanning single bins (most common case), use bincount
    single_bin_mask = low_bins == high_bins
    if np.any(single_bin_mask):
        single_volumes = np.bincount(
            low_bins[single_bin_mask],
            weights=volume_per_bin[single_bin_mask],
            minlength=num_bins,
        )
        volumes += single_volumes[:num_bins]

    # For candles spanning multiple bins, use loop (typically fewer iterations)
    multi_bin_indices = np.where(~single_bin_mask)[0]
    for idx in multi_bin_indices:
        start_bin = low_bins[idx]
        end_bin = high_bins[idx]
        volumes[start_bin : end_bin + 1] += volume_per_bin[idx]

    return {"price_levels": price_levels.tolist(), "volumes": volumes.tolist()}


def calculate_vpt(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume-Price Trend (VPT).

    VPT is similar to OBV but uses percentage price changes.
    It's more sensitive to the magnitude of price movements.

    Args:
        data: DataFrame with 'Close' and 'Volume' columns

    Returns:
        Series containing VPT values
    """
    vpt = [0]
    for i in range(1, len(data)):
        prev_close = data["Close"].iloc[i - 1]
        curr_close = data["Close"].iloc[i]
        price_change_pct = (curr_close - prev_close) / prev_close
        vpt.append(vpt[-1] + data["Volume"].iloc[i] * price_change_pct)

    return pd.Series(vpt, index=data.index)


def calculate_mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index (MFI).

    MFI is a volume-weighted version of RSI. It oscillates between 0 and 100,
    with readings above 80 indicating overbought conditions and below 20
    indicating oversold conditions.

    Args:
        data: DataFrame with 'High', 'Low', 'Close', and 'Volume' columns
        period: Lookback period for MFI calculation

    Returns:
        Series containing MFI values
    """
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    money_flow = typical_price * data["Volume"]

    # Determine positive and negative money flow
    positive_flow = []
    negative_flow = []

    for i in range(len(data)):
        if i == 0:
            positive_flow.append(0)
            negative_flow.append(0)
        elif typical_price.iloc[i] > typical_price.iloc[i - 1]:
            positive_flow.append(money_flow.iloc[i])
            negative_flow.append(0)
        elif typical_price.iloc[i] < typical_price.iloc[i - 1]:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)

    positive_flow = pd.Series(positive_flow, index=data.index)
    negative_flow = pd.Series(negative_flow, index=data.index)

    # Calculate Money Flow Ratio and MFI
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    mfr = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + mfr))

    return mfi


def analyze_volume_trends(data: pd.DataFrame, window: int = 20) -> dict[str, Any]:
    """
    Analyze volume trends and provide insights.

    Args:
        data: DataFrame with 'Volume' and 'Close' columns
        window: Rolling window for average calculations

    Returns:
        Dictionary with volume trend analysis
    """
    avg_volume = data["Volume"].rolling(window=window).mean()
    current_volume = data["Volume"].iloc[-1]
    current_avg = avg_volume.iloc[-1]

    # Calculate volume trend
    volume_increasing = data["Volume"].iloc[-5:].is_monotonic_increasing

    # Calculate price-volume divergence
    price_direction = "up" if data["Close"].iloc[-1] > data["Close"].iloc[-window] else "down"
    volume_direction = "up" if current_volume > current_avg else "down"

    divergence = (price_direction == "up" and volume_direction == "down") or (
        price_direction == "down" and volume_direction == "up"
    )

    divergence_type = (
        f"Price {price_direction}, Volume {volume_direction}" if divergence else "None"
    )

    return {
        "current_volume": int(current_volume),
        "average_volume": int(current_avg),
        "volume_vs_average": f"{((current_volume / current_avg - 1) * 100):.2f}%",
        "volume_trend": "increasing" if volume_increasing else "decreasing",
        "price_direction": price_direction,
        "divergence_detected": divergence,
        "divergence_type": divergence_type,
    }


# ============================================================================
# VOLATILITY INDICATORS (Critical for Options Trading)
# ============================================================================


def calculate_historical_volatility(
    data: pd.DataFrame, window: int = 20, annualize: bool = True
) -> pd.Series:
    """
    Calculate Historical Volatility (HV) / Realized Volatility.

    HV measures the actual price volatility over a historical period.
    Essential for options traders to compare against Implied Volatility.

    Args:
        data: DataFrame with 'Close' column
        window: Lookback period in days
        annualize: If True, annualizes the volatility (assuming 252 trading days)

    Returns:
        Series containing historical volatility values
    """
    # Calculate log returns
    log_returns = (data["Close"] / data["Close"].shift(1)).apply(np.log)

    # Calculate rolling standard deviation
    volatility = log_returns.rolling(window=window).std()

    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(252)

    return volatility


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range of price
    movement. Critical for position sizing and stop-loss placement in options.

    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns
        period: Lookback period for ATR

    Returns:
        Series containing ATR values
    """
    # Calculate True Range
    high_low = data["High"] - data["Low"]
    high_close = abs(data["High"] - data["Close"].shift(1))
    low_close = abs(data["Low"] - data["Close"].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate ATR as moving average of True Range
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_bollinger_bands(
    data: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.

    Bollinger Bands help identify overbought/oversold conditions and
    potential breakouts. Useful for timing options entries and identifying
    volatility contraction (squeeze).

    Args:
        data: DataFrame with 'Close' column
        period: Moving average period
        num_std: Number of standard deviations for bands

    Returns:
        Dictionary with 'upper', 'middle', 'lower' bands and 'bandwidth'
    """
    middle_band = data["Close"].rolling(window=period).mean()
    std = data["Close"].rolling(window=period).std()

    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)

    # Bandwidth: measure of volatility (useful for identifying squeezes)
    bandwidth = (upper_band - lower_band) / middle_band

    return {
        "upper": upper_band,
        "middle": middle_band,
        "lower": lower_band,
        "bandwidth": bandwidth,
        "percent_b": (data["Close"] - lower_band) / (upper_band - lower_band),  # %B indicator
    }


# ============================================================================
# ADVANCED VOLUME INDICATORS
# ============================================================================


def calculate_accumulation_distribution(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Accumulation/Distribution Line (A/D Line).

    More sophisticated than OBV - considers where the close is within the
    high-low range. Better detects institutional buying/selling pressure.

    Args:
        data: DataFrame with 'High', 'Low', 'Close', 'Volume' columns

    Returns:
        Series containing A/D Line values
    """
    # Money Flow Multiplier
    mfm = ((data["Close"] - data["Low"]) - (data["High"] - data["Close"])) / (
        data["High"] - data["Low"]
    )

    # Handle division by zero (when High == Low)
    mfm = mfm.fillna(0)

    # Money Flow Volume
    mfv = mfm * data["Volume"]

    # Accumulation/Distribution Line
    ad_line = mfv.cumsum()

    return ad_line


def calculate_chaikin_money_flow(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Chaikin Money Flow (CMF).

    Measures buying and selling pressure over a period. Ranges from -1 to +1.
    Values above 0 indicate buying pressure, below 0 indicate selling pressure.

    Args:
        data: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
        period: Lookback period

    Returns:
        Series containing CMF values
    """
    # Money Flow Multiplier (same as A/D Line)
    mfm = ((data["Close"] - data["Low"]) - (data["High"] - data["Close"])) / (
        data["High"] - data["Low"]
    )
    mfm = mfm.fillna(0)

    # Money Flow Volume
    mfv = mfm * data["Volume"]

    # CMF = Sum of MFV over period / Sum of Volume over period
    cmf = mfv.rolling(window=period).sum() / data["Volume"].rolling(window=period).sum()

    return cmf


def calculate_relative_volume(data: pd.DataFrame, period: int = 20) -> dict[str, Any]:
    """
    Calculate Relative Volume (RVOL).

    Compares current volume to average volume. RVOL > 1.5 indicates
    significantly higher than average activity (potential catalyst).

    Args:
        data: DataFrame with 'Volume' column
        period: Lookback period for average volume

    Returns:
        Dictionary with RVOL data and analysis
    """
    avg_volume = data["Volume"].rolling(window=period).mean()
    rvol = data["Volume"] / avg_volume

    current_rvol = rvol.iloc[-1]

    # Determine significance
    if current_rvol > 2.0:
        significance = "Extremely High - Major catalyst likely"
    elif current_rvol > 1.5:
        significance = "High - Significant interest"
    elif current_rvol > 1.0:
        significance = "Above Average - Moderate interest"
    elif current_rvol > 0.5:
        significance = "Below Average - Weak interest"
    else:
        significance = "Very Low - Minimal activity"

    return {
        "rvol_series": rvol,
        "current_rvol": float(current_rvol),
        "average_volume": int(avg_volume.iloc[-1]),
        "current_volume": int(data["Volume"].iloc[-1]),
        "significance": significance,
    }


def detect_volume_breakout(
    data: pd.DataFrame, threshold_multiplier: float = 2.0, period: int = 20
) -> dict[str, Any]:
    """
    Detect Volume Breakouts.

    Identifies when volume exceeds historical thresholds, indicating
    potential trend changes or significant events.

    Args:
        data: DataFrame with 'Volume' and 'Close' columns
        threshold_multiplier: Volume must be this many times above average
        period: Lookback period for average calculation

    Returns:
        Dictionary with breakout detection results
    """
    avg_volume = data["Volume"].rolling(window=period).mean()
    current_volume = data["Volume"].iloc[-1]
    threshold = avg_volume.iloc[-1] * threshold_multiplier

    is_breakout = current_volume > threshold

    # Get price direction on breakout
    price_change = data["Close"].iloc[-1] - data["Close"].iloc[-2]
    direction = "bullish" if price_change > 0 else "bearish"

    # Find recent breakouts
    breakouts: pd.Series[bool] = data["Volume"] > (avg_volume * threshold_multiplier)  # type: ignore[type-arg]
    recent_breakout_count = int(breakouts.iloc[-5:].sum())  # type: ignore[arg-type]

    return {
        "is_breakout": bool(is_breakout),
        "current_volume": int(current_volume),
        "threshold_volume": int(threshold),
        "multiplier_above_avg": float(current_volume / avg_volume.iloc[-1]),
        "direction": direction if is_breakout else "none",
        "recent_breakouts": int(recent_breakout_count),
        "signal": f"Volume breakout ({direction})" if is_breakout else "No breakout",
    }


def calculate_vwma(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Volume-Weighted Moving Average (VWMA).

    Moving average that gives more weight to periods with higher volume.
    More responsive to institutional activity than simple MA.

    Args:
        data: DataFrame with 'Close' and 'Volume' columns
        period: Moving average period

    Returns:
        Series containing VWMA values
    """
    vwma = (data["Close"] * data["Volume"]).rolling(window=period).sum() / data["Volume"].rolling(
        window=period
    ).sum()

    return vwma


def calculate_price_roc(
    data: pd.DataFrame, period: int = 12, volume_confirmation: bool = True
) -> dict[str, Any]:
    """
    Calculate Price Rate of Change (ROC) with optional volume confirmation.

    Momentum indicator showing percentage change in price. When combined
    with volume, helps confirm trend strength.

    Args:
        data: DataFrame with 'Close' and 'Volume' columns
        period: Lookback period for ROC
        volume_confirmation: If True, check if volume supports the price move

    Returns:
        Dictionary with ROC data and volume confirmation
    """
    # Calculate ROC
    roc = ((data["Close"] - data["Close"].shift(period)) / data["Close"].shift(period)) * 100

    current_roc = roc.iloc[-1]

    # Volume confirmation
    if volume_confirmation:
        avg_volume = data["Volume"].rolling(window=period).mean()
        recent_avg_volume = data["Volume"].iloc[-period:].mean()
        volume_confirmed = recent_avg_volume > avg_volume.iloc[-period - 1]
    else:
        volume_confirmed = None

    # Determine signal strength
    if abs(current_roc) > 10:
        strength = "Strong"
    elif abs(current_roc) > 5:
        strength = "Moderate"
    elif abs(current_roc) > 2:
        strength = "Weak"
    else:
        strength = "Neutral"

    direction = "bullish" if current_roc > 0 else "bearish"
    vol_status = "confirmed" if volume_confirmed else "not confirmed"
    vol_suffix = f" (volume {vol_status})" if volume_confirmation else ""

    return {
        "roc_series": roc,
        "current_roc": float(current_roc),
        "direction": direction,
        "strength": strength,
        "volume_confirmed": volume_confirmed,
        "signal": f"{strength} {direction} momentum{vol_suffix}",
    }


def calculate_enhanced_volume_profile(
    data: pd.DataFrame, num_bins: int = 20, value_area_pct: float = 0.70
) -> dict[str, Any]:
    """
    Calculate Enhanced Volume Profile with Value Area High/Low.

    Extends basic volume profile to include:
    - Point of Control (POC): Price with highest volume
    - Value Area High (VAH): Upper bound of 70% of volume
    - Value Area Low (VAL): Lower bound of 70% of volume

    These levels are critical for options strike selection.

    Args:
        data: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
        num_bins: Number of price bins
        value_area_pct: Percentage of volume for value area (default 70%)

    Returns:
        Dictionary with POC, VAH, VAL, and full profile data
    """
    # Get basic profile
    basic_profile = calculate_volume_profile(data, num_bins)

    price_levels = np.array(basic_profile["price_levels"])
    volumes = np.array(basic_profile["volumes"])

    # Find POC
    max_volume_idx = np.argmax(volumes)
    poc = price_levels[max_volume_idx]

    # Calculate Value Area
    total_volume = volumes.sum()
    value_area_volume = total_volume * value_area_pct

    # Sort indices by volume (descending)
    sorted_indices = np.argsort(volumes)[::-1]

    # Accumulate volume until we reach value_area_pct
    accumulated_volume = 0
    value_area_indices = []

    for idx in sorted_indices:
        value_area_indices.append(idx)
        accumulated_volume += volumes[idx]
        if accumulated_volume >= value_area_volume:
            break

    # Find VAH and VAL
    vah = price_levels[max(value_area_indices)]
    val = price_levels[min(value_area_indices)]

    # Current price position relative to value area
    current_price = data["Close"].iloc[-1]

    if current_price > vah:
        position = "above_value_area"
        interpretation = "Price above value area - potential resistance at VAH"
    elif current_price < val:
        position = "below_value_area"
        interpretation = "Price below value area - potential support at VAL"
    else:
        position = "within_value_area"
        interpretation = "Price within value area - balanced market"

    return {
        "price_levels": basic_profile["price_levels"],
        "volumes": basic_profile["volumes"],
        "poc": float(poc),
        "vah": float(vah),
        "val": float(val),
        "value_area_pct": value_area_pct,
        "current_price": float(current_price),
        "position": position,
        "interpretation": interpretation,
        "poc_distance_pct": float(((current_price / poc) - 1) * 100),
        "vah_distance_pct": float(((current_price / vah) - 1) * 100),
        "val_distance_pct": float(((current_price / val) - 1) * 100),
    }


# ============================================================================
# TREND STRENGTH INDICATORS
# ============================================================================


def calculate_adx(data: pd.DataFrame, period: int = 14) -> dict[str, Any]:
    """
    Calculate Average Directional Index (ADX) with +DI and -DI.

    ADX measures trend strength regardless of direction:
    - ADX > 25: Strong trend (good for directional options plays)
    - ADX < 20: Weak/no trend (better for premium selling strategies)
    - +DI > -DI: Bullish trend
    - -DI > +DI: Bearish trend

    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns
        period: Lookback period for ADX calculation

    Returns:
        Dictionary with ADX, +DI, -DI values and trend analysis
    """
    # Calculate True Range
    high_low = data["High"] - data["Low"]
    high_close = abs(data["High"] - data["Close"].shift(1))
    low_close = abs(data["Low"] - data["Close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate Directional Movement
    up_move = data["High"] - data["High"].shift(1)
    down_move = data["Low"].shift(1) - data["Low"]

    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=data.index)
    minus_dm = pd.Series(minus_dm, index=data.index)

    # Smoothed TR, +DM, -DM using Wilder's smoothing
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    # Current values - extract as scalars
    adx_val = adx.iloc[-1]
    plus_di_val = plus_di.iloc[-1]
    minus_di_val = minus_di.iloc[-1]
    current_adx: float = 0.0 if pd.isna(adx_val) else float(adx_val)  # type: ignore[arg-type]
    current_plus_di: float = 0.0 if pd.isna(plus_di_val) else float(plus_di_val)  # type: ignore[arg-type]
    current_minus_di: float = 0.0 if pd.isna(minus_di_val) else float(minus_di_val)  # type: ignore[arg-type]

    # Trend strength interpretation
    if current_adx > 50:
        trend_strength = "very_strong"
        strength_desc = "Very strong trend - high conviction directional plays"
    elif current_adx > 25:
        trend_strength = "strong"
        strength_desc = "Strong trend - good for directional options"
    elif current_adx > 20:
        trend_strength = "moderate"
        strength_desc = "Moderate trend - use caution with directional plays"
    else:
        trend_strength = "weak"
        strength_desc = "Weak/no trend - consider premium selling strategies"

    # Trend direction
    if current_plus_di > current_minus_di:
        trend_direction = "bullish"
    elif current_minus_di > current_plus_di:
        trend_direction = "bearish"
    else:
        trend_direction = "neutral"

    # ADX trend (is the trend strengthening or weakening?)
    if len(adx) >= 4:
        adx_slope = "strengthening" if adx.iloc[-1] > adx.iloc[-3] else "weakening"
    else:
        adx_slope = "unknown"

    return {
        "adx": float(current_adx),
        "plus_di": float(current_plus_di),
        "minus_di": float(current_minus_di),
        "adx_series": adx,
        "plus_di_series": plus_di,
        "minus_di_series": minus_di,
        "trend_strength": trend_strength,
        "trend_direction": trend_direction,
        "adx_slope": adx_slope,
        "interpretation": strength_desc,
        "signal": f"{trend_strength}_{trend_direction}",
    }


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI oscillates between 0 and 100:
    - RSI > 70: Overbought
    - RSI < 30: Oversold

    Args:
        data: DataFrame with 'Close' column
        period: Lookback period

    Returns:
        Series containing RSI values
    """
    delta = data["Close"].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def detect_rsi_divergence(data: pd.DataFrame, rsi: pd.Series, lookback: int = 10) -> dict[str, Any]:
    """
    Detect RSI divergences (bullish and bearish).

    Bullish Divergence: Price makes lower low, RSI makes higher low
    Bearish Divergence: Price makes higher high, RSI makes lower high

    Args:
        data: DataFrame with 'Close', 'High', 'Low' columns
        rsi: Pre-calculated RSI series
        lookback: Number of bars to look back for divergence detection

    Returns:
        Dictionary with divergence detection results
    """
    if len(data) < lookback + 5:
        return {
            "bullish_divergence": False,
            "bearish_divergence": False,
            "divergence_type": "none",
            "signal": "neutral",
        }

    recent_data = data.iloc[-lookback:]
    recent_rsi = rsi.iloc[-lookback:]

    # Find recent price lows and highs
    price_low_idx = recent_data["Low"].idxmin()
    price_high_idx = recent_data["High"].idxmax()

    # Check for bullish divergence (price lower low, RSI higher low)
    bullish_divergence = False
    if price_low_idx != recent_data.index[0]:  # Not at the start
        # Compare current low area to previous low area
        mid_point = len(recent_data) // 2
        first_half_low = recent_data.iloc[:mid_point]["Low"].min()
        second_half_low = recent_data.iloc[mid_point:]["Low"].min()
        first_half_rsi_low = recent_rsi.iloc[:mid_point].min()
        second_half_rsi_low = recent_rsi.iloc[mid_point:].min()

        if second_half_low < first_half_low and second_half_rsi_low > first_half_rsi_low:
            bullish_divergence = True

    # Check for bearish divergence (price higher high, RSI lower high)
    bearish_divergence = False
    if price_high_idx != recent_data.index[0]:
        mid_point = len(recent_data) // 2
        first_half_high = recent_data.iloc[:mid_point]["High"].max()
        second_half_high = recent_data.iloc[mid_point:]["High"].max()
        first_half_rsi_high = recent_rsi.iloc[:mid_point].max()
        second_half_rsi_high = recent_rsi.iloc[mid_point:].max()

        if second_half_high > first_half_high and second_half_rsi_high < first_half_rsi_high:
            bearish_divergence = True

    # Determine signal
    if bullish_divergence:
        divergence_type = "bullish"
        signal = "potential_reversal_up"
        interpretation = "Bullish divergence - price weakness not confirmed by momentum"
    elif bearish_divergence:
        divergence_type = "bearish"
        signal = "potential_reversal_down"
        interpretation = "Bearish divergence - price strength not confirmed by momentum"
    else:
        divergence_type = "none"
        signal = "neutral"
        interpretation = "No divergence detected"

    return {
        "bullish_divergence": bullish_divergence,
        "bearish_divergence": bearish_divergence,
        "divergence_type": divergence_type,
        "signal": signal,
        "interpretation": interpretation,
        "current_rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
    }


def calculate_rsi_with_divergence(
    data: pd.DataFrame, period: int = 14, divergence_lookback: int = 10
) -> dict[str, Any]:
    """
    Calculate RSI with divergence detection.

    Combines RSI calculation with divergence analysis for better entry timing.

    Args:
        data: DataFrame with OHLC data
        period: RSI period
        divergence_lookback: Bars to check for divergence

    Returns:
        Dictionary with RSI values and divergence analysis
    """
    rsi = calculate_rsi(data, period)
    divergence = detect_rsi_divergence(data, rsi, divergence_lookback)

    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    # RSI condition
    if current_rsi > 70:
        condition = "overbought"
    elif current_rsi < 30:
        condition = "oversold"
    else:
        condition = "neutral"

    return {
        "rsi": float(current_rsi),
        "rsi_series": rsi,
        "condition": condition,
        "period": period,
        **divergence,
    }


# ============================================================================
# OPTIONS-SPECIFIC CALCULATIONS
# ============================================================================


def calculate_iv_percentile(
    data: pd.DataFrame, hv_window: int = 20, lookback_days: int = 252
) -> dict[str, Any]:
    """
    Calculate IV Percentile proxy using Historical Volatility.

    Since we don't have real IV data, we use HV percentile as a proxy:
    - Compares current HV to its range over the lookback period
    - High percentile = volatility is elevated (options expensive)
    - Low percentile = volatility is compressed (options cheap)

    Args:
        data: DataFrame with 'Close' column
        hv_window: Window for current HV calculation
        lookback_days: Days to look back for percentile calculation

    Returns:
        Dictionary with IV percentile proxy data
    """
    # Calculate rolling HV
    log_returns = (data["Close"] / data["Close"].shift(1)).apply(np.log)
    hv = log_returns.rolling(window=hv_window).std() * np.sqrt(252)

    # Get available HV values for percentile calculation
    hv_values = hv.dropna()

    if len(hv_values) < 20:
        hv_val = hv.iloc[-1]
        return {
            "iv_percentile": 50.0,
            "current_hv": 0.0 if pd.isna(hv_val) else float(hv_val),  # type: ignore[arg-type]
            "hv_min": 0.0,
            "hv_max": 0.0,
            "interpretation": "Insufficient data for percentile calculation",
            "options_implication": "neutral",
        }

    # Use available data up to lookback_days
    lookback_hv = hv_values.iloc[-min(lookback_days, len(hv_values)) :]
    current_hv = hv.iloc[-1]

    # Calculate percentile
    hv_min = lookback_hv.min()
    hv_max = lookback_hv.max()
    hv_range = hv_max - hv_min

    if hv_range > 0:
        iv_percentile = ((current_hv - hv_min) / hv_range) * 100
    else:
        iv_percentile = 50.0

    # Interpretation
    if iv_percentile > 80:
        interpretation = "Volatility at highs - options are expensive"
        options_implication = "sell_premium"
        strategy_suggestion = "Consider credit spreads, iron condors, or selling naked options"
    elif iv_percentile > 60:
        interpretation = "Above average volatility"
        options_implication = "slightly_expensive"
        strategy_suggestion = "Favor spreads over naked long options"
    elif iv_percentile < 20:
        interpretation = "Volatility at lows - options are cheap"
        options_implication = "buy_premium"
        strategy_suggestion = "Good time for long straddles, strangles, or directional plays"
    elif iv_percentile < 40:
        interpretation = "Below average volatility"
        options_implication = "slightly_cheap"
        strategy_suggestion = "Directional long options are reasonably priced"
    else:
        interpretation = "Average volatility levels"
        options_implication = "neutral"
        strategy_suggestion = "No strong edge from volatility - focus on direction"

    return {
        "iv_percentile": float(iv_percentile),
        "current_hv": 0.0 if pd.isna(current_hv) else float(current_hv),  # type: ignore[arg-type]
        "hv_min": float(hv_min),
        "hv_max": float(hv_max),
        "lookback_days": min(lookback_days, len(lookback_hv)),
        "interpretation": interpretation,
        "options_implication": options_implication,
        "strategy_suggestion": strategy_suggestion,
    }


def calculate_expected_move(
    data: pd.DataFrame, days_to_expiration: int = 14, hv_window: int = 20
) -> dict[str, Any]:
    """
    Calculate Expected Move for options expiration.

    Formula: Expected Move = Price × HV × √(DTE/252)

    This estimates the expected price range by expiration based on
    historical volatility. Critical for:
    - Strike selection
    - Spread width decisions
    - Probability assessment

    Args:
        data: DataFrame with 'Close' column
        days_to_expiration: Days until options expiration
        hv_window: Window for HV calculation

    Returns:
        Dictionary with expected move calculations
    """
    current_price = data["Close"].iloc[-1]

    # Calculate current HV
    log_returns = (data["Close"] / data["Close"].shift(1)).apply(np.log)
    hv = log_returns.rolling(window=hv_window).std() * np.sqrt(252)
    current_hv = hv.iloc[-1]

    if pd.isna(current_hv):  # type: ignore[arg-type]
        current_hv = 0.20  # Default to 20% if insufficient data

    # Expected move calculation
    # Formula: Price × σ × √(T) where T is time in years
    time_factor = np.sqrt(days_to_expiration / 252)
    expected_move_dollars = current_price * current_hv * time_factor
    expected_move_percent = current_hv * time_factor * 100

    # Calculate price targets
    upper_target = current_price + expected_move_dollars
    lower_target = current_price - expected_move_dollars

    # 1 standard deviation covers ~68% of expected outcomes
    # 1.5 std dev covers ~87%
    # 2 std dev covers ~95%
    targets = {
        "1_std_dev": {
            "probability": "68%",
            "upper": float(upper_target),
            "lower": float(lower_target),
            "range_dollars": float(expected_move_dollars * 2),
        },
        "1.5_std_dev": {
            "probability": "87%",
            "upper": float(current_price + expected_move_dollars * 1.5),
            "lower": float(current_price - expected_move_dollars * 1.5),
            "range_dollars": float(expected_move_dollars * 3),
        },
        "2_std_dev": {
            "probability": "95%",
            "upper": float(current_price + expected_move_dollars * 2),
            "lower": float(current_price - expected_move_dollars * 2),
            "range_dollars": float(expected_move_dollars * 4),
        },
    }

    return {
        "current_price": float(current_price),
        "days_to_expiration": days_to_expiration,
        "historical_volatility": float(current_hv),
        "expected_move_dollars": float(expected_move_dollars),
        "expected_move_percent": float(expected_move_percent),
        "upper_target_1std": float(upper_target),
        "lower_target_1std": float(lower_target),
        "targets": targets,
        "interpretation": (
            f"Based on {current_hv:.1%} HV, price is expected to move "
            f"±${expected_move_dollars:.2f} ({expected_move_percent:.1f}%) "
            f"by expiration (68% probability)"
        ),
        "strike_guidance": {
            "atm_strike": float(round(current_price)),
            "otm_call_1std": float(round(upper_target)),
            "otm_put_1std": float(round(lower_target)),
            "safe_short_call": float(round(current_price + expected_move_dollars * 1.5)),
            "safe_short_put": float(round(current_price - expected_move_dollars * 1.5)),
        },
    }


# ============================================================================
# COMPOSITE SIGNAL SCORING
# ============================================================================


def calculate_composite_score(data: pd.DataFrame, holding_period: int = 14) -> dict[str, Any]:
    """
    Calculate composite signal score for options trading.

    Aggregates multiple indicators into a single score from -10 to +10:
    - Positive scores favor bullish plays (calls)
    - Negative scores favor bearish plays (puts)
    - Scores near 0 suggest neutral/range-bound strategies

    Args:
        data: DataFrame with OHLC and Volume data
        holding_period: Days for options holding period (affects indicator tuning)

    Returns:
        Dictionary with composite score and breakdown
    """
    # Adaptive parameters based on holding period
    if holding_period <= 14:
        mfi_period = 7
        volume_window = 10
        rsi_period = 7
        adx_period = 10
    elif holding_period <= 21:
        mfi_period = 10
        volume_window = 14
        rsi_period = 10
        adx_period = 14
    else:  # 22-30 days
        mfi_period = 14
        volume_window = 20
        rsi_period = 14
        adx_period = 14

    # Calculate indicators
    obv = calculate_obv(data)
    ad_line = calculate_accumulation_distribution(data)
    vwap = calculate_vwap(data)
    vwma = calculate_vwma(data, volume_window)
    mfi = calculate_mfi(data, mfi_period)
    cmf = calculate_chaikin_money_flow(data, volume_window)
    rsi_data = calculate_rsi_with_divergence(data, rsi_period, volume_window)
    adx_data = calculate_adx(data, adx_period)
    breakout = detect_volume_breakout(data, 2.0, volume_window)

    latest_close = data["Close"].iloc[-1]
    latest_vwap = vwap.iloc[-1]
    latest_vwma = vwma.iloc[-1]
    latest_mfi = mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50.0
    latest_cmf = cmf.iloc[-1] if not pd.isna(cmf.iloc[-1]) else 0.0
    latest_rsi = rsi_data["rsi"]

    # Scoring components (each from -2 to +2, some weighted more)
    score_breakdown = {}

    # 1. Price vs VWAP (+2/-2)
    if latest_close > latest_vwap * 1.02:
        score_breakdown["price_vs_vwap"] = 2
    elif latest_close > latest_vwap:
        score_breakdown["price_vs_vwap"] = 1
    elif latest_close < latest_vwap * 0.98:
        score_breakdown["price_vs_vwap"] = -2
    else:
        score_breakdown["price_vs_vwap"] = -1

    # 2. Price vs VWMA (+1/-1)
    score_breakdown["price_vs_vwma"] = 1 if latest_close > latest_vwma else -1

    # 3. OBV momentum (+2/-2)
    if len(obv) >= 6:
        obv_momentum = obv.iloc[-1] > obv.iloc[-3]
        obv_strong = obv.iloc[-1] > obv.iloc[-5]
        if obv_momentum and obv_strong:
            score_breakdown["obv_momentum"] = 2
        elif obv_momentum:
            score_breakdown["obv_momentum"] = 1
        elif not obv_momentum and not obv_strong:
            score_breakdown["obv_momentum"] = -2
        else:
            score_breakdown["obv_momentum"] = -1
    else:
        score_breakdown["obv_momentum"] = 0

    # 4. A/D Line momentum (+1/-1)
    if len(ad_line) >= 4:
        ad_momentum = ad_line.iloc[-1] > ad_line.iloc[-3]
        score_breakdown["ad_momentum"] = 1 if ad_momentum else -1
    else:
        ad_momentum = False
        score_breakdown["ad_momentum"] = 0

    # 5. MFI condition (+2/-2)
    if latest_mfi < 25:
        score_breakdown["mfi"] = 2  # Oversold = bullish
    elif latest_mfi < 40:
        score_breakdown["mfi"] = 1
    elif latest_mfi > 75:
        score_breakdown["mfi"] = -2  # Overbought = bearish
    elif latest_mfi > 60:
        score_breakdown["mfi"] = -1
    else:
        score_breakdown["mfi"] = 0

    # 6. CMF (+1/-1)
    if latest_cmf > 0.1:
        score_breakdown["cmf"] = 1
    elif latest_cmf < -0.1:
        score_breakdown["cmf"] = -1
    else:
        score_breakdown["cmf"] = 0

    # 7. RSI condition (+2/-2)
    if latest_rsi < 30:
        score_breakdown["rsi"] = 2
    elif latest_rsi < 40:
        score_breakdown["rsi"] = 1
    elif latest_rsi > 70:
        score_breakdown["rsi"] = -2
    elif latest_rsi > 60:
        score_breakdown["rsi"] = -1
    else:
        score_breakdown["rsi"] = 0

    # 8. RSI divergence (+2/-2) - high weight reversal signal
    if rsi_data["bullish_divergence"]:
        score_breakdown["rsi_divergence"] = 2
    elif rsi_data["bearish_divergence"]:
        score_breakdown["rsi_divergence"] = -2
    else:
        score_breakdown["rsi_divergence"] = 0

    # 9. ADX trend direction (+1/-1) - only if trend is strong
    if adx_data["adx"] > 25:
        if adx_data["trend_direction"] == "bullish":
            score_breakdown["adx_direction"] = 1
        elif adx_data["trend_direction"] == "bearish":
            score_breakdown["adx_direction"] = -1
        else:
            score_breakdown["adx_direction"] = 0
    else:
        score_breakdown["adx_direction"] = 0

    # 10. Volume breakout (+1/-1)
    if breakout["is_breakout"]:
        score_breakdown["volume_breakout"] = 1 if breakout["direction"] == "bullish" else -1
    else:
        score_breakdown["volume_breakout"] = 0

    # Calculate total score
    total_score = sum(score_breakdown.values())

    # Normalize to -10 to +10 scale
    max_possible = 15  # Sum of max positive scores
    normalized_score = (total_score / max_possible) * 10

    # Determine recommendation
    if normalized_score >= 5:
        recommendation = "strong_bullish"
        action = "High conviction call options or bull spreads"
    elif normalized_score >= 2:
        recommendation = "bullish"
        action = "Consider call options or call spreads"
    elif normalized_score <= -5:
        recommendation = "strong_bearish"
        action = "High conviction put options or bear spreads"
    elif normalized_score <= -2:
        recommendation = "bearish"
        action = "Consider put options or put spreads"
    else:
        recommendation = "neutral"
        action = "Consider iron condors, strangles, or wait for clearer signals"

    # Signal quality based on ADX
    if adx_data["adx"] > 30:
        signal_quality = "high"
        quality_note = "Strong trend supports directional trades"
    elif adx_data["adx"] > 20:
        signal_quality = "medium"
        quality_note = "Moderate trend - use appropriate position sizing"
    else:
        signal_quality = "low"
        quality_note = "Weak trend - premium selling may be better"

    return {
        "composite_score": float(normalized_score),
        "raw_score": total_score,
        "max_score": max_possible,
        "recommendation": recommendation,
        "action": action,
        "signal_quality": signal_quality,
        "quality_note": quality_note,
        "score_breakdown": score_breakdown,
        "indicator_summary": {
            "price_above_vwap": latest_close > latest_vwap,
            "price_above_vwma": latest_close > latest_vwma,
            "obv_bullish": obv_momentum,
            "ad_bullish": ad_momentum,
            "mfi": float(latest_mfi),
            "cmf": float(latest_cmf),
            "rsi": float(latest_rsi),
            "rsi_divergence": rsi_data["divergence_type"],
            "adx": float(adx_data["adx"]),
            "adx_trend": adx_data["trend_direction"],
            "volume_breakout": breakout["is_breakout"],
        },
    }
