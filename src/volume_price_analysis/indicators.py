"""Volume-Price analysis indicators and calculations."""

from typing import Any

import numpy as np
import pandas as pd


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Apply Wilder's smoothing: SMA seed over first `period` non-NaN values, then recursive.

    This matches the standard Wilder smoothing used by TradingView, ThinkorSwim, etc.
    Formula: smoothed[t] = (smoothed[t-1] * (period - 1) + value[t]) / period

    Handles leading NaN values correctly (e.g., when ADX double-smooths DX which
    already has NaN from the first smoothing pass).
    """
    values = series.to_numpy(dtype=float)
    arr = np.full(len(values), np.nan)

    # Find the first `period` non-NaN values to seed the SMA
    valid_indices = np.where(~np.isnan(values))[0]
    if len(valid_indices) < period:
        return pd.Series(arr, index=series.index)

    seed_end = valid_indices[period - 1]
    arr[seed_end] = np.mean(values[valid_indices[:period]])

    # Apply Wilder's recursive smoothing from the next position onward
    for i in range(seed_end + 1, len(values)):
        if np.isnan(values[i]):
            arr[i] = arr[i - 1]
            continue
        arr[i] = (arr[i - 1] * (period - 1) + values[i]) / period

    return pd.Series(arr, index=series.index)


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
    cum_vol = data["Volume"].cumsum()
    vwap = (typical_price * data["Volume"]).cumsum() / cum_vol
    vwap = vwap.where(cum_vol != 0, np.nan)
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
    # Empty input: min()/max() return NaN and np.linspace would yield all-NaN
    # price levels (silent garbage). Degrade to neutral zero-filled bins.
    if data.empty:
        return {"price_levels": [0.0] * num_bins, "volumes": [0.0] * num_bins}

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

    # For candles spanning single bins, use fast bincount
    single_bin_mask = low_bins == high_bins
    if np.any(single_bin_mask):
        single_volumes = np.bincount(
            low_bins[single_bin_mask],
            weights=volume_per_bin[single_bin_mask],
            minlength=num_bins,
        )
        volumes += single_volumes[:num_bins]

    # For candles spanning multiple bins, distribute volume across bins
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
        # A zero prior close makes the percentage change undefined (div-by-zero
        # -> inf/NaN). Treat it as no measurable change so VPT stays finite.
        if prev_close == 0:
            price_change_pct = 0.0
        else:
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

    # Guard division by zero in the money-flow ratio. Computing the ratio only
    # where negative_mf > 0 avoids inf/NaN; the two .where() passes then assign
    # the conventional values for the degenerate windows:
    #   - negative_mf == 0 with positive flow -> fully overbought (MFI = 100)
    #   - flat window (no flow either way)     -> neutral (MFI = 50)
    # Leading insufficient-history values (rolling sum still NaN) stay NaN.
    mfr = positive_mf / negative_mf.where(negative_mf > 0)
    mfi = 100 - (100 / (1 + mfr))
    mfi = mfi.where(~((negative_mf == 0) & (positive_mf > 0)), 100.0)
    mfi = mfi.where(~((negative_mf == 0) & (positive_mf == 0)), 50.0)

    # Flat window: no positive AND no negative money flow gives 0/0 -> NaN.
    # (One-sided flow already resolves correctly: negative_mf == 0 with positive
    # flow yields mfr == inf -> MFI == 100.) Degrade the undefined 0/0 case to a
    # neutral 50 rather than emitting NaN. The rolling-window warmup stays NaN
    # because its sums are NaN (NaN == 0 is False), so this only touches fully
    # flat windows past warmup.
    flat_window = (positive_mf == 0) & (negative_mf == 0)
    mfi = mfi.mask(flat_window, 50.0)

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
    # Empty input has no last bar to analyse; return neutral, non-throwing
    # defaults rather than raising IndexError on the .iloc[-1] lookups below.
    if data.empty:
        return {
            "current_volume": 0,
            "average_volume": 0,
            "volume_vs_average": "N/A",
            "volume_trend": "unknown",
            "price_direction": "unknown",
            "divergence_detected": False,
            "divergence_type": "None",
        }

    # When history is shorter than the window, a plain rolling mean is all-NaN
    # (int(NaN) raises) and iloc[-window] raises IndexError. Clamp the lookback
    # and average over whatever history exists (min_periods=1) so the analysis
    # still degrades to a real answer. For full history this is identical to the
    # original rolling mean at the final bar.
    lookback = min(window, len(data))
    avg_volume = data["Volume"].rolling(window=window, min_periods=1).mean()
    current_volume = data["Volume"].iloc[-1]
    current_avg = avg_volume.iloc[-1]

    # Calculate volume trend
    volume_increasing = data["Volume"].iloc[-5:].is_monotonic_increasing

    # Calculate price-volume divergence
    price_direction = "up" if data["Close"].iloc[-1] > data["Close"].iloc[-lookback] else "down"

    # A zero/NaN current volume or average (all-zero or NaN volume window) makes
    # the ratio undefined; fall back to a neutral comparison and "N/A" display.
    if pd.isna(current_volume) or pd.isna(current_avg) or current_avg == 0:
        volume_direction = "down"
        volume_vs_average = "N/A"
    else:
        volume_direction = "up" if current_volume > current_avg else "down"
        volume_vs_average = f"{((current_volume / current_avg - 1) * 100):.2f}%"

    divergence = (price_direction == "up" and volume_direction == "down") or (
        price_direction == "down" and volume_direction == "up"
    )

    divergence_type = (
        f"Price {price_direction}, Volume {volume_direction}" if divergence else "None"
    )

    return {
        "current_volume": 0 if pd.isna(current_volume) else int(current_volume),
        "average_volume": 0 if pd.isna(current_avg) else int(current_avg),
        "volume_vs_average": volume_vs_average,
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

    # Calculate ATR using Wilder's smoothing (SMA-seeded, then recursive)
    atr = _wilder_smooth(true_range, period)

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
    if len(data) < 2:
        return {
            "is_breakout": False,
            "current_volume": int(data["Volume"].iloc[-1]) if len(data) else 0,
            "threshold_volume": 0,
            "multiplier_above_avg": 0.0,
            "direction": "none",
            "recent_breakouts": 0,
            "signal": "No breakout",
        }

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
    # Empty input has no last close: data["Close"].iloc[-1] below would raise
    # IndexError, and the delegate calculate_volume_profile yields NaN price
    # levels for an empty frame. Degrade gracefully to neutral, non-NaN defaults
    # consistent with the delegate's HOM-37 hardening (see HOM-41).
    if data.empty:
        return {
            "price_levels": [0.0] * num_bins,
            "volumes": [0.0] * num_bins,
            "poc": 0.0,
            "vah": 0.0,
            "val": 0.0,
            "value_area_pct": value_area_pct,
            "current_price": 0.0,
            "position": "within_value_area",
            "interpretation": "No price data available",
            "poc_distance_pct": 0.0,
            "vah_distance_pct": 0.0,
            "val_distance_pct": 0.0,
        }

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

    def _distance_pct(level: float) -> float:
        # A zero reference level (e.g. degenerate all-zero prices give POC/VAH/VAL
        # of 0) makes the percentage undefined; report 0.0 instead of inf/NaN.
        if level == 0:
            return 0.0
        return float(((current_price / level) - 1) * 100)

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
        "poc_distance_pct": _distance_pct(poc),
        "vah_distance_pct": _distance_pct(vah),
        "val_distance_pct": _distance_pct(val),
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

    # Smoothed TR, +DM, -DM using Wilder's smoothing (SMA-seeded, then recursive)
    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus_dm = _wilder_smooth(plus_dm, period)
    smoothed_minus_dm = _wilder_smooth(minus_dm, period)
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = _wilder_smooth(dx, period)

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

    gain = _wilder_smooth(delta.clip(lower=0), period)
    loss = _wilder_smooth((-delta).clip(lower=0), period)

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def find_pivots(series: pd.Series, left: int = 3, right: int = 3) -> tuple[pd.Series, pd.Series]:
    """Mark strict structural swing highs and lows.

    A bar ``i`` is a **pivot high** iff ``series[i]`` is strictly greater than
    each of the ``left`` bars before it and each of the ``right`` bars after it
    (both windows must fully exist). A **pivot low** is the strict-minimum
    analogue. Ties never produce a pivot, so flat plateaus are ignored.

    Causality note: the last ``right`` bars can never be pivots because their
    right-hand confirmation window has not finished forming. A caller that only
    treats a pivot at index ``i`` as "known" once bar ``i + right`` has printed
    therefore never peeks at future bars — this is the property that makes the
    divergence detector strictly causal (see :func:`detect_rsi_divergence`).

    Args:
        series: Numeric series to scan (e.g. ``data["High"]`` or ``data["Low"]``).
        left: Bars required to the left of a pivot.
        right: Bars required to the right (the confirmation window).

    Returns:
        ``(pivot_high_mask, pivot_low_mask)`` — boolean Series aligned to
        ``series.index``.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    high_mask = np.zeros(n, dtype=bool)
    low_mask = np.zeros(n, dtype=bool)

    if left >= 1 and right >= 1:
        for i in range(left, n - right):
            v = values[i]
            window = values[i - left : i + right + 1]
            if np.isnan(window).any():
                continue
            left_win = window[:left]
            right_win = window[left + 1 :]
            if v > left_win.max() and v > right_win.max():
                high_mask[i] = True
            elif v < left_win.min() and v < right_win.min():
                low_mask[i] = True

    return (
        pd.Series(high_mask, index=series.index),
        pd.Series(low_mask, index=series.index),
    )


def _structural_pivot_indices(
    data: pd.DataFrame, pivot_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Positional indices of structural swing highs (on High) and lows (on Low)."""
    pivot_high_mask, _ = find_pivots(data["High"], pivot_window, pivot_window)
    _, pivot_low_mask = find_pivots(data["Low"], pivot_window, pivot_window)
    return (
        np.flatnonzero(pivot_high_mask.to_numpy()),
        np.flatnonzero(pivot_low_mask.to_numpy()),
    )


def _effective_search(lookback: int, pivot_window: int, search_window: int | None) -> int:
    """Resolve the bar span searched for the two most-recent confirmed pivots."""
    if search_window is not None:
        return search_window
    return max(lookback * 3, 4 * pivot_window + 2)


def _two_most_recent(idx_array: np.ndarray, lo: int, hi: int) -> tuple[int, int] | None:
    """Return the two most-recent indices in ``idx_array`` within ``[lo, hi]``."""
    sel = idx_array[(idx_array >= lo) & (idx_array <= hi)]
    if len(sel) < 2:
        return None
    return int(sel[-2]), int(sel[-1])


def _divergence_from_pivots(
    lows: np.ndarray,
    highs: np.ndarray,
    rsi_vals: np.ndarray,
    pivot_high_idx: np.ndarray,
    pivot_low_idx: np.ndarray,
    t: int,
    pivot_window: int,
    search: int,
) -> tuple[bool, bool]:
    """Evaluate bullish/bearish divergence *as of* bar ``t`` (strictly causal).

    Only pivots whose confirmation window has closed by ``t`` (index
    ``p <= t - pivot_window``) and that fall within the recency ``search`` span
    are considered, so no value past bar ``t`` can influence the result.
    """
    confirm_cap = t - pivot_window
    lo = t - search + 1

    bullish = False
    low_pair = _two_most_recent(pivot_low_idx, lo, confirm_cap)
    if low_pair is not None:
        p1, p2 = low_pair
        if (
            not np.isnan(rsi_vals[p1])
            and not np.isnan(rsi_vals[p2])
            and lows[p2] < lows[p1]  # price prints a lower low
            and rsi_vals[p2] > rsi_vals[p1]  # momentum prints a higher low
        ):
            bullish = True

    bearish = False
    high_pair = _two_most_recent(pivot_high_idx, lo, confirm_cap)
    if high_pair is not None:
        p1, p2 = high_pair
        if (
            not np.isnan(rsi_vals[p1])
            and not np.isnan(rsi_vals[p2])
            and highs[p2] > highs[p1]  # price prints a higher high
            and rsi_vals[p2] < rsi_vals[p1]  # momentum prints a lower high
        ):
            bearish = True

    return bullish, bearish


def _format_divergence(bullish: bool, bearish: bool, current_rsi: float) -> dict[str, Any]:
    """Render the divergence result dict (stable MCP-facing shape).

    Tie-break: if both a bullish and a bearish divergence are detected in the
    same bar (rare — it needs both the two recent pivot lows and the two recent
    pivot highs to diverge), bullish takes priority. The output booleans are kept
    **mutually exclusive** so ``divergence_type`` and the boolean flags can never
    contradict each other for a downstream client. The composite scorer reads
    ``bullish_divergence`` first, so this matches its ±2 weighting.
    """
    if bullish:
        divergence_type = "bullish"
        signal = "potential_reversal_up"
        interpretation = "Bullish divergence - price weakness not confirmed by momentum"
        bullish_out, bearish_out = True, False
    elif bearish:
        divergence_type = "bearish"
        signal = "potential_reversal_down"
        interpretation = "Bearish divergence - price strength not confirmed by momentum"
        bullish_out, bearish_out = False, True
    else:
        divergence_type = "none"
        signal = "neutral"
        interpretation = "No divergence detected"
        bullish_out, bearish_out = False, False

    return {
        "bullish_divergence": bullish_out,
        "bearish_divergence": bearish_out,
        "divergence_type": divergence_type,
        "signal": signal,
        "interpretation": interpretation,
        "current_rsi": current_rsi,
    }


def detect_rsi_divergence(
    data: pd.DataFrame,
    rsi: pd.Series,
    lookback: int = 10,
    *,
    pivot_window: int = 3,
    search_window: int | None = None,
) -> dict[str, Any]:
    """
    Detect RSI divergences (bullish and bearish) at the latest bar.

    Bullish Divergence: price makes a lower low while RSI makes a higher low.
    Bearish Divergence: price makes a higher high while RSI makes a lower high.

    The detector compares the **two most recent confirmed swing pivots** (found
    with :func:`find_pivots`) rather than an arbitrary first-half/second-half
    window. A pivot is only "confirmed" once its right-hand window has fully
    printed (``index <= last_bar - pivot_window``), so the result at the latest
    bar uses no future information — it is strictly causal. See
    :func:`rsi_divergence_signal_series` and the no-lookahead tests.

    Args:
        data: DataFrame with 'High', 'Low' columns.
        rsi: Pre-calculated RSI series aligned to ``data``.
        lookback: Minimum-history gate and, by default, the basis for the pivot
            recency search span (``max(lookback * 3, ...)``).
        pivot_window: Bars required on each side of a swing pivot (strength).
        search_window: Override for how many recent bars to search for the two
            comparison pivots. Defaults to a multiple of ``lookback``.

    Returns:
        Dictionary with divergence detection results (keys unchanged for MCP
        contract stability).
    """
    n = len(data)
    current_rsi = float(rsi.iloc[-1]) if n > 0 and not pd.isna(rsi.iloc[-1]) else 50.0

    if n < lookback + 5:
        result = _format_divergence(False, False, current_rsi)
        result["interpretation"] = "Insufficient data for divergence detection"
        return result

    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)
    rsi_vals = rsi.to_numpy(dtype=float)
    pivot_high_idx, pivot_low_idx = _structural_pivot_indices(data, pivot_window)
    search = _effective_search(lookback, pivot_window, search_window)

    bullish, bearish = _divergence_from_pivots(
        lows, highs, rsi_vals, pivot_high_idx, pivot_low_idx, n - 1, pivot_window, search
    )
    return _format_divergence(bullish, bearish, current_rsi)


def rsi_divergence_signal_series(
    data: pd.DataFrame,
    rsi: pd.Series,
    lookback: int = 10,
    *,
    pivot_window: int = 3,
    search_window: int | None = None,
) -> pd.Series:
    """Per-bar causal divergence signal (+1 bullish / -1 bearish / 0 none).

    Element ``t`` is the divergence verdict computed using only bars ``<= t`` —
    identical to ``detect_rsi_divergence(data.iloc[: t + 1], ...)`` — so the
    series is strictly causal by construction. Structural pivots are computed
    once over the full series and then gated per bar by their confirmation
    window, which is why appending future bars can never change a past value.

    This is primarily an evidence/validation surface (used by the no-lookahead
    tests and the A8 backtest); production scoring only needs the latest bar via
    :func:`detect_rsi_divergence`.
    """
    n = len(data)
    out = np.zeros(n, dtype=int)

    if n >= lookback + 5:
        highs = data["High"].to_numpy(dtype=float)
        lows = data["Low"].to_numpy(dtype=float)
        rsi_vals = rsi.to_numpy(dtype=float)
        pivot_high_idx, pivot_low_idx = _structural_pivot_indices(data, pivot_window)
        search = _effective_search(lookback, pivot_window, search_window)
        # First evaluable bar mirrors detect_rsi_divergence's len < lookback + 5
        # guard: a slice of length lookback + 5 ends at bar lookback + 4.
        for t in range(lookback + 4, n):
            bullish, bearish = _divergence_from_pivots(
                lows, highs, rsi_vals, pivot_high_idx, pivot_low_idx, t, pivot_window, search
            )
            out[t] = 1 if bullish else (-1 if bearish else 0)

    return pd.Series(out, index=data.index, dtype=int)


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


def calculate_rsi_divergence(
    data: pd.DataFrame, period: int = 14, divergence_lookback: int = 10
) -> dict[str, Any]:
    """JSON-serializable RSI-divergence result for the standalone MCP tool.

    Wraps :func:`calculate_rsi_with_divergence`, dropping ``rsi_series``
    so the output can be safely serialized to JSON.
    """
    inner = calculate_rsi_with_divergence(data, period, divergence_lookback)
    return {
        "rsi": inner["rsi"],
        "condition": inner["condition"],
        "period": inner["period"],
        "bullish_divergence": inner["bullish_divergence"],
        "bearish_divergence": inner["bearish_divergence"],
        "divergence_type": inner["divergence_type"],
        "signal": inner["signal"],
        "interpretation": inner["interpretation"],
        "current_rsi": inner["current_rsi"],
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
        Dictionary with IV percentile proxy data. ``iv_percentile`` is retained
        for backward compatibility, but the value is derived from historical
        volatility, NOT options-market implied volatility. The honestly-labeled
        ``hv_percentile`` carries the same number, with ``basis`` and ``is_proxy``
        marking it as an HV-based proxy.
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
            "hv_percentile": 50.0,
            "basis": "historical_volatility",
            "is_proxy": True,
            "current_hv": 0.0 if pd.isna(hv_val) else float(hv_val),  # type: ignore[arg-type]
            "hv_min": 0.0,
            "hv_max": 0.0,
            "lookback_days": 0,
            "interpretation": "Insufficient data for percentile calculation",
            "options_implication": "neutral",
            "strategy_suggestion": "Use standard position sizing, insufficient volatility data",
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

    # Guard NaN propagation: a NaN current_hv (e.g. a NaN final Close) would make
    # the percentile NaN. Degrade to a neutral 50.0 rather than leak NaN to callers.
    if pd.isna(iv_percentile):
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
        "hv_percentile": float(iv_percentile),
        "basis": "historical_volatility",
        "is_proxy": True,
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


def composite_adx_period(holding_period: int) -> int:
    """Return the ADX lookback the composite score uses for a holding period.

    Short holding periods use a more responsive ADX(10); 15+ day holds use the
    standard ADX(14). Centralised so the scan can report and filter on the exact
    same period the composite score consumed, rather than a separate fixed ADX(14)
    (see HOM-48). Keep this the single source of truth for the rule.
    """
    return 10 if holding_period <= 14 else 14


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
    elif holding_period <= 21:
        mfi_period = 10
        volume_window = 14
        rsi_period = 10
    else:  # 22-30 days
        mfi_period = 14
        volume_window = 20
        rsi_period = 14
    adx_period = composite_adx_period(holding_period)

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
        obv_momentum = False
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
        # Surface the ADX the score actually consumed (adaptive to holding_period) so
        # callers can report a value coherent with signal_quality/adx_direction instead
        # of recomputing a separate, fixed-period ADX. JSON-safe scalars only (no Series).
        "adx_period": adx_period,
        "adx_summary": {
            "period": adx_period,
            "adx": float(adx_data["adx"]),
            "plus_di": float(adx_data["plus_di"]),
            "minus_di": float(adx_data["minus_di"]),
            "trend_strength": adx_data["trend_strength"],
            "trend_direction": adx_data["trend_direction"],
            "adx_slope": adx_data["adx_slope"],
        },
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
