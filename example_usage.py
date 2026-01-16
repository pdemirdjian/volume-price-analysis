#!/usr/bin/env python3
"""
Example script to test the volume-price analysis functions directly.
This demonstrates the analysis capabilities without running the MCP server.
"""

from src.volume_price_analysis.data_fetcher import fetch_stock_data
from src.volume_price_analysis.indicators import (
    analyze_volume_trends,
    calculate_mfi,
    calculate_obv,
    calculate_volume_profile,
    calculate_vwap,
)


def main():
    """Run example analysis on a stock."""

    # Configuration
    symbol = "AAPL"
    period = "1mo"

    print(f"\n{'=' * 60}")
    print(f"Volume-Price Analysis for {symbol}")
    print(f"{'=' * 60}\n")

    # Fetch data
    print(f"Fetching {period} of data for {symbol}...")
    data = fetch_stock_data(symbol, period=period)
    print(f"✓ Retrieved {len(data)} data points")
    start_date = data["Date"].iloc[0].strftime("%Y-%m-%d")
    end_date = data["Date"].iloc[-1].strftime("%Y-%m-%d")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Latest close: ${data['Close'].iloc[-1]:.2f}")
    print()

    # Calculate OBV
    print("1. On-Balance Volume (OBV)")
    print("-" * 40)
    obv = calculate_obv(data)
    print(f"   Current OBV: {obv.iloc[-1]:,.0f}")
    print(f"   Trend: {'📈 Increasing' if obv.iloc[-1] > obv.iloc[-5] else '📉 Decreasing'}")
    print()

    # Calculate VWAP
    print("2. Volume Weighted Average Price (VWAP)")
    print("-" * 40)
    vwap = calculate_vwap(data)
    latest_close = data["Close"].iloc[-1]
    latest_vwap = vwap.iloc[-1]
    diff_pct = (latest_close / latest_vwap - 1) * 100
    print(f"   Current VWAP: ${latest_vwap:.2f}")
    print(f"   Latest Close: ${latest_close:.2f}")
    print(f"   Difference: {diff_pct:+.2f}%")
    print(f"   Position: Price is {'above' if latest_close > latest_vwap else 'below'} VWAP")
    print()

    # Calculate Volume Profile
    print("3. Volume Profile")
    print("-" * 40)
    profile = calculate_volume_profile(data, num_bins=15)
    max_volume_idx = profile["volumes"].index(max(profile["volumes"]))
    poc = profile["price_levels"][max_volume_idx]
    print(f"   Point of Control (POC): ${poc:.2f}")
    print(f"   POC Volume: {profile['volumes'][max_volume_idx]:,.0f}")
    price_min = min(profile["price_levels"])
    price_max = max(profile["price_levels"])
    print(f"   Price Range: ${price_min:.2f} - ${price_max:.2f}")
    print()

    # Calculate MFI
    print("4. Money Flow Index (MFI)")
    print("-" * 40)
    mfi = calculate_mfi(data, period=14)
    latest_mfi = mfi.iloc[-1]
    if latest_mfi > 80:
        condition = "🔴 Overbought"
    elif latest_mfi < 20:
        condition = "🟢 Oversold"
    else:
        condition = "🟡 Neutral"
    print(f"   Current MFI: {latest_mfi:.2f}")
    print(f"   Condition: {condition}")
    print()

    # Volume Trends
    print("5. Volume Trend Analysis")
    print("-" * 40)
    trends = analyze_volume_trends(data, window=20)
    print(f"   Current Volume: {trends['current_volume']:,}")
    print(f"   Average Volume: {trends['average_volume']:,}")
    print(f"   vs Average: {trends['volume_vs_average']}")
    print(f"   Trend: {trends['volume_trend']}")
    if trends["divergence_detected"]:
        print(f"   ⚠️  Divergence: {trends['divergence_type']}")
    else:
        print("   ✓ No divergence detected")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary_points = []

    # Price vs VWAP
    if latest_close > latest_vwap:
        summary_points.append("✓ Price trading above VWAP (bullish sentiment)")
    else:
        summary_points.append("✗ Price trading below VWAP (bearish sentiment)")

    # OBV
    if obv.iloc[-1] > obv.iloc[-5]:
        summary_points.append("✓ OBV increasing (accumulation)")
    else:
        summary_points.append("✗ OBV decreasing (distribution)")

    # MFI
    if latest_mfi > 80:
        summary_points.append("⚠️  MFI overbought (potential pullback)")
    elif latest_mfi < 20:
        summary_points.append("⚠️  MFI oversold (potential bounce)")
    else:
        summary_points.append("✓ MFI in neutral zone")

    # Divergence
    if trends["divergence_detected"]:
        summary_points.append(f"⚠️  Price-volume divergence: {trends['divergence_type']}")

    for point in summary_points:
        print(f"  {point}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("Make sure you have installed dependencies:")
        print("  pip install -e .")
