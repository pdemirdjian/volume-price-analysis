"""Tests for volume-price indicators."""

import numpy as np
import pandas as pd
import pytest

from volume_price_analysis.indicators import (
    analyze_volume_trends,
    calculate_mfi,
    calculate_obv,
    calculate_volume_profile,
    calculate_vpt,
    calculate_vwap,
)


class TestOBV:
    """Tests for On-Balance Volume calculation."""

    def test_obv_basic_calculation(self, sample_stock_data):
        """Test basic OBV calculation."""
        obv = calculate_obv(sample_stock_data)

        assert len(obv) == len(sample_stock_data)
        assert obv.iloc[0] == 0  # First value should be 0
        assert isinstance(obv, pd.Series)

    def test_obv_uptrend(self, uptrend_data):
        """Test OBV increases with uptrend."""
        obv = calculate_obv(uptrend_data)

        # OBV should generally increase with price
        assert obv.iloc[-1] > obv.iloc[0]
        # Should be monotonically increasing for consistent uptrend
        assert all(obv.iloc[i] >= obv.iloc[i - 1] for i in range(1, len(obv)))

    def test_obv_downtrend(self, downtrend_data):
        """Test OBV decreases with downtrend."""
        obv = calculate_obv(downtrend_data)

        # OBV should decrease with price decline
        assert obv.iloc[-1] < obv.iloc[0]
        # Should be monotonically decreasing
        assert all(obv.iloc[i] <= obv.iloc[i - 1] for i in range(1, len(obv)))

    def test_obv_flat_price(self, flat_price_data):
        """Test OBV behavior with flat prices."""
        obv = calculate_obv(flat_price_data)

        # OBV should stay relatively flat when price doesn't change
        assert obv.iloc[0] == obv.iloc[-1]


class TestVWAP:
    """Tests for Volume Weighted Average Price calculation."""

    def test_vwap_basic_calculation(self, sample_stock_data):
        """Test basic VWAP calculation."""
        vwap = calculate_vwap(sample_stock_data)

        assert len(vwap) == len(sample_stock_data)
        assert all(vwap > 0)
        assert isinstance(vwap, pd.Series)

    def test_vwap_within_price_range(self, sample_stock_data):
        """Test VWAP is within the high-low range."""
        vwap = calculate_vwap(sample_stock_data)

        # VWAP should generally be within the price range
        assert all(vwap >= sample_stock_data["Low"].min())
        assert all(vwap <= sample_stock_data["High"].max())

    def test_vwap_cumulative(self, uptrend_data):
        """Test VWAP is cumulative and smooths over time."""
        vwap = calculate_vwap(uptrend_data)

        # First VWAP should equal typical price of first candle
        typical_price_first = (
            uptrend_data["High"].iloc[0]
            + uptrend_data["Low"].iloc[0]
            + uptrend_data["Close"].iloc[0]
        ) / 3
        assert abs(vwap.iloc[0] - typical_price_first) < 0.01

    def test_vwap_no_nan(self, sample_stock_data):
        """Test VWAP has no NaN values."""
        vwap = calculate_vwap(sample_stock_data)
        assert not vwap.isna().any()


class TestVolumeProfile:
    """Tests for Volume Profile calculation."""

    def test_volume_profile_basic(self, sample_stock_data):
        """Test basic volume profile calculation."""
        profile = calculate_volume_profile(sample_stock_data, num_bins=20)

        assert "price_levels" in profile
        assert "volumes" in profile
        assert len(profile["price_levels"]) == 20
        assert len(profile["volumes"]) == 20

    def test_volume_profile_total_volume(self, sample_stock_data):
        """Test total volume in profile matches input."""
        profile = calculate_volume_profile(sample_stock_data, num_bins=20)

        total_profile_volume = sum(profile["volumes"])
        total_input_volume = sample_stock_data["Volume"].sum()

        # Should be approximately equal (within 1% due to binning)
        assert abs(total_profile_volume - total_input_volume) / total_input_volume < 0.01

    def test_volume_profile_price_range(self, sample_stock_data):
        """Test volume profile covers the entire price range."""
        profile = calculate_volume_profile(sample_stock_data, num_bins=10)

        min_price_level = min(profile["price_levels"])
        max_price_level = max(profile["price_levels"])
        min_low = sample_stock_data["Low"].min()
        max_high = sample_stock_data["High"].max()

        assert min_price_level >= min_low
        assert max_price_level <= max_high

    def test_volume_profile_different_bins(self, sample_stock_data):
        """Test volume profile with different bin counts."""
        profile_10 = calculate_volume_profile(sample_stock_data, num_bins=10)
        profile_50 = calculate_volume_profile(sample_stock_data, num_bins=50)

        assert len(profile_10["price_levels"]) == 10
        assert len(profile_50["price_levels"]) == 50


class TestMFI:
    """Tests for Money Flow Index calculation."""

    def test_mfi_basic_calculation(self, sample_stock_data):
        """Test basic MFI calculation."""
        mfi = calculate_mfi(sample_stock_data, period=14)

        assert len(mfi) == len(sample_stock_data)
        assert isinstance(mfi, pd.Series)

    def test_mfi_range(self, sample_stock_data):
        """Test MFI is within 0-100 range."""
        mfi = calculate_mfi(sample_stock_data, period=14)

        # Remove NaN values from the beginning
        mfi_valid = mfi.dropna()

        assert all(mfi_valid >= 0)
        assert all(mfi_valid <= 100)

    def test_mfi_initial_nans(self, sample_stock_data):
        """Test MFI has NaN values at the beginning due to period."""
        mfi = calculate_mfi(sample_stock_data, period=14)

        # First value should be NaN (or close to it)
        assert pd.isna(mfi.iloc[0])

    def test_mfi_different_periods(self, uptrend_data):
        """Test MFI with different periods."""
        mfi_10 = calculate_mfi(uptrend_data, period=10)
        mfi_20 = calculate_mfi(uptrend_data, period=20)

        # Both should have values
        assert len(mfi_10.dropna()) > 0
        assert len(mfi_20.dropna()) > 0

        # Shorter period should have more non-NaN values
        assert len(mfi_10.dropna()) >= len(mfi_20.dropna())


class TestVPT:
    """Tests for Volume-Price Trend calculation."""

    def test_vpt_basic_calculation(self, sample_stock_data):
        """Test basic VPT calculation."""
        vpt = calculate_vpt(sample_stock_data)

        assert len(vpt) == len(sample_stock_data)
        assert vpt.iloc[0] == 0  # First value should be 0
        assert isinstance(vpt, pd.Series)

    def test_vpt_uptrend(self, uptrend_data):
        """Test VPT increases with uptrend."""
        vpt = calculate_vpt(uptrend_data)

        # VPT should increase with price
        assert vpt.iloc[-1] > vpt.iloc[0]

    def test_vpt_downtrend(self, downtrend_data):
        """Test VPT decreases with downtrend."""
        vpt = calculate_vpt(downtrend_data)

        # VPT should decrease with price decline
        assert vpt.iloc[-1] < vpt.iloc[0]


class TestVolumeTrends:
    """Tests for volume trend analysis."""

    def test_analyze_volume_trends_basic(self, sample_stock_data):
        """Test basic volume trend analysis."""
        trends = analyze_volume_trends(sample_stock_data, window=20)

        assert "current_volume" in trends
        assert "average_volume" in trends
        assert "volume_vs_average" in trends
        assert "volume_trend" in trends
        assert "divergence_detected" in trends

    def test_analyze_volume_trends_divergence_detection(self, uptrend_data):
        """Test divergence detection."""
        # Modify data to create divergence: price up, volume down
        data = uptrend_data.copy()
        data["Volume"] = [2000000 - i * 50000 for i in range(20)]

        trends = analyze_volume_trends(data, window=10)

        # Should detect divergence
        assert trends["divergence_detected"] is True
        assert "up" in trends["divergence_type"].lower()
        assert "down" in trends["divergence_type"].lower()

    def test_analyze_volume_trends_no_divergence(self, uptrend_data):
        """Test when there's no divergence."""
        trends = analyze_volume_trends(uptrend_data, window=10)

        # With price and volume both up, might not detect divergence
        # depending on the lookback window
        assert isinstance(trends["divergence_detected"], bool)

    def test_volume_vs_average_format(self, sample_stock_data):
        """Test volume vs average is formatted as percentage."""
        trends = analyze_volume_trends(sample_stock_data, window=15)

        assert "%" in trends["volume_vs_average"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

        with pytest.raises((IndexError, ValueError, ZeroDivisionError)):
            calculate_obv(empty_df)

    def test_single_row_dataframe(self):
        """Test handling of single row DataFrame."""
        single_row = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2024-01-01")],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1000000],
            }
        )

        obv = calculate_obv(single_row)
        assert len(obv) == 1
        assert obv.iloc[0] == 0

    def test_zero_volume(self):
        """Test handling of zero volume."""
        data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=5, freq="D"),
                "Open": [100, 101, 102, 101, 100],
                "High": [101, 102, 103, 102, 101],
                "Low": [99, 100, 101, 100, 99],
                "Close": [100, 101, 102, 101, 100],
                "Volume": [1000000, 0, 1000000, 0, 1000000],
            }
        )

        # Should handle zero volume gracefully
        vwap = calculate_vwap(data)
        assert not any(np.isinf(vwap))
