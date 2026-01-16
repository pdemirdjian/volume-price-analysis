"""Pytest configuration and fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")

    # Create realistic-looking price data
    base_price = 100.0
    prices = []
    volumes = []

    for i in range(30):
        # Add some randomness but keep it realistic
        price_change = (i % 5 - 2) * 2  # Oscillating pattern
        current_price = base_price + price_change + i * 0.5

        prices.append(
            {
                "Open": current_price - 1,
                "High": current_price + 2,
                "Low": current_price - 2,
                "Close": current_price,
            }
        )

        # Volume inversely correlated with price for some divergence testing
        volumes.append(1000000 + (30 - i) * 50000)

    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": [p["Open"] for p in prices],
            "High": [p["High"] for p in prices],
            "Low": [p["Low"] for p in prices],
            "Close": [p["Close"] for p in prices],
            "Volume": volumes,
        }
    )

    return data


@pytest.fixture
def uptrend_data():
    """Create data with clear uptrend and increasing volume."""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="D")

    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100 + i * 2 for i in range(20)],
            "High": [102 + i * 2 for i in range(20)],
            "Low": [99 + i * 2 for i in range(20)],
            "Close": [101 + i * 2 for i in range(20)],
            "Volume": [1000000 + i * 100000 for i in range(20)],
        }
    )

    return data


@pytest.fixture
def downtrend_data():
    """Create data with clear downtrend and increasing volume."""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="D")

    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": [200 - i * 2 for i in range(20)],
            "High": [202 - i * 2 for i in range(20)],
            "Low": [199 - i * 2 for i in range(20)],
            "Close": [201 - i * 2 for i in range(20)],
            "Volume": [1000000 + i * 100000 for i in range(20)],
        }
    )

    return data


@pytest.fixture
def flat_price_data():
    """Create data with flat price but varying volume."""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="D")

    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0] * 20,
            "High": [101.0] * 20,
            "Low": [99.0] * 20,
            "Close": [100.0] * 20,
            "Volume": [1000000 + (i % 5) * 200000 for i in range(20)],
        }
    )

    return data
