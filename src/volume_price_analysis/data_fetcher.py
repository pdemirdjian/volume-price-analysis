"""Stock data fetching functionality using yfinance."""

import pandas as pd
import yfinance as yf


def fetch_stock_data(
    symbol: str, start_date: str | None = None, end_date: str | None = None, period: str = "1mo"
) -> pd.DataFrame:
    """
    Fetch historical stock data for a given symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        period: Period to fetch if dates not specified (e.g., '1mo', '3mo', '1y')

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Date
    """
    ticker = yf.Ticker(symbol)

    if start_date and end_date:
        data = ticker.history(start=start_date, end=end_date)
    else:
        data = ticker.history(period=period)

    if data.empty:
        raise ValueError(f"No data found for symbol {symbol}")

    # Reset index to make Date a column
    data = data.reset_index()

    # Keep only the columns we need
    columns_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    data = data[columns_to_keep]

    return data


def validate_symbol(symbol: str) -> bool:
    """
    Validate that a stock symbol exists and has data.

    Args:
        symbol: Stock ticker symbol

    Returns:
        True if symbol is valid, False otherwise
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return "symbol" in info or "shortName" in info
    except Exception:
        return False
