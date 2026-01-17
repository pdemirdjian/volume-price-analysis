"""Stock data fetching functionality using yfinance."""

import logging
import re

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Default timeout for yfinance requests (in seconds)
DEFAULT_TIMEOUT = 30

# Valid symbol pattern: starts with letter (e.g., AAPL, BRK-B) or ^ for indices (e.g., ^GSPC)
SYMBOL_PATTERN = re.compile(r"^(?:[A-Za-z][A-Za-z0-9.^-]{0,9}|\^[A-Za-z][A-Za-z0-9.-]{0,8})$")


def validate_symbol_format(symbol: str) -> bool:
    """
    Validate that a stock symbol has a valid format.

    Args:
        symbol: Stock ticker symbol to validate

    Returns:
        True if the symbol format is valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    return bool(SYMBOL_PATTERN.match(symbol))


def fetch_stock_data(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "1mo",
    timeout: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """
    Fetch historical stock data for a given symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        period: Period to fetch if dates not specified (e.g., '1mo', '3mo', '1y')
        timeout: Request timeout in seconds (default: 30)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Date

    Raises:
        ValueError: If symbol format is invalid or no data found
    """
    # Validate symbol format
    if not validate_symbol_format(symbol):
        logger.warning("Invalid symbol format rejected: %s", symbol)
        raise ValueError(
            f"Invalid symbol format: '{symbol}'. "
            "Symbols must be 1-10 alphanumeric characters (may include . - ^)"
        )

    logger.debug("Fetching data for %s (period=%s, timeout=%ds)", symbol, period, timeout)
    ticker = yf.Ticker(symbol)

    if start_date and end_date:
        logger.debug("Using date range: %s to %s", start_date, end_date)
        data = ticker.history(start=start_date, end=end_date, timeout=timeout)
    else:
        data = ticker.history(period=period, timeout=timeout)

    if data.empty:
        logger.warning("No data returned for symbol: %s", symbol)
        raise ValueError(f"No data found for symbol {symbol}")

    logger.debug("Fetched %d rows for %s", len(data), symbol)

    # Reset index to make Date a column
    data = data.reset_index()

    # Keep only the columns we need
    columns_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    data = data[columns_to_keep]

    return data


def validate_symbol(symbol: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """
    Validate that a stock symbol exists and has data.

    Args:
        symbol: Stock ticker symbol
        timeout: Request timeout in seconds (default: 30)

    Returns:
        True if symbol is valid, False otherwise
    """
    # First check format
    if not validate_symbol_format(symbol):
        return False

    try:
        ticker = yf.Ticker(symbol)
        # Use fast_info which is lighter weight than info
        info = ticker.fast_info
        return hasattr(info, "last_price") and info.last_price is not None
    except Exception:
        return False
