"""Stock data fetching: the single seam between this package and Yahoo Finance.

All market data enters the package through the :class:`DataSource` protocol.
``YFinanceDataSource`` is the production adapter (yfinance); ``InMemoryDataSource``
is the test adapter, so tests can inject data instead of patching import paths.
"""

import datetime
import logging
import re
from typing import Protocol, runtime_checkable

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Default timeout for yfinance requests (in seconds)
DEFAULT_TIMEOUT = 30

# Valid symbol pattern: starts with letter (e.g., AAPL, BRK-B) or ^ for indices (e.g., ^GSPC)
SYMBOL_PATTERN = re.compile(r"^(?:[A-Za-z][A-Za-z0-9.^-]{0,9}|\^[A-Za-z][A-Za-z0-9.-]{0,8})$")

# Valid period values accepted by yfinance
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}

# Date format: YYYY-MM-DD
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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


@runtime_checkable
class DataSource(Protocol):
    """The one seam through which market data enters this package.

    Every module that needs prices or earnings dates depends on this protocol,
    never on yfinance directly. Production code gets ``YFinanceDataSource``;
    tests get ``InMemoryDataSource``.

    DataFrame contract for :meth:`fetch` — every implementation must honour it,
    because every downstream indicator assumes it:

    - Columns are exactly ``Date, Open, High, Low, Close, Volume``, in that
      order. ``Date`` is a promoted index column (yfinance returns the date as
      the index; the adapter calls ``reset_index()``), so rows are addressed
      positionally (``.iloc``), never by date label.
    - Rows are ascending by date, one row per trading session; the last row is
      the most recent session. Non-trading days are absent, not zero-filled.
    - ``Open/High/Low/Close`` are floats and ``Volume`` is numeric. Values are
      split- and dividend-adjusted exactly as yfinance returns them.
    - The frame is never empty: an empty result is an error, not a value.
    - Every failure — bad symbol, network error, missing OHLCV columns, no
      data — surfaces as ``ValueError``. Callers catch ``ValueError`` alone and
      never see provider-specific exception types.
    - No retries and no caching: one call is one fetch. Callers needing either
      implement it themselves.

    :meth:`earnings_date` returns a timezone-aware ``datetime`` for the next
    known earnings event, or ``None`` when the provider has no usable date.
    Unlike ``fetch``, it reports "unknown" rather than raising.
    """

    def fetch(
        self,
        symbol: str,
        *,
        period: str = "1mo",
        start: str | None = None,
        end: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> pd.DataFrame:
        """Fetch OHLCV history for ``symbol``. See the class docstring contract."""
        ...

    def earnings_date(self, symbol: str) -> datetime.datetime | None:
        """Return the next known earnings datetime (tz-aware), or None if unknown."""
        ...


class YFinanceDataSource:
    """Production ``DataSource`` backed by yfinance."""

    def fetch(
        self,
        symbol: str,
        *,
        period: str = "1mo",
        start: str | None = None,
        end: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            period: Period to fetch if dates not specified (e.g., '1mo', '3mo', '1y')
            start: Start date in 'YYYY-MM-DD' format
            end: End date in 'YYYY-MM-DD' format
            timeout: Request timeout in seconds (default: 30)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume

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

        # Validate that both dates are provided together
        if bool(start) != bool(end):
            raise ValueError("Both start_date and end_date must be provided together")

        # Validate date formats and calendar validity
        if start and not DATE_PATTERN.match(start):
            raise ValueError(f"Invalid start_date format: '{start}'. Must be YYYY-MM-DD")
        if end and not DATE_PATTERN.match(end):
            raise ValueError(f"Invalid end_date format: '{end}'. Must be YYYY-MM-DD")

        if start and end:
            try:
                parsed_start = datetime.date.fromisoformat(start)
            except ValueError:
                raise ValueError(f"Invalid calendar date for start_date: '{start}'") from None
            try:
                parsed_end = datetime.date.fromisoformat(end)
            except ValueError:
                raise ValueError(f"Invalid calendar date for end_date: '{end}'") from None
            if parsed_end < parsed_start:
                raise ValueError("end_date must be on or after start_date")

        # Validate period only when date range is not provided
        if not (start and end) and period not in VALID_PERIODS:
            raise ValueError(
                f"Invalid period: '{period}'. Must be one of: {', '.join(sorted(VALID_PERIODS))}"
            )

        logger.debug("Fetching data for %s (period=%s, timeout=%ds)", symbol, period, timeout)
        ticker = yf.Ticker(symbol)

        try:
            if start and end:
                logger.debug("Using date range: %s to %s", start, end)
                data = ticker.history(start=start, end=end, timeout=timeout)
            else:
                data = ticker.history(period=period, timeout=timeout)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {e}") from e

        if data.empty:
            logger.warning("No data returned for symbol: %s", symbol)
            raise ValueError(f"No data found for symbol {symbol}")

        logger.debug("Fetched %d rows for %s", len(data), symbol)

        # Reset index to make Date a column
        data = data.reset_index()

        # Keep only the columns we need
        columns_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
        critical_columns = {"Open", "High", "Low", "Close", "Volume"}
        missing_critical = critical_columns - set(data.columns)
        if missing_critical:
            raise ValueError(
                f"Data for {symbol} is missing critical columns: {sorted(missing_critical)}"
            )
        data = data[[c for c in columns_to_keep if c in data.columns]]

        return data

    def earnings_date(self, symbol: str) -> datetime.datetime | None:
        """
        Return the next known earnings datetime for ``symbol``, or None.

        Yahoo reports the date as either a POSIX timestamp or a datetime, and
        sometimes as a list (an estimated range) — the first entry is taken.
        Naive datetimes are assumed to be UTC. Any provider failure or
        unparseable value is reported as ``None``, never raised.
        """
        try:
            info = yf.Ticker(symbol).info
            raw = info.get("earningsDate") or info.get("earningsTimestamp")
            if raw is None:
                return None

            # yfinance may return a list (range) or a single value
            if isinstance(raw, list):
                raw = raw[0]

            # Normalise to an aware datetime
            if isinstance(raw, (int, float)):
                return datetime.datetime.fromtimestamp(raw, tz=datetime.UTC)
            if isinstance(raw, datetime.datetime):
                return raw if raw.tzinfo else raw.replace(tzinfo=datetime.UTC)
            return None
        except Exception:
            logger.debug("Earnings lookup failed for %s", symbol, exc_info=True)
            return None


class InMemoryDataSource:
    """Test ``DataSource`` backed by in-memory dicts.

    Lives in ``src`` (not ``tests``) so it ships with the package and any test
    module can import it. Injecting one of these replaces patching
    ``fetch_stock_data`` at each call site.

    Args:
        frames: symbol -> OHLCV DataFrame honouring the ``DataSource`` contract.
        earnings: symbol -> earnings datetime (tz-aware by convention).
        errors: symbol -> exception raised by ``fetch`` instead of returning
            data, for exercising failure paths.
    """

    def __init__(
        self,
        frames: dict[str, pd.DataFrame] | None = None,
        earnings: dict[str, datetime.datetime | None] | None = None,
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self.frames: dict[str, pd.DataFrame] = dict(frames or {})
        self.earnings: dict[str, datetime.datetime | None] = dict(earnings or {})
        self.errors: dict[str, Exception] = dict(errors or {})
        self.fetch_calls: list[str] = []
        self.earnings_calls: list[str] = []

    def fetch(
        self,
        symbol: str,
        *,
        period: str = "1mo",
        start: str | None = None,
        end: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> pd.DataFrame:
        """Return the configured frame for ``symbol``.

        Mirrors production by raising ``ValueError`` for an unknown symbol.
        A symbol configured in ``errors`` raises that exception instead.
        """
        self.fetch_calls.append(symbol)
        if symbol in self.errors:
            raise self.errors[symbol]
        if symbol not in self.frames:
            raise ValueError(f"No data found for symbol {symbol}")
        return self.frames[symbol].copy()

    def earnings_date(self, symbol: str) -> datetime.datetime | None:
        """Return the configured earnings datetime, or None when unconfigured."""
        self.earnings_calls.append(symbol)
        return self.earnings.get(symbol)


# Module-level production adapter behind the thin function wrappers below.
_DEFAULT_DATA_SOURCE = YFinanceDataSource()


def get_default_data_source() -> DataSource:
    """Return the production ``DataSource``, used when a caller passes None."""
    return _DEFAULT_DATA_SOURCE


def fetch_stock_data(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "1mo",
    timeout: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """
    Fetch historical stock data via the default production data source.

    Thin wrapper over ``YFinanceDataSource.fetch`` kept for callers (notably
    ``server.py``) that do not thread a ``DataSource`` through.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        period: Period to fetch if dates not specified (e.g., '1mo', '3mo', '1y')
        timeout: Request timeout in seconds (default: 30)

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume

    Raises:
        ValueError: If symbol format is invalid or no data found
    """
    return _DEFAULT_DATA_SOURCE.fetch(
        symbol, period=period, start=start_date, end=end_date, timeout=timeout
    )


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
