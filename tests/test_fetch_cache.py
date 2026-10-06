"""Price history caching at the production data source boundary."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pandas as pd
import pytest

from volume_price_analysis.data_fetcher import YFinanceDataSource


def test_history_is_reused_and_expires(sample_stock_data):
    now = [0.0]
    source = YFinanceDataSource(cache_ttl=900, clock=lambda: now[0])
    with patch("volume_price_analysis.data_fetcher.yf.Ticker") as provider:
        provider.return_value.history.return_value = sample_stock_data.set_index("Date")
        first = source.fetch("AAPL", period="3mo")
        first.loc[0, "Close"] = -1
        now[0] = 899
        pd.testing.assert_frame_equal(source.fetch("AAPL", period="3mo"), sample_stock_data)
        assert provider.call_count == 1
        now[0] = 900
        source.fetch("AAPL", period="3mo")
        assert provider.call_count == 2


def test_clear_cache_forces_refetch(sample_stock_data):
    source = YFinanceDataSource()
    with patch("volume_price_analysis.data_fetcher.yf.Ticker") as provider:
        provider.return_value.history.return_value = sample_stock_data.set_index("Date")
        source.fetch("AAPL")
        source.clear_cache()
        source.fetch("AAPL")
        assert provider.call_count == 2


def test_each_fetch_parameter_has_a_distinct_key(sample_stock_data):
    source = YFinanceDataSource()
    requests = [
        ("AAPL", {}),
        ("MSFT", {}),
        ("AAPL", {"period": "3mo"}),
        ("AAPL", {"start": "2024-01-01", "end": "2024-02-01"}),
        ("AAPL", {"start": "2024-01-02", "end": "2024-02-01"}),
        ("AAPL", {"start": "2024-01-01", "end": "2024-02-02"}),
    ]
    with patch("volume_price_analysis.data_fetcher.yf.Ticker") as provider:
        provider.return_value.history.return_value = sample_stock_data.set_index("Date")
        for _ in range(2):
            for symbol, args in requests:
                source.fetch(symbol, **args)
        assert provider.call_count == len(requests)


def test_failures_and_empty_results_are_retried(sample_stock_data):
    source = YFinanceDataSource()
    with patch("volume_price_analysis.data_fetcher.yf.Ticker") as provider:
        provider.return_value.history.side_effect = [
            ConnectionError("offline"),
            pd.DataFrame(),
            sample_stock_data.set_index("Date"),
        ]
        for message in ["Failed to fetch", "No data found"]:
            with pytest.raises(ValueError, match=message):
                source.fetch("AAPL")
        source.fetch("AAPL")
        source.fetch("AAPL")
        assert provider.return_value.history.call_count == 3


def test_concurrent_fetches_share_history_and_return_independent_frames(sample_stock_data):
    source = YFinanceDataSource()
    barrier = Barrier(4)

    def fetch():
        barrier.wait(timeout=5)
        return source.fetch("AAPL", period="3mo")

    with patch("volume_price_analysis.data_fetcher.yf.Ticker") as provider:
        provider.return_value.history.return_value = sample_stock_data.set_index("Date")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch) for _ in range(4)]
            frames = [future.result(timeout=5) for future in futures]
        assert provider.call_count == 1
        frames[0].loc[0, "Close"] = -1
        for frame in frames[1:]:
            pd.testing.assert_frame_equal(frame, sample_stock_data)


def test_ttl_environment_configuration_can_disable_cache(monkeypatch, sample_stock_data):
    monkeypatch.setenv("DATA_CACHE_TTL_SECONDS", "0")
    source = YFinanceDataSource()
    with patch("volume_price_analysis.data_fetcher.yf.Ticker") as provider:
        provider.return_value.history.return_value = sample_stock_data.set_index("Date")
        source.fetch("AAPL")
        source.fetch("AAPL")
        assert provider.call_count == 2


@pytest.mark.parametrize("raw", ["", "15m", "-1", "inf", "nan"])
def test_invalid_ttl_environment_falls_back_to_900_seconds(
    monkeypatch, caplog, sample_stock_data, raw
):
    monkeypatch.setenv("DATA_CACHE_TTL_SECONDS", raw)
    now = [0.0]
    source = YFinanceDataSource(clock=lambda: now[0])
    assert "DATA_CACHE_TTL_SECONDS" in caplog.text
    assert "900" in caplog.text
    with patch("volume_price_analysis.data_fetcher.yf.Ticker") as provider:
        provider.return_value.history.return_value = sample_stock_data.set_index("Date")
        source.fetch("AAPL")
        now[0] = 899
        source.fetch("AAPL")
        assert provider.call_count == 1
        now[0] = 900
        source.fetch("AAPL")
        assert provider.call_count == 2


def test_timeout_does_not_change_cache_key(sample_stock_data):
    source = YFinanceDataSource()
    with patch("volume_price_analysis.data_fetcher.yf.Ticker") as provider:
        provider.return_value.history.return_value = sample_stock_data.set_index("Date")
        source.fetch("AAPL", timeout=12)
        pd.testing.assert_frame_equal(source.fetch("AAPL", timeout=30), sample_stock_data)
        provider.return_value.history.assert_called_once_with(period="1mo", timeout=12)
