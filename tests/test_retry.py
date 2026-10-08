"""Bounded retries at the callable boundary."""

from unittest.mock import Mock, call

from volume_price_analysis.retry import retry_call


def test_recovers_with_exponential_backoff(caplog):
    fn = Mock(side_effect=[OSError("private body"), TimeoutError("secret"), "done"])
    sleep = Mock()

    assert retry_call(fn, attempts=3, base_delay=2, retry_on=lambda e: True, sleep=sleep) == "done"
    assert fn.call_count == 3
    assert sleep.call_args_list == [call(2), call(4)]
    assert "attempt 1/3" in caplog.text
    assert "attempt 2/3" in caplog.text
    assert "OSError" in caplog.text
    assert "TimeoutError" in caplog.text
    assert "private body" not in caplog.text
    assert "secret" not in caplog.text


def test_exhaustion_reraises_last_error():
    import pytest

    errors = [OSError("first"), OSError("second"), OSError("last")]
    fn = Mock(side_effect=errors)
    sleep = Mock()
    with pytest.raises(OSError) as raised:
        retry_call(fn, attempts=3, base_delay=2, retry_on=lambda e: True, sleep=sleep)
    assert raised.value is errors[-1]
    assert fn.call_count == 3
    assert sleep.call_args_list == [call(2), call(4)]


def test_non_retryable_error_propagates_without_sleep():
    import pytest

    error = ValueError("bad input")
    fn = Mock(side_effect=error)
    sleep = Mock()
    with pytest.raises(ValueError) as raised:
        retry_call(
            fn, attempts=3, base_delay=2, retry_on=lambda e: isinstance(e, OSError), sleep=sleep
        )
    assert raised.value is error
    fn.assert_called_once()
    sleep.assert_not_called()
