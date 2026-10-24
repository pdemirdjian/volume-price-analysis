"""Regression tests for suite-wide test isolation."""

import socket
from pathlib import Path

import pytest
from pytest_socket import SocketConnectBlockedError


def test_outbound_connection_is_blocked():
    """A real TCP connection attempt fails at the suite's socket guard."""
    with pytest.raises(SocketConnectBlockedError):
        socket.create_connection(("203.0.113.1", 443), timeout=0.1)


def test_socketpair_remains_available():
    """Asyncio wakeups work with Unix or Windows loopback socket pairs."""
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(b"wake")
        assert receiver.recv(4) == b"wake"


_CONFTEST = Path(__file__).with_name("conftest.py")

# Each inner test reports whether pytest-socket let it reach a non-loopback host.
_GATED_TESTS = """
import socket

import pytest
from pytest_socket import SocketConnectBlockedError


def _outbound_allowed():
    try:
        socket.create_connection(("203.0.113.1", 443), timeout=0.01).close()
    except SocketConnectBlockedError:
        return False
    except OSError:
        pass  # Socket guard lifted; the unroutable host just doesn't answer.
    return True


@pytest.mark.integration
def test_network():
    assert _outbound_allowed()


def test_unit():
    assert not _outbound_allowed()
"""


@pytest.fixture
def gated_suite(pytester):
    """An inner suite using this repo's conftest, markers, and socket guard."""
    pytester.makeconftest(_CONFTEST.read_text(encoding="utf-8"))
    pytester.makeini(
        """
        [pytest]
        addopts = --strict-markers --allow-hosts=127.0.0.1,::1 --allow-unix-socket
        markers =
            integration: requires real network access; run with --run-integration
        """
    )
    pytester.makepyfile(_GATED_TESTS)
    return pytester


class TestIntegrationGating:
    """`integration` tests are opt-in and the only ones that get real network."""

    def test_default_run_skips_integration_tests(self, gated_suite):
        result = gated_suite.runpytest_subprocess("-rs", "-p", "no:xdist", "-p", "no:cov")

        result.assert_outcomes(passed=1, skipped=1)
        result.stdout.fnmatch_lines(["*SKIPPED*--run-integration*"])

    def test_run_integration_enables_network_for_integration_tests_only(self, gated_suite):
        result = gated_suite.runpytest_subprocess(
            "--run-integration", "-p", "no:xdist", "-p", "no:cov"
        )

        result.assert_outcomes(passed=2)

    def test_marker_selection_runs_only_integration_tests(self, gated_suite):
        result = gated_suite.runpytest_subprocess(
            "--run-integration", "-m", "integration", "-p", "no:xdist", "-p", "no:cov"
        )

        result.assert_outcomes(passed=1, deselected=1)
