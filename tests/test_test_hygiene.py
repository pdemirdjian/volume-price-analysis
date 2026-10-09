"""Regression tests for suite-wide test isolation."""

import socket

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
