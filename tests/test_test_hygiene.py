"""Regression tests for suite-wide test isolation."""

import socket

import pytest
from pytest_socket import SocketBlockedError


def test_outbound_connection_is_blocked():
    """A real TCP connection attempt fails at the suite's socket guard."""
    with pytest.raises(SocketBlockedError):
        socket.create_connection(("127.0.0.1", 0), timeout=0.1)


def test_unix_sockets_remain_available():
    """Asyncio can still use Unix sockets for its event loop wakeups."""
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(b"wake")
        assert receiver.recv(4) == b"wake"
