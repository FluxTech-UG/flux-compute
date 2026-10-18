"""Suite-wide guard: the tests are hermetic and touch no network.

Every outbound socket connection raises, so a test that forgets to stub a network
call (the public-IP lookup, an OpenStack client, an SSH probe) fails loudly with the
address it tried to reach instead of passing or failing on the machine's
connectivity.

The follower's start time comes from `ps`, which the sandbox denies and which
the tests never run: `sweep._own_start` reads as the fixed `TEST_START`.
"""
import socket

import pytest


class NetworkAccessInTests(BaseException):
    """A BaseException so that production code's broad `except Exception` (the
    public-IP lookup has one) cannot swallow it and turn a missing stub into a
    silent sentinel value."""


TEST_START = "Sat Oct 18 09:00:00 2026"


@pytest.fixture(autouse=True)
def _no_ps(monkeypatch):
    from flux_compute import sweep
    monkeypatch.setattr(sweep, "_own_start", lambda: TEST_START)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(*args, **kwargs):
        raise NetworkAccessInTests(f"test attempted a network connection: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)
