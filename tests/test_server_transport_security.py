from argparse import Namespace

import pytest

from src.lindorm_mcp_server import server


def _args(**overrides):
    values = {"transport": None, "host": None, "port": None}
    values.update(overrides)
    return Namespace(**values)


def test_configure_transport_defaults_to_stdio_localhost(monkeypatch):
    for key in ("SERVER_TRANSPORT", "SERVER_HOST", "SERVER_PORT", "API_KEY", "ALLOW_PUBLIC_BINDING"):
        monkeypatch.delenv(key, raising=False)

    assert server._configure_transport(_args()) == "stdio"
    assert server.mcp.settings.host == "127.0.0.1"


def test_configure_transport_requires_api_key_for_sse(monkeypatch):
    monkeypatch.setenv("SERVER_TRANSPORT", "sse")
    monkeypatch.delenv("API_KEY", raising=False)

    with pytest.raises(ValueError, match="API_KEY is required"):
        server._configure_transport(_args())


def test_configure_transport_rejects_public_bind_without_opt_in(monkeypatch):
    monkeypatch.setenv("SERVER_TRANSPORT", "sse")
    monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("API_KEY", "s" * 32)
    monkeypatch.delenv("ALLOW_PUBLIC_BINDING", raising=False)

    with pytest.raises(ValueError, match="Refusing public bind"):
        server._configure_transport(_args())


def test_configure_transport_allows_authenticated_local_sse(monkeypatch):
    monkeypatch.setenv("SERVER_TRANSPORT", "sse")
    monkeypatch.setenv("SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("API_KEY", "s" * 32)
    monkeypatch.delenv("ALLOW_PUBLIC_BINDING", raising=False)

    assert server._configure_transport(_args()) == "sse"
    assert server.mcp.settings.host == "127.0.0.1"
    assert server.mcp.api_key == "s" * 32


def test_configure_transport_rejects_short_api_key(monkeypatch):
    monkeypatch.setenv("SERVER_TRANSPORT", "sse")
    monkeypatch.setenv("SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("API_KEY", "short")

    with pytest.raises(ValueError, match="at least 32 characters"):
        server._configure_transport(_args())
