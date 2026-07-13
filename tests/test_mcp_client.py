"""Tests for the shared MCP client: outbound token binding."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from unittest.mock import AsyncMock

import httpx
import pytest

from orchestrator_agent.auth import token_manager
from orchestrator_agent.mcp_client import _ContextVarBearerAuth, _outbound_token, bind_outbound_token


@pytest.mark.parametrize("token,expected", [("tok-123", "tok-123"), (None, None)])
def test_bind_outbound_token(token, expected):
    assert _outbound_token.get() is None
    with bind_outbound_token(token):
        assert _outbound_token.get() == expected
    assert _outbound_token.get() is None


async def _run_request(auth: httpx.Auth, handler) -> tuple[httpx.Response, list[str | None]]:
    """Drive a real httpx request through ``auth`` and return the response plus each attempt's Authorization header."""
    seen_headers = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("authorization"))
        return handler(len(seen_headers))

    async with httpx.AsyncClient(transport=httpx.MockTransport(wrapped), auth=auth) as client:
        response = await client.get("https://core.example.com/mcp/")
    return response, seen_headers


@pytest.mark.parametrize(
    "first_status,expect_refresh",
    [
        pytest.param(401, True, id="401-refreshes-and-retries"),
        pytest.param(403, True, id="403-refreshes-and-retries"),
        pytest.param(200, False, id="200-no-refresh"),
        pytest.param(404, False, id="404-no-refresh"),
    ],
)
async def test_service_token_refreshed_on_auth_error(monkeypatch, first_status, expect_refresh):
    monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", True)
    monkeypatch.setattr(token_manager, "get_token", AsyncMock(return_value="old-token"))
    refresh_mock = AsyncMock(return_value="new-token")
    monkeypatch.setattr(token_manager, "refresh_token", refresh_mock)

    response, seen_headers = await _run_request(
        _ContextVarBearerAuth(), lambda attempt: httpx.Response(first_status if attempt == 1 else 200)
    )

    expected_headers = ["Bearer old-token", "Bearer new-token"] if expect_refresh else ["Bearer old-token"]
    assert seen_headers == expected_headers
    assert refresh_mock.await_count == (1 if expect_refresh else 0)
    assert response.status_code == (first_status if not expect_refresh else 200)


async def test_per_run_token_not_refreshed_on_401(monkeypatch):
    refresh_mock = AsyncMock(return_value="new-token")
    monkeypatch.setattr(token_manager, "refresh_token", refresh_mock)

    with bind_outbound_token("user-token"):
        response, seen_headers = await _run_request(_ContextVarBearerAuth(), lambda _attempt: httpx.Response(401))

    assert seen_headers == ["Bearer user-token"]
    refresh_mock.assert_not_awaited()
    assert response.status_code == 401


async def test_no_retry_when_outbound_auth_disabled(monkeypatch):
    monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", False)
    monkeypatch.setattr(token_manager, "get_token", AsyncMock(return_value=None))
    refresh_mock = AsyncMock(return_value="new-token")
    monkeypatch.setattr(token_manager, "refresh_token", refresh_mock)

    response, seen_headers = await _run_request(_ContextVarBearerAuth(), lambda _attempt: httpx.Response(401))

    assert seen_headers == [None]
    refresh_mock.assert_not_awaited()
    assert response.status_code == 401
