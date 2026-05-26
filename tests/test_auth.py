"""Tests for OAuth2 token management and authenticated HTTP calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from orchestrator_agent.auth import OAuthTokenManager


@pytest.fixture
def _enable_oauth(monkeypatch):
    monkeypatch.setattr("orchestrator_agent.auth.oauth2lib_settings.OAUTH2_ACTIVE", True)
    monkeypatch.setattr("orchestrator_agent.auth.oauth2lib_settings.OAUTH2_TOKEN_URL", "https://idp.example.com/token")
    monkeypatch.setattr("orchestrator_agent.auth.oauth2lib_settings.OAUTH2_RESOURCE_SERVER_ID", "test-client")
    monkeypatch.setattr("orchestrator_agent.auth.oauth2lib_settings.OAUTH2_RESOURCE_SERVER_SECRET", "test-secret")


@pytest.fixture
def _disable_oauth(monkeypatch):
    monkeypatch.setattr("orchestrator_agent.auth.oauth2lib_settings.OAUTH2_ACTIVE", False)


def _make_response(status_code: int = 200, json: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://idp.example.com/token")
    return httpx.Response(status_code, json=json or {}, request=request)


@pytest.fixture
def token_response():
    return _make_response(200, {"access_token": "tok-123", "token_type": "bearer", "expires_in": 3600})


class TestOAuthTokenManager:
    async def test_get_token_returns_none_when_disabled(self, _disable_oauth):
        mgr = OAuthTokenManager()
        assert await mgr.get_token() is None

    async def test_get_token_fetches_and_caches(self, _enable_oauth, token_response):
        mgr = OAuthTokenManager()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = token_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("orchestrator_agent.auth.httpx.AsyncClient", return_value=mock_client):
            token = await mgr.get_token()
            assert token == "tok-123"

            # Second call should use cache, not fetch again
            token2 = await mgr.get_token()
            assert token2 == "tok-123"
            assert mock_client.post.call_count == 1

    async def test_refresh_token_clears_cache(self, _enable_oauth, token_response):
        mgr = OAuthTokenManager()
        mgr._token = "old-token"

        new_response = _make_response(200, {"access_token": "new-tok", "token_type": "bearer"})
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = new_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("orchestrator_agent.auth.httpx.AsyncClient", return_value=mock_client):
            token = await mgr.refresh_token()
            assert token == "new-tok"
            assert mgr._token == "new-tok"

    async def test_get_auth_headers_with_token(self):
        mgr = OAuthTokenManager()
        mgr._token = "my-token"
        assert mgr.get_auth_headers() == {"Authorization": "Bearer my-token"}

    async def test_get_auth_headers_without_token(self):
        mgr = OAuthTokenManager()
        assert mgr.get_auth_headers() == {}

    async def test_fetch_token_sends_correct_payload(self, _enable_oauth):
        mgr = OAuthTokenManager()
        response = _make_response(200, {"access_token": "t", "token_type": "bearer"})

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("orchestrator_agent.auth.httpx.AsyncClient", return_value=mock_client):
            await mgr.get_token()

        mock_client.post.assert_called_once_with(
            "https://idp.example.com/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "test-client",
                "client_secret": "test-secret",
            },
            timeout=30,
        )
