"""Core's GraphQL API, called as the person asking: the URL, the token, the answer or its errors."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from orchestrator_agent.auth import token_manager
from orchestrator_agent.form_fill.widgets.graphql import CoreGraphQL, GraphQLError, graphql_url
from orchestrator_agent.mcp_client import bind_outbound_token

URL = "https://core.example.com/api/graphql"


@pytest.mark.parametrize(
    "mcp_url,configured,expected",
    [
        pytest.param("http://core:8080/mcp", None, "http://core:8080/api/graphql", id="derived"),
        pytest.param("http://core:8080/mcp/", None, "http://core:8080/api/graphql", id="trailing-slash"),
        pytest.param("http://core:8080/mcp", "http://gql/x", "http://gql/x", id="configured-wins"),
    ],
)
def test_the_url_follows_the_mcp_url_unless_configured(mcp_url, configured, expected):
    assert graphql_url(mcp_url, configured) == expected


def _client(handler) -> tuple[CoreGraphQL, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        # A snapshot: the auth flow retries on this very request, and sets its Authorization header again.
        seen.append(httpx.Request(request.method, request.url, headers=request.headers, content=request.content))
        return handler(request, len(seen))

    return CoreGraphQL(URL, transport=httpx.MockTransport(record)), seen


@pytest.fixture(autouse=True)
def no_service_token(monkeypatch):
    """No outbound OAuth2 unless a test turns it on: nothing may go looking for a token URL."""
    monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", False)


async def test_the_query_goes_out_with_the_callers_token_and_its_data_comes_back():
    client, seen = _client(lambda request, n: httpx.Response(200, json={"data": {"customers": {"page": []}}}))
    with bind_outbound_token("user-token"):
        assert await client("query { customers { page { customerId } } }", {"first": 5}) == {"customers": {"page": []}}
    (request,) = seen
    assert request.headers["authorization"] == "Bearer user-token"
    assert json.loads(request.content) == {
        "query": "query { customers { page { customerId } } }",
        "variables": {"first": 5},
    }


async def test_the_service_token_is_refreshed_once_on_401(monkeypatch):
    monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", True)
    monkeypatch.setattr(token_manager, "get_token", AsyncMock(return_value="old"))
    monkeypatch.setattr(token_manager, "refresh_token", AsyncMock(return_value="new"))
    client, seen = _client(lambda request, n: httpx.Response(401 if n == 1 else 200, json={"data": {}}))
    assert await client("query { version { applicationVersions } }") == {}
    assert [request.headers["authorization"] for request in seen] == ["Bearer old", "Bearer new"]


@pytest.mark.parametrize(
    "response,error",
    [
        pytest.param(httpx.Response(200, json={"errors": [{"message": "Not authorized"}]}), GraphQLError, id="errors"),
        pytest.param(httpx.Response(200, json={}), GraphQLError, id="no-data"),
        pytest.param(httpx.Response(500, text="boom"), httpx.HTTPStatusError, id="status"),
    ],
)
async def test_a_failed_answer_raises(response, error):
    client, _ = _client(lambda request, n: response)
    with pytest.raises(error):
        await client("query { customers { page { customerId } } }")
