"""Tests for the shared MCP client: outbound token binding."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest

from orchestrator_agent.mcp_client import _outbound_token, bind_outbound_token


@pytest.mark.parametrize("token,expected", [("tok-123", "tok-123"), (None, None)])
def test_bind_outbound_token(token, expected):
    assert _outbound_token.get() is None
    with bind_outbound_token(token):
        assert _outbound_token.get() == expected
    assert _outbound_token.get() is None
