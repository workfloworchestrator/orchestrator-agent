"""Tests for the startup MCP tool-contract verification."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest

from orchestrator_agent.mcp_client import verify_tool_contract
from orchestrator_agent.tool_names import ALL_TOOL_NAMES


class _FakeToolset:
    """Minimal stand-in for MCPToolset: an async context manager that lists tools (or fails)."""

    def __init__(self, names=None, error=None):
        self._names = names
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self):
        if self._error is not None:
            raise self._error
        return [SimpleNamespace(name=n) for n in self._names]


async def test_passes_when_all_tools_present():
    await verify_tool_contract(_FakeToolset(names=list(ALL_TOOL_NAMES)))  # no raise


async def test_raises_when_a_tool_is_missing():
    present = list(ALL_TOOL_NAMES)[:-1]  # drop one
    with pytest.raises(RuntimeError, match=ALL_TOOL_NAMES[-1]):
        await verify_tool_contract(_FakeToolset(names=present))


async def test_unreachable_server_is_warned_not_raised():
    # A connection failure is operational, not contract drift — must not block startup.
    await verify_tool_contract(_FakeToolset(error=ConnectionError("core down")))  # no raise


async def test_non_connection_error_is_raised():
    # A non-connection failure (auth, bad response, bug) is a real misconfig — fail loud at startup.
    with pytest.raises(ValueError, match="boom"):
        await verify_tool_contract(_FakeToolset(error=ValueError("boom")))
