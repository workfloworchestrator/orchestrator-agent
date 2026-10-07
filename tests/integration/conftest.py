"""The orchestrator-core of ``tests/integration/docker-compose.yml``; every test here is skipped without it."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from pydantic_ai.mcp import MCPToolset

from orchestrator_agent.form_fill.widgets import CoreGraphQL, WidgetContext

CORE_URL = os.environ.get("WFO_INTEGRATION_CORE_URL")


def pytest_collection_modifyitems(config, items):
    here = os.path.dirname(__file__)
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.integration)
            if not CORE_URL:
                item.add_marker(pytest.mark.skip(reason="WFO_INTEGRATION_CORE_URL is not set"))


@pytest.fixture(autouse=True)
def authless(monkeypatch):
    """The stack runs authless: no service token to fetch for GraphQL.

    An environment default would come too late: the root conftest has built ``agent_settings`` by now.
    """
    monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", False)


@pytest.fixture
def core() -> MCPToolset:
    return MCPToolset(f"{CORE_URL}/mcp")  # core answers on /mcp/; the MCP client follows its redirect from /mcp


@pytest.fixture
def graphql() -> CoreGraphQL:
    return CoreGraphQL(f"{CORE_URL}/api/graphql")


@pytest.fixture
def ctx(core, graphql) -> WidgetContext:
    return WidgetContext(call_tool=core.direct_call_tool, graphql=graphql)
