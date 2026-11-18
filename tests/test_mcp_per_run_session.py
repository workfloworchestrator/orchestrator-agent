"""Runs for different people must not share an MCP session, and so must not share a token.

The fake core is a real MCP server whose one tool returns the ``Authorization`` header it was called with, so each
run can be checked to have run as its own person. An MCP session's requests are made by background tasks that keep the
context they were started in: a session shared between runs would send the first run's token for every run that joins
it, which core would then judge (and audit) as the wrong person.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.test import TestModel

from orchestrator_agent.mcp_client import bind_outbound_token, build_core_toolset, run_copy

SLOW = 0.3  # long enough for runs started a moment apart to overlap


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _header(ctx: Context[Any, Any, Any], name: str) -> str:
    request = ctx.request_context.request
    return request.headers.get(name, "<none>") if request is not None else "<no request>"


@pytest.fixture(scope="module")
def core_url() -> Iterator[str]:
    """A real MCP server (as core's /mcp is) with tools that say which token and which session they were called with."""
    port = _free_port()
    fake_core = FastMCP("fake-core", stateless_http=False)

    @fake_core.tool()
    async def whoami(ctx: Context[Any, Any, Any]) -> str:
        await asyncio.sleep(SLOW)
        return _header(ctx, "authorization")

    @fake_core.tool()
    async def session(ctx: Context[Any, Any, Any]) -> str:
        return _header(ctx, "mcp-session-id")

    server = uvicorn.Server(
        uvicorn.Config(fake_core.streamable_http_app(), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def toolset(monkeypatch: pytest.MonkeyPatch, core_url: str):
    monkeypatch.setattr("orchestrator_agent.mcp_client.agent_settings.WFO_CORE_MCP_URL", core_url)
    monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", False)  # no service token
    return build_core_toolset()


def _tool_return(result: Any, tool_name: str) -> str:
    return next(
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    )


async def _run_as(agent: Agent[None, str], token: str | None, *, start_after: float = 0.0) -> str:
    """One run the way the adapters make it: the person's token bound, the model run."""
    await asyncio.sleep(start_after)
    with bind_outbound_token(token):
        return _tool_return(await agent.run("who am I"), "whoami")


def _agent(toolset: Any) -> Agent[None, str]:
    return Agent(TestModel(call_tools=["whoami"]), toolsets=[toolset])


def test_the_toolset_is_a_per_run_template():
    # Pinned because it rests on pydantic-ai's `auth`-as-a-function support: if an upgrade stops treating the
    # toolset as per-run, every run would share one session again.
    assert build_core_toolset()._resolves_per_run is True


@pytest.mark.parametrize(
    "second_starts_after",
    [
        pytest.param(0.0, id="simultaneous"),
        pytest.param(SLOW / 2, id="second-joins-while-the-first-is-mid-session"),
    ],
)
async def test_overlapping_runs_each_run_as_their_own_person(toolset, second_starts_after):
    agent = _agent(toolset)  # one agent and one toolset, shared, as the app builds them

    first, second = await asyncio.gather(
        _run_as(agent, "token-A"), _run_as(agent, "token-B", start_after=second_starts_after)
    )

    assert (first, second) == ("Bearer token-A", "Bearer token-B")


async def test_a_run_after_another_has_finished_uses_its_own_token(toolset):
    agent = _agent(toolset)

    first = await _run_as(agent, "token-A")
    second = await _run_as(agent, "token-B")

    assert (first, second) == ("Bearer token-A", "Bearer token-B")


async def test_a_run_with_no_bound_token_sends_no_authorization(toolset):
    assert await _run_as(_agent(toolset), None) == "<none>"


class _CallsCoreBeforeEachRequest(AbstractCapability[None]):
    """Calls core from a hook inside the run, as the form-fill skill does, and notes the session each call went over."""

    def __init__(self, toolset: Any, through: Callable[[Any, RunContext[None]], Any]) -> None:
        self.toolset, self.through = toolset, through
        self.sessions: list[str] = []

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.sessions.append(str(await self.through(self.toolset, ctx).direct_call_tool("session", {})))
        return request_context


@pytest.mark.parametrize(
    ("through", "shares_the_runs_session"),
    [
        pytest.param(run_copy, True, id="through-the-runs-copy"),
        pytest.param(lambda toolset, _ctx: toolset, False, id="through-the-shared-template"),
    ],
)
async def test_a_call_from_inside_a_run_goes_over_the_runs_own_session(toolset, through, shares_the_runs_session):
    calls_core = _CallsCoreBeforeEachRequest(toolset, through)
    agent = Agent(TestModel(call_tools=["session"]), toolsets=[toolset], capabilities=[calls_core])

    with bind_outbound_token("token-A"):
        runs_session = _tool_return(await agent.run("which session"), "session")  # the model's call: the run's own

    assert calls_core.sessions, "the hook never called core"
    assert (set(calls_core.sessions) == {runs_session}) is shares_the_runs_session


async def test_calls_outside_a_run_use_the_token_bound_at_the_call(toolset):
    async def call_as(token: str) -> str:
        with bind_outbound_token(token):
            result = await toolset.direct_call_tool("whoami", {})
        return str(result)

    # form fill and the chat endpoints call like this, while runs for other people are in flight
    assert await asyncio.gather(call_as("token-A"), call_as("token-B")) == ["Bearer token-A", "Bearer token-B"]


async def test_the_tool_list_is_readable_outside_a_run(toolset):
    # what verify_tool_contract does at startup, with no run and no person
    assert "whoami" in {tool.name for tool in await toolset.list_tools()}
