# Copyright 2019-2025 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared MCP client for orchestrator-core's domain tools.

The agent is a thin MCP client: every domain tool (search, aggregate, entity
lookup, export) is served by orchestrator-core's ``/mcp`` endpoint. We open a
single shared :class:`~pydantic_ai.mcp.MCPToolset` connection (``WFO_CORE_MCP_URL``)
and slice it per capability via ``.filtered(...)``.

Auth forwarding (best-effort)
-----------------------------
pydantic-ai has no first-class per-run header API for MCP toolsets, so the
incoming user's bearer token can't be passed through ``agent.run(...)``. We
bridge it with a :class:`contextvars.ContextVar` and a custom
:class:`httpx.Auth` that reads the var at request time. Adapters set the var
(via :func:`bind_outbound_token`) for the duration of a run; the ``MCPToolset``
is built once with this auth attached, so each HTTP request to core picks up
whatever token is bound on the current context.

If no per-request token is bound, we fall back to the service
client-credentials token from :mod:`orchestrator_agent.auth`.
"""

from __future__ import annotations

import contextvars
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from typing import Any

import httpx
import structlog
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset

from orchestrator_agent.auth import token_manager
from orchestrator_agent.settings import agent_settings

logger = structlog.get_logger(__name__)

# The bearer token for the *current* run, set by adapters from the incoming
# request. ``None`` means "fall back to the service token".
_outbound_token: contextvars.ContextVar[str | None] = contextvars.ContextVar("wfo_outbound_token", default=None)


@contextmanager
def bind_outbound_token(token: str | None) -> Generator[None, None, None]:
    """Bind a bearer token to forward to core's MCP for the current context.

    Adapters wrap a run in this so MCP HTTP requests carry the user's token::

        with bind_outbound_token(user_token):
            async with agent:
                await agent.run(...)

    Passing ``None`` is a no-op (the service token is used instead).
    """
    reset_token = _outbound_token.set(token)
    try:
        yield
    finally:
        _outbound_token.reset(reset_token)


class _ContextVarBearerAuth(httpx.Auth):
    """httpx auth that injects a bearer token resolved at request time.

    Prefers the per-run token bound via :func:`bind_outbound_token`; otherwise
    falls back to the cached service (client-credentials) token. The token is
    read inside :meth:`async_auth_flow` so a single long-lived ``MCPToolset``
    connection serves requests for different users without rebuilding.
    """

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token = _outbound_token.get()
        if token is None:
            # Service (client-credentials) token; returns None when outbound auth is disabled.
            token = await token_manager.get_token()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        # MCP HTTP transport is async-only; sync flow is never exercised but httpx requires it.
        raise RuntimeError("_ContextVarBearerAuth only supports async flows")


def build_core_toolset() -> MCPToolset[Any]:
    """Build the single shared MCPToolset pointed at orchestrator-core's ``/mcp``.

    Auth is attached via :class:`_ContextVarBearerAuth` so requests forward the
    bound per-run token (or the service token as a fallback). The session must
    be opened with ``async with agent:`` before tools can be listed/called.
    """
    logger.debug("Building core MCP toolset", url=agent_settings.WFO_CORE_MCP_URL)
    return MCPToolset(
        agent_settings.WFO_CORE_MCP_URL,
        id="wfo-core",
        auth=_ContextVarBearerAuth(),
    )


__all__ = [
    "AbstractToolset",
    "bind_outbound_token",
    "build_core_toolset",
]
