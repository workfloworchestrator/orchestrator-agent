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

"""MCP client for orchestrator-core's domain tools.

The agent is a thin MCP client: every domain tool (search, aggregate, entity lookup, export) is served by
orchestrator-core's ``/mcp`` endpoint (``WFO_CORE_MCP_URL``), through the toolset :func:`build_core_toolset` builds.

Auth forwarding
---------------
Core must see each call as the person who asked. Adapters bind the incoming request's bearer token with
:func:`bind_outbound_token` for the duration of a run. The toolset's ``auth`` is a function of the run, so pydantic-ai
gives each run a copy of the toolset with an MCP session of its own, authenticated with the token bound when the run
starts: runs for different people never share a session.

The token is fixed per session rather than read per request because an MCP session's requests are sent by background
tasks, which keep the context they were started in. A session shared between runs would send the first run's token
for all of them.

Code inside a run that calls core itself (the form-fill skill) goes through the run's copy (:func:`run_copy`), so a
run keeps to one session. A call outside a run (the chats endpoints, the startup contract check) opens a short-lived
session as the person whose token is bound.

With no token bound, the service (client-credentials) token from :mod:`orchestrator_agent.auth` is used.
"""

from __future__ import annotations

import contextvars
import json
import re
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, cast

import httpx
import structlog
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.mcp import CallToolFunc, MCPToolset, ToolResult
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from orchestrator_agent.auth import token_manager
from orchestrator_agent.settings import agent_settings
from orchestrator_agent.tool_names import ALL_TOOL_NAMES

logger = structlog.get_logger(__name__)

# The bearer token for the *current* run, set by adapters from the incoming
# request. ``None`` means "fall back to the service token".
_outbound_token: contextvars.ContextVar[str | None] = contextvars.ContextVar("wfo_outbound_token", default=None)


@contextmanager
def bind_outbound_token(token: str | None) -> Generator[None, None, None]:
    """Bind a bearer token to forward to core's MCP for the current context.

    Adapters wrap a run in this so MCP HTTP requests carry the user's token::

        with bind_outbound_token(user_token):
            await agent.run(...)

    The token is picked up when the run starts, and that run's MCP session keeps it: each run opens a session of its
    own (see :func:`build_core_toolset`), so runs for different people never share one.

    Passing ``None`` is a no-op (the service token is used instead).
    """
    reset_token = _outbound_token.set(token)
    try:
        yield
    finally:
        _outbound_token.reset(reset_token)


class _ContextVarBearerAuth(httpx.Auth):
    """httpx auth that injects a bearer token.

    With no ``token`` it resolves the token at request time: the one bound via :func:`bind_outbound_token`, else the
    cached service (client-credentials) token. That is right for a client that makes its requests from the caller's own
    task. It is wrong for an MCP session, whose requests are made by background tasks that keep the context they were
    started in, so a session shared between runs would send the first run's token for all of them. An MCP session is
    therefore given the person's ``token`` up front, and keeps it for the run.

    On a 401/403 against the service token, the cache is stale (client-credentials
    tokens expire) -- refresh once and retry, mirroring
    ``oauth2_lib.async_api_client.AsyncAuthMixin.request``. A person's token is
    forwarded as-is and never refreshed: it's the caller's credential, not ours.
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        per_run_token = self._token if self._token is not None else _outbound_token.get()
        token = per_run_token if per_run_token is not None else await token_manager.get_token()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"

        response = yield request

        if per_run_token is None and response.status_code in (401, 403) and token_manager.auth_enabled:
            token = await token_manager.refresh_token()
            request.headers["Authorization"] = f"Bearer {token}"
            yield request

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        # MCP HTTP transport is async-only; sync flow is never exercised but httpx requires it.
        raise RuntimeError("_ContextVarBearerAuth only supports async flows")


def _accepts_container(schema: dict[str, Any]) -> bool:
    """Whether a JSON-schema property can hold an object or an array.

    A ``$ref`` counts: core's structured arguments, such as ``filters``, are referenced models.
    """
    return any(
        member.get("type") in ("object", "array") or "$ref" in member for member in schema.get("anyOf") or [schema]
    )


def _decoded(value: Any) -> Any:
    """The object or array a string holds as JSON; any other value, and any other string, unchanged."""
    if not (isinstance(value, str) and value.lstrip().startswith(("{", "["))):
        return value
    try:
        decoded = json.loads(value)
    except ValueError:
        return value
    return decoded if isinstance(decoded, dict | list) else value


def decode_stringified_json(args: dict[str, Any], properties: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments with every JSON string that sits in an object or array parameter decoded.

    Some models (observed with Qwen3.8) send a nested argument such as ``filters`` as a JSON-encoded string.
    Core rejects that with a 422 whose message does not say the value was a string, so the model repeats it until
    the run fails. Parameters the schema declares as plain strings are left alone, so a free-text value that
    happens to look like JSON is not rewritten.

    Args:
        args: The arguments as the model produced them.
        properties: The tool's JSON-schema ``properties``; an argument it does not list is left alone.
    """
    return {
        key: _decoded(value) if _accepts_container(properties.get(key, {})) else value for key, value in args.items()
    }


_VALIDATION_REJECTION = re.compile(r"HTTP error 422: .*? - (?P<body>\{.*\})\s*$", re.DOTALL)
# Pydantic error types that say "expected an object / an array"
_CONTAINER_EXPECTED = ("model_attributes_type", "dict_type", "list_type")


def _describe_rejection(item: dict[str, Any], args: dict[str, Any]) -> str:
    location = [str(part) for part in item.get("loc", []) if part != "body"]
    line = f"- {'.'.join(location) or 'request'}: {item.get('msg', 'invalid value')}"
    if item.get("type") in _CONTAINER_EXPECTED and location and isinstance(args.get(location[0]), str):
        line += f" (you sent {location[0]} as a string; pass it as a JSON object or array, not as quoted JSON text)"
    return line


def explain_core_rejection(error: str, args: dict[str, Any]) -> str:
    """Core's 422 validation rejection as per-argument feedback the model can act on.

    The raw text is a one-line HTTP error that embeds the rejected input and never says what was wrong with its
    shape, so a model tends to resend the same call. Anything that is not a validation rejection of the known
    shape is returned unchanged.

    Args:
        error: The message of the tool error, as the toolset raised it.
        args: The arguments the call was made with, to tell which ones were sent as strings.
    """
    found = _VALIDATION_REJECTION.search(error)
    try:
        details = json.loads(found["body"])["detail"] if found else None
    except (ValueError, KeyError, TypeError):
        return error
    if not isinstance(details, list) or not details:
        return error
    lines = [_describe_rejection(item, args) for item in details if isinstance(item, dict)]
    return (
        "orchestrator-core rejected the arguments, so nothing was executed:\n"
        + "\n".join(lines)
        + "\nCorrect these arguments and call the tool again."
    )


# Schema keywords that nest one argument's schema in another: a union member or a list's items
_NESTED_SCHEMA_KEYWORDS = ("anyOf", "oneOf", "allOf", "items")


def _inline_ref(node: Any, defs: dict[str, Any], hops: int) -> Any:
    """``node`` with each ``$ref`` replaced by the definition it names, following up to ``hops`` nested keywords."""
    if isinstance(node, list):
        return [_inline_ref(member, defs, hops) for member in node]
    if not isinstance(node, dict) or hops < 0:
        return node
    target = defs.get(str(node.get("$ref", "")).removeprefix("#/$defs/"))
    resolved = {**target, **{key: value for key, value in node.items() if key != "$ref"}} if target else node
    return {
        key: _inline_ref(value, defs, hops - 1) if key in _NESTED_SCHEMA_KEYWORDS else value
        for key, value in resolved.items()
    }


def inline_argument_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """A tool's input schema with the ``$ref`` of each argument replaced by the definition it names.

    Qwen's tool-call parser does not resolve ``$ref``. For an argument whose schema is a reference it cannot tell the
    value is an object, so it passes the model's JSON through as a string: core's ``filters`` arrived as a string in
    8 of 8 samples, and as an object in 8 of 8 once its reference was inlined. Only the references at the argument
    level are replaced. ``FilterTree`` refers to itself, so inlining it completely cannot terminate, and the references
    inside it need no resolving because the object is parsed as one JSON value.
    """
    defs = schema.get("$defs", {})
    properties = {name: _inline_ref(prop, defs, 4) for name, prop in schema.get("properties", {}).items()}
    return {**schema, "properties": properties} if properties else schema


class _CoreToolset(MCPToolset[Any]):
    """The core MCP toolset, describing each tool's arguments the way open-weight models' tool parsers can read them.

    Its ``auth`` is a function of the run, so this object is a template: every agent run enters a copy of its own
    (``for_run``), with its own MCP session, and the shared template itself never connects. Code inside a run reaches
    the run's copy with :func:`run_copy`. A call outside a run (the chats endpoints, the startup contract check) gets
    a short-lived session of the person whose token is bound.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The template a copy was made from: ``for_run`` copies this attribute, which is how ``run_copy`` tells
        # this template's copy from any other toolset in a run.
        self._template = self
        # The arguments of each tool, by tool name, as ``get_tools`` last described them. A copy made for a run
        # shares this dict, which is how the call hook finds a tool's schema without a connection of its own.
        self._argument_schemas: dict[str, dict[str, Any]] = {}

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        tools = await super().get_tools(ctx)
        described = {
            name: replace(
                tool,
                tool_def=replace(
                    tool.tool_def, parameters_json_schema=inline_argument_refs(tool.tool_def.parameters_json_schema)
                ),
            )
            for name, tool in tools.items()
        }
        self._argument_schemas.update(
            {name: tool.tool_def.parameters_json_schema.get("properties", {}) for name, tool in described.items()}
        )
        return described

    async def _session_for_bound_person(self) -> "_CoreToolset":
        """A copy of this template with a session of its own, using the token bound right now.

        ``for_run`` hands the run to the ``auth`` function, which reads the bound token and ignores the run.
        """
        return await self.for_run(cast("RunContext[Any]", None))

    async def direct_call_tool(
        self, name: str, args: dict[str, Any], *, metadata: dict[str, Any] | None = None, use_task: bool = False
    ) -> Any:
        if not self._resolves_per_run:  # a run's own copy, or a toolset with fixed auth
            return await super().direct_call_tool(name, args, metadata=metadata, use_task=use_task)
        session = await self._session_for_bound_person()
        return await session.direct_call_tool(name, args, metadata=metadata, use_task=use_task)

    async def list_tools(self) -> list[Any]:
        if not self._resolves_per_run:
            return await super().list_tools()
        session = await self._session_for_bound_person()
        async with session:
            return await session.list_tools()


def _auth_for_run(ctx: RunContext[Any]) -> httpx.Auth:
    """The auth of one run's MCP session: the person's token as bound when the run starts.

    It is read here, in the run's own context, and handed to the session: the session's requests are made by
    background tasks, which would otherwise keep whichever token was bound when the *first* run opened the session.
    """
    return _ContextVarBearerAuth(_outbound_token.get())


def run_copy(toolset: MCPToolset[Any], ctx: RunContext[Any]) -> MCPToolset[Any]:
    """The copy of the core ``toolset`` that the run in ``ctx`` connected with; ``toolset`` when the run has none.

    For code inside a run that calls core itself, such as the form-fill skill: a call through the copy goes over the
    run's own MCP session, where one through the template would open a session for every call.
    """
    leaves: list[AbstractToolset[Any]] = []
    if ctx.tool_manager is not None:
        ctx.tool_manager.toolset.apply(leaves.append)
    return next((leaf for leaf in leaves if isinstance(leaf, _CoreToolset) and leaf._template is toolset), toolset)


def build_core_toolset() -> MCPToolset[Any]:
    """Build the MCPToolset template pointed at orchestrator-core's ``/mcp``.

    Every agent run enters a copy of it with a session of its own, authenticated as the person the run belongs to
    (:func:`bind_outbound_token`), or with the service token when none is bound. Runs for different people therefore
    never share a session, and so never share a token. See :class:`_CoreToolset` for calls outside a run.
    The tool schemas are sent with each argument's ``$ref`` inlined (see :func:`inline_argument_refs`); a model that
    still sends stringified JSON in an object argument has it decoded before the call (see
    :func:`decode_stringified_json`), and core's validation rejections are reworded for the model (see
    :func:`explain_core_rejection`).
    """
    logger.debug("Building core MCP toolset", url=agent_settings.WFO_CORE_MCP_URL)

    async def call_with_decoded_arguments(
        ctx: RunContext[Any], call_tool: CallToolFunc, name: str, tool_args: dict[str, Any]
    ) -> ToolResult:
        # Filled by get_tools, which the run calls to present the tools before the model can call one
        decoded = decode_stringified_json(tool_args, toolset._argument_schemas.get(name, {}))
        try:
            return await call_tool(name, decoded)
        except ModelRetry as exc:
            raise ModelRetry(explain_core_rejection(str(exc), decoded)) from exc

    toolset: _CoreToolset = _CoreToolset(
        agent_settings.WFO_CORE_MCP_URL,
        id="wfo-core",
        auth=_auth_for_run,
        process_tool_call=call_with_decoded_arguments,
    )
    return toolset


async def verify_tool_contract(toolset: MCPToolset[Any]) -> None:
    """Assert every tool name the agent depends on exists on the live MCP server.

    The names in ``tool_names.ALL_TOOL_NAMES`` are referenced two ways that pydantic-ai cannot
    check for us: literally in the plugin prompts, and as match keys in artifact mapping (the
    builders in ``capabilities/behavior/artifacts.py``). pydantic-ai only validates tool calls the
    *model* makes at runtime — it is name-agnostic about our code's expectations — so a rename in
    orchestrator-core would otherwise drift silently. This opens one session, lists the server's
    tools, and raises on any missing name.

    Only a *connection* failure (core momentarily unreachable) is logged and skipped — that is an
    operational condition, not contract drift. Any other failure (auth, HTTP status, malformed
    response) is a real misconfiguration and is raised so startup fails loud rather than serving a
    broken agent.
    """
    try:
        async with toolset:
            live = {tool.name for tool in await toolset.list_tools()}
    except (httpx.TransportError, ConnectionError, OSError, TimeoutError) as exc:
        logger.warning(
            "Could not reach the MCP server to verify the tool contract; skipping",
            url=agent_settings.WFO_CORE_MCP_URL,
            error=str(exc),
        )
        return

    missing = sorted(name for name in ALL_TOOL_NAMES if name not in live)
    if missing:
        raise RuntimeError(
            f"MCP server at {agent_settings.WFO_CORE_MCP_URL} is missing tools the agent depends on: "
            f"{missing}. Available tools: {sorted(live)}. The orchestrator-core tool contract "
            f"(tool_names.py) has drifted — fix the names before serving."
        )
    logger.debug("MCP tool contract verified", tools=sorted(live))


__all__ = [
    "AbstractToolset",
    "bind_outbound_token",
    "build_core_toolset",
    "run_copy",
    "verify_tool_contract",
]
