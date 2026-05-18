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

"""MCP client for orchestrator-core's workflow form primitives.

The agent calls orchestrator-core's MCP server (mounted at ``/mcp``) for the
three tools needed by the workflow form-fill skill:

- ``list_workflows``
- ``get_workflow_form``
- ``create_workflow``

A new streamable-HTTP session is opened per call. This is simpler and
sufficient for the demo; revisit if call volume grows.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import structlog
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from orchestrator_agent.settings import agent_settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def _session(auth_token: str | None = None) -> AsyncIterator[ClientSession]:
    headers: dict[str, str] | None = None
    if auth_token:
        headers = {"Authorization": f"Bearer {auth_token}"}
    async with streamablehttp_client(agent_settings.WFO_CORE_MCP_URL, headers=headers) as (
        read,
        write,
        _get_session_id,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _unwrap_tool_result(result: Any) -> Any:
    """Return the structured payload of an MCP tool call.

    FastMCP serialises Pydantic responses into ``structuredContent`` when
    available; otherwise the JSON body is in ``content[0].text``.
    """
    if getattr(result, "isError", False):
        err_text = ""
        if result.content:
            err_text = getattr(result.content[0], "text", "") or ""
        raise RuntimeError(f"MCP tool returned error: {err_text or result}")

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured

    if result.content:
        first = result.content[0]
        text: str | None = getattr(first, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return None


async def list_workflows(
    target: str | None = None,
    is_task: bool | None = None,
    auth_token: str | None = None,
) -> list[dict[str, Any]]:
    """Call orchestrator-core's ``list_workflows`` MCP tool."""
    # Always send the full body — FastMCP forwards ``arguments`` as the request
    # body and an empty dict produces ``null``, which fails Pydantic validation.
    args: dict[str, Any] = {"target": target, "is_task": is_task}

    async with _session(auth_token) as session:
        result = await session.call_tool("list_workflows", arguments=args)
    payload = _unwrap_tool_result(result)
    # FastMCP wraps list returns in {"result": [...]} when from_fastapi infers a non-object root.
    if isinstance(payload, dict) and "result" in payload and isinstance(payload["result"], list):
        return payload["result"]
    if isinstance(payload, list):
        return payload
    return []


async def get_workflow_form(
    workflow_key: str,
    page_inputs: list[dict[str, Any]] | None = None,
    auth_token: str | None = None,
) -> dict[str, Any]:
    """Call ``get_workflow_form`` and return the ``WorkflowFormPage`` dict.

    Returns a dict with keys ``page: int``, ``complete: bool``, ``schema: dict | None``.
    """
    args: dict[str, Any] = {"workflow_key": workflow_key, "page_inputs": page_inputs}

    async with _session(auth_token) as session:
        result = await session.call_tool("get_workflow_form", arguments=args)
    payload = _unwrap_tool_result(result)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected get_workflow_form payload: {payload!r}")
    return payload


async def create_workflow(
    workflow_key: str,
    page_inputs: list[dict[str, Any]],
    auth_token: str | None = None,
) -> dict[str, Any]:
    """Call ``create_workflow``. Returns ``{"id": "<process_uuid>"}``."""
    args = {"workflow_key": workflow_key, "json_data": page_inputs}

    async with _session(auth_token) as session:
        result = await session.call_tool("create_workflow", arguments=args)
    payload = _unwrap_tool_result(result)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected create_workflow payload: {payload!r}")
    return payload
