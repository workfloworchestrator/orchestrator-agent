# Copyright 2019-2026 SURF, GÉANT.
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

"""Core's GraphQL API, as the person asking: where the frontend's components get most of their options.

Customers have no MCP tool in core; the frontend reads them over GraphQL, and so does a widget. The request
carries the same token as the agent's MCP calls, so a widget sees what the person may see.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from orchestrator_agent.mcp_client import _ContextVarBearerAuth


class GraphQLError(Exception):
    """Core answered the query with errors, or without data."""


def graphql_url(mcp_url: str, configured: str | None = None) -> str:
    """Core's GraphQL endpoint: as configured, else beside its MCP endpoint (``…/mcp`` -> ``…/api/graphql``)."""
    if configured:
        return configured
    base = mcp_url.rstrip("/")
    return (base.removesuffix("/mcp") if base.endswith("/mcp") else base) + "/api/graphql"


class CoreGraphQL:
    """A ``GraphQL`` (``widgets.base``) over HTTP, authenticated like the agent's MCP calls.

    One client serves every query, so they share its connections; the app closes it (``aclose``) when it stops.
    """

    def __init__(self, url: str, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0) -> None:
        self.url = url
        self._client = httpx.AsyncClient(auth=_ContextVarBearerAuth(), transport=transport, timeout=timeout)

    async def __call__(self, query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]:
        response = await self._client.post(self.url, json={"query": query, "variables": dict(variables or {})})
        response.raise_for_status()
        body = response.json()
        if body.get("errors"):
            raise GraphQLError("; ".join(str(error.get("message", error)) for error in body["errors"]))
        if not isinstance(body.get("data"), dict):
            raise GraphQLError(f"no data in the answer of {self.url}")
        return body["data"]

    async def aclose(self) -> None:
        """Close the client and its connections."""
        await self._client.aclose()


__all__ = ["CoreGraphQL", "GraphQLError", "graphql_url"]
