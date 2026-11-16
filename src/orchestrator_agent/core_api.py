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

"""Core's REST API, for the one thing the agent passes on rather than reads: an export's file."""

from __future__ import annotations

from urllib.parse import urljoin, urlsplit

import httpx

from orchestrator_agent.mcp_client import _ContextVarBearerAuth


class SourceError(Exception):
    """A URL that is not core's: the person's token is sent nowhere else."""


def api_base(mcp_url: str, configured: str | None = None) -> str:
    """Where a relative URL starts: as configured, else the origin of core's MCP endpoint."""
    if configured:
        return configured.rstrip("/")
    parts = urlsplit(mcp_url)
    return f"{parts.scheme}://{parts.netloc}"


class CoreApi:
    """GET from core's API, authenticated like the agent's MCP calls. One client; the app closes it (``aclose``)."""

    def __init__(self, base: str, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0) -> None:
        self.base = base.rstrip("/")
        self._client = httpx.AsyncClient(auth=_ContextVarBearerAuth(), transport=transport, timeout=timeout)

    async def get(self, url: str) -> httpx.Response:
        """Core's response for a URL, as it came. An error status raises."""
        target = urljoin(self.base + "/", url)
        if urlsplit(target).netloc != urlsplit(self.base).netloc:
            raise SourceError(f"{target} is not on core's host")
        response = await self._client.get(target)
        response.raise_for_status()
        return response

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["CoreApi", "SourceError", "api_base"]
