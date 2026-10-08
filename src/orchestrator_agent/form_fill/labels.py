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

"""The labels the WFO frontend shows for form fields: core's form translations (``GET /api/translations/<lang>``).

A page schema titles a field after its Python name (``Customer Id``) or its type (``PortEnum``); the frontend
shows what the deployment's translations say (``Customer``, ``Port Mode``), with a help text under it
(``<field>_info``). Core serves them without authentication; they are read once per agent and cached.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import structlog

logger = structlog.get_logger(__name__)

LANGUAGE = "en-GB"
_TTL = 3600.0  # seconds the labels are kept before they are read again


@dataclass(frozen=True)
class FieldLabels:
    """What the frontend calls a field and the help it shows with it, by field name."""

    fields: Mapping[str, str] = field(default_factory=dict)

    def title(self, name: str) -> str | None:
        return self.fields.get(name) or None

    def info(self, name: str) -> str | None:
        return self.fields.get(f"{name}_info") or None


NO_LABELS = FieldLabels()


class LabelSource(Protocol):
    """Where a form's field labels come from."""

    async def labels(self) -> FieldLabels: ...


def _form_fields(body: object) -> dict[str, str]:
    """The ``forms.fields`` table of a translations answer; empty when it has none."""
    match body:
        case {"forms": {"fields": dict() as fields}}:
            return {str(key): str(value) for key, value in fields.items() if isinstance(value, str)}
        case _:
            return {}


def core_api_url(mcp_url: str) -> str:
    """Core's REST API beside its MCP endpoint (``…/mcp`` -> ``…/api``)."""
    base = mcp_url.rstrip("/")
    return (base.removesuffix("/mcp") if base.endswith("/mcp") else base) + "/api"


class CoreLabels:
    """Core's form translations, read once and kept for an hour; no labels when core cannot give them."""

    def __init__(self, api_url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.url = f"{api_url.rstrip('/')}/translations/{LANGUAGE}"
        self.transport = transport
        self._labels: FieldLabels | None = None
        self._at = 0.0

    async def labels(self) -> FieldLabels:
        if self._labels is not None and time.monotonic() - self._at < _TTL:
            return self._labels
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=10.0) as client:
                response = await client.get(self.url)
            response.raise_for_status()
            fields = _form_fields(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Form labels unavailable; the schema's titles are used", url=self.url, error=str(exc))
            fields = {}
        self._labels, self._at = FieldLabels(fields), time.monotonic()
        return self._labels


__all__ = ["LANGUAGE", "NO_LABELS", "CoreLabels", "FieldLabels", "LabelSource", "core_api_url"]
