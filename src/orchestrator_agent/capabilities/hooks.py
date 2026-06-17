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

"""Agent capability assembly + the cross-cutting hooks.

``build_capabilities`` turns each plugin spec into a capability — a behavior-carrying
``PluginCapability`` when one is registered (see ``behavior/``), else an instructions-only
``Capability`` — and appends the hooks that are *not* plugin-owned because they are single invariants
across all tools:

- ``FilterPathGuard`` — a filtered/grouped call must follow ``discover_filter_paths``. It self-scopes
  by tool *schema* (any tool exposing ``filters``/``group_by``), so it is one cross-cutting instance,
  not something each filtering plugin owns a copy of.
- ``ProcessHistory`` — sliding-window history trimming, agent-wide.

Per-tool *result* behavior (mapping a tool's JSON into rich artifacts) is owned by the plugins
instead, in ``behavior/``.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.capabilities import AbstractCapability, ProcessHistory
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart
from pydantic_ai.tools import RunContext, ToolDefinition

from orchestrator_agent.capabilities.behavior import build_plugin_capability, owned_tool_names
from orchestrator_agent.capabilities.loader import load_plugin_specs
from orchestrator_agent.capabilities.spec import PluginSpec
from orchestrator_agent.tool_names import DISCOVER_FILTER_PATHS_TOOL, PATH_CONSUMING_PARAMS


def build_capabilities(specs: list[PluginSpec] | None = None) -> list[AbstractCapability[Any]]:
    """Build the agent's capability list: one capability per plugin, plus the cross-cutting hooks.

    The full MCP toolset is passed to the Agent; a plugin's instructions hide when it is deferred,
    and ``DeferredToolGate`` hides that plugin's *tools* until the model loads it (both revealed by
    one ``load_capability`` call). ``FilterPathGuard`` and history trimming are the other
    cross-cutting hooks. Tools owned by no plugin are always available (auto-appear).
    """
    resolved = specs if specs is not None else load_plugin_specs()
    plugin_caps = [build_plugin_capability(spec) for spec in resolved]
    return [
        *plugin_caps,
        DeferredToolGate(resolved),
        FilterPathGuard(),
        ProcessHistory[Any](processor=trim_history),
    ]


# --- Tool visibility for deferred plugins ---------------------------------------------------


class DeferredToolGate(AbstractCapability[Any]):
    """Hide a deferred plugin's owned tools until the model loads the plugin.

    pydantic-ai's ``defer_loading`` hides a capability's *instructions* but not tools provided as a
    toolset, so without this the model sees a deferred plugin's tool as callable while its guidance
    is hidden — a contradiction that makes it pick a fully-documented tool instead. This
    ``prepare_tools`` hook drops any tool owned by a deferred plugin that is not yet in
    ``ctx.loaded_capability_ids``; ``load_capability`` then reveals the tool and its instructions
    together. Always-on plugins and unowned tools are never touched.
    """

    def __init__(self, specs: list[PluginSpec]) -> None:
        # live tool name -> the plugin ids that own it, and whether each owner is deferred.
        self._owners: dict[str, list[tuple[str, bool]]] = {}
        for spec in specs:
            for name in owned_tool_names(spec):
                self._owners.setdefault(name, []).append((spec.id, spec.defer_loading))
        self._has_deferred = any(deferred for owners in self._owners.values() for _, deferred in owners)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def prepare_tools(self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        if not self._has_deferred:
            return tool_defs
        loaded = ctx.loaded_capability_ids
        # Hide a tool only when every owner is deferred AND not yet loaded; an always-on owner or any
        # loaded deferred owner keeps it visible (handles tools shared across plugins).
        return [
            td
            for td in tool_defs
            if (owners := self._owners.get(td.name)) is None
            or any(not deferred or pid in loaded for pid, deferred in owners)
        ]


# --- Deterministic search flow (discover filter paths before searching) --------------------


class FilterPathGuard(AbstractCapability[Any]):
    """Enforce the discover-before-filter ordering that makes search deterministic.

    Filter and ``group_by`` paths are database-specific — the model cannot guess them and
    must read them from ``discover_filter_paths`` first. Rather than trust prompt discipline
    (the old failure mode: invented paths, context bleed), this ``before_tool_execute`` hook
    makes the ordering a hard precondition: when the model calls *any* tool that supplies a
    path-consuming argument without having called the discovery tool earlier in the run, it
    raises ``ModelRetry`` to force the ``discover_filter_paths`` -> ``search``/``aggregate``
    sequence.

    Which arguments count as "path-consuming" is detected from each tool's *own* JSON schema
    (the names in ``PATH_CONSUMING_PARAMS``), so there is no hardcoded ``search``/``aggregate``
    tool-name list — any future tool that accepts ``filters``/``group_by`` is gated the same way.
    """

    def __init__(
        self,
        discovery_tool: str = DISCOVER_FILTER_PATHS_TOOL,
        path_params: tuple[str, ...] = PATH_CONSUMING_PARAMS,
    ) -> None:
        self.discovery_tool = discovery_tool
        self.path_params = path_params

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (behaviour, not data).

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        schema_props = (tool_def.parameters_json_schema or {}).get("properties", {})
        used = [p for p in self.path_params if p in schema_props and args.get(p)]
        if not used or self._discovery_called(ctx):
            return args
        raise ModelRetry(
            f"'{tool_def.name}' was called with {used}, but filter/group-by paths are "
            f"database-specific and must not be guessed. Call '{self.discovery_tool}' first to "
            f"obtain valid paths for those fields, then retry '{tool_def.name}'."
        )

    def _discovery_called(self, ctx: RunContext[Any]) -> bool:
        """True if the discovery tool was already called earlier in this run."""
        return any(
            isinstance(part, ToolCallPart) and part.tool_name == self.discovery_tool
            for message in ctx.messages
            for part in getattr(message, "parts", [])
        )


# --- History processing (sliding window) ----------------------------------------------------

_MAX_HISTORY_MESSAGES = 30


def trim_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Cap message history to a sliding window, preserving leading instructions/system parts.

    Keeps any leading request that carries instructions (so the system prompt stays in
    place) plus the most recent ``_MAX_HISTORY_MESSAGES`` messages.
    """
    if len(messages) <= _MAX_HISTORY_MESSAGES:
        return messages

    head: list[ModelMessage] = []
    rest = messages
    first = messages[0]
    if isinstance(first, ModelRequest) and getattr(first, "instructions", None):
        head = [first]
        rest = messages[1:]

    tail = rest[-_MAX_HISTORY_MESSAGES:]
    return head + tail


__all__ = [
    "DeferredToolGate",
    "FilterPathGuard",
    "build_capabilities",
    "trim_history",
]
