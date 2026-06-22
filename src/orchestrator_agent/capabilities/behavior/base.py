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

"""Base for behavior-carrying plugins.

A plugin's instructions come from its Markdown file (see ``loader``); a plugin that also maps its
own tools' results to rich artifacts is one ``PluginCapability`` carrying an **artifact builder** —
a ``(tool_name, payload) -> ToolArtifact | None`` function selected by the frontmatter ``artifact:``
declaration (see ``behavior/__init__.py``). One instance is one pydantic-ai capability identity: it
carries the Markdown instructions, the plugin's ``defer_loading`` flag, and the result hook — all
under a single ``id``, so when ``defer_loading`` is later flipped on, the instructions and the hook
scope together automatically.

To extend with a new artifact type, write a builder function (see ``behavior/artifacts.py``) and
register it in ``behavior.ARTIFACT_BUILDERS``. Cross-cutting hooks (the filter-path guard, history
trimming) are *not* plugins — they live in ``hooks.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.tools import RunContext, ToolDefinition

from orchestrator_agent.artifacts import ToolArtifact
from orchestrator_agent.capabilities.spec import PluginSpec
from orchestrator_agent.tool_names import TOOL_NAME_PLACEHOLDERS

logger = structlog.get_logger(__name__)

# A builder maps a tool's raw result payload to an artifact (or None when there is nothing to attach).
ArtifactBuilder = Callable[[str, Any], "ToolArtifact | None"]


def owned_tool_names(spec: PluginSpec) -> set[str]:
    """Resolve a plugin's declared ``tools:`` constants to the live MCP tool names it owns.

    Frontmatter declares ownership by constant (``SEARCH_TOOL``); the tools the model actually calls
    carry the live name (``search``). Resolving here ties the two together — used by the
    artifact-ownership filter and by ``DeferredToolGate`` (which hides a deferred plugin's tools).
    Raises on an unknown constant so a typo'd ``tools:`` fails loud at startup, not silently as
    no-ownership.
    """
    names: set[str] = set()
    for const in spec.tools:
        if const not in TOOL_NAME_PLACEHOLDERS:
            raise KeyError(
                f"Plugin '{spec.id}' declares unknown tool '{const}' in `tools:` — "
                f"not a constant in tool_names.py. Known: {sorted(TOOL_NAME_PLACEHOLDERS)}."
            )
        names.add(TOOL_NAME_PLACEHOLDERS[const])
    return names


class PluginCapability(AbstractCapability[Any]):
    """A plugin as one capability: Markdown instructions + an owned tool-result artifact builder.

    The ``after_tool_execute`` hook is fully handled here: it fires only for the plugin's *own* tools
    (``spec.tools``) — necessary because, while ``defer_loading`` is False, every capability's hooks
    run for every tool call — maps the result via the injected ``builder``, and attaches the artifact
    (plus a summarise-don't-restate notice) as ``ToolReturn`` metadata.
    """

    def __init__(self, spec: PluginSpec, builder: ArtifactBuilder) -> None:
        self.id = spec.id
        self.description = spec.description
        self.defer_loading = spec.defer_loading
        self._instructions = spec.instructions
        self._tools = owned_tool_names(spec)  # live names this plugin owns
        self._builder = builder

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # behavior, not declarative spec-data

    def get_instructions(self) -> str:
        return self._instructions

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        result: Any,
    ) -> Any:
        if tool_def.name not in self._tools:
            return result
        try:
            artifact = self._builder(tool_def.name, _result_payload(result))
        except Exception:  # noqa: BLE001 - artifact mapping must never break tool execution
            logger.exception("Artifact mapping failed; returning raw result", tool=tool_def.name)
            return result
        if artifact is None:
            return result
        return _attach_artifact(result, artifact)


def _result_payload(result: Any) -> Any:
    """Unwrap a ToolReturn to its underlying value; pass other results through."""
    return result.return_value if isinstance(result, ToolReturn) else result


def _attach_artifact(result: Any, artifact: ToolArtifact) -> Any:
    """Attach the artifact as ToolReturn metadata, plus a one-line 'summarise, don't restate' notice.

    When a chart/table is injected by the adapter the agent never sees it, so we tell it (only on the
    calls that actually inject) to summarise rather than list the rows — otherwise the prose duplicates.
    """
    notice = _injection_notice(artifact)
    if isinstance(result, ToolReturn):
        result.metadata = artifact
        if notice and result.content is None:
            result.content = notice
        return result
    return ToolReturn(return_value=result, metadata=artifact, content=notice)


def _injection_notice(artifact: ToolArtifact) -> str | None:
    """A note for the agent when this result carries a chart/table the adapter will append, else None."""
    block = getattr(artifact, "rendered_block", None)
    if block is None:
        return None
    kind = "chart" if block.type == "mermaid" else "table"
    return (
        f"A {kind} for this result is shown to the user automatically below your reply. "
        "Summarise in one sentence; do not reproduce it or restate its rows/numbers."
    )


__all__ = ["ArtifactBuilder", "PluginCapability", "owned_tool_names"]
