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

"""Registry binding each ``ArtifactType`` to its shared artifact builder.

A plugin declares ``artifact: query`` (etc.) to map its tool results to that artifact type; the
builders are shared and reusable, so a new plugin reusing a standard artifact needs *no code* — just
the declaration (built-in or external). A plugin with no ``artifact:`` is instructions-only. To add a
genuinely new artifact type, add a value to ``ArtifactType`` and a builder here (a per-plugin
custom-render escape hatch is a future layer, not implemented yet).
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.capabilities import AbstractCapability, Capability

from orchestrator_agent.capabilities.behavior.artifacts import data_artifact, export_artifact, query_artifact
from orchestrator_agent.capabilities.behavior.base import ArtifactBuilder, PluginCapability, owned_tool_names
from orchestrator_agent.capabilities.spec import ArtifactType, PluginSpec

# ArtifactType -> the builder that maps a tool result to that artifact.
ARTIFACT_BUILDERS: dict[ArtifactType, ArtifactBuilder] = {
    ArtifactType.QUERY: query_artifact,
    ArtifactType.DATA: data_artifact,
    ArtifactType.EXPORT: export_artifact,
}
# Every declared artifact type must have a builder — caught at import, not at runtime.
if set(ARTIFACT_BUILDERS) != set(ArtifactType):
    raise RuntimeError(
        f"ARTIFACT_BUILDERS must cover every ArtifactType; missing: {set(ArtifactType) - set(ARTIFACT_BUILDERS)}"
    )


def build_plugin_capability(spec: PluginSpec) -> AbstractCapability[Any]:
    """Project a plugin spec to a capability, selecting its artifact builder by ``artifact:``.

    With ``artifact: <type>`` -> a ``PluginCapability`` carrying that type's builder; without it -> an
    instructions-only ``Capability``. Either way the capability carries ``defer_loading`` (so its
    *instructions* hide until loaded); its *tools* are hidden by the cross-cutting ``DeferredToolGate``.
    The ``artifact`` value is already validated against ``ArtifactType`` by the spec.
    """
    if spec.artifact is not None:
        return PluginCapability(spec, ARTIFACT_BUILDERS[spec.artifact])
    return Capability[Any](
        id=spec.id,
        description=spec.description,
        instructions=spec.instructions,
        defer_loading=spec.defer_loading,
    )


__all__ = [
    "ARTIFACT_BUILDERS",
    "PluginCapability",
    "build_plugin_capability",
    "owned_tool_names",
]
