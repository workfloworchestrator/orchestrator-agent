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

"""Shared stream consumption for protocol adapters (A2A, MCP).

Extracts artifact results and final LLM output from the agent event stream.
AG-UI has its own streaming logic (SSE events to frontend) and does not use this.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from pydantic_ai.messages import FunctionToolResultEvent, ToolReturnPart
from pydantic_ai.run import AgentRunResultEvent

from orchestrator_agent.artifacts import ToolArtifact

NO_RESULTS = "No results"


async def collect_stream_output(event_stream: AsyncIterator[Any]) -> str:
    """Consume an agent event stream and return a combined result string.

    Collects ToolArtifact-bearing tool results and the final LLM output.
    Returns artifact JSON (via model_response_str) if any, otherwise the
    LLM's final output, otherwise "No results".
    """
    artifact_results: list[ToolReturnPart] = []
    final_output = ""

    async for event in event_stream:
        if isinstance(event, FunctionToolResultEvent):
            result = event.result
            if isinstance(result, ToolReturnPart) and isinstance(result.metadata, ToolArtifact):
                artifact_results.append(result)
        if isinstance(event, AgentRunResultEvent):
            final_output = str(event.result.output)

    if artifact_results:
        return "\n\n".join(part.model_response_str() for part in artifact_results)

    return final_output or NO_RESULTS
