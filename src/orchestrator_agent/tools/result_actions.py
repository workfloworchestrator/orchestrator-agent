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

import json
from uuid import UUID

import httpx
import structlog
from orchestrator.search.core.types import EntityType
from orchestrator.search.query.results import ExportData
from pydantic_ai import RunContext
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolReturn
from pydantic_ai.toolsets import FunctionToolset

from orchestrator_agent.artifacts import DataArtifact, ExportArtifact
from orchestrator_agent.auth import token_manager
from orchestrator_agent.memory import ToolStep
from orchestrator_agent.settings import agent_settings
from orchestrator_agent.state import SearchState

logger = structlog.get_logger(__name__)

result_actions_toolset: FunctionToolset[StateDeps[SearchState]] = FunctionToolset(max_retries=2)


@result_actions_toolset.tool
async def fetch_entity_details(
    ctx: RunContext[StateDeps[SearchState]],
    entity_id: str,
    entity_type: EntityType,
) -> ToolReturn:
    """Fetch detailed information for a single entity by its ID.

    Args:
        ctx: Runtime context for agent (injected).
        entity_id: The UUID of the entity to fetch details for.
        entity_type: Type of entity.

    Returns:
        ToolReturn with entity JSON and ExportArtifact metadata.
    """
    logger.debug(
        "Fetching detailed entity data",
        entity_type=entity_type.value,
        entity_id=entity_id,
    )

    url = agent_settings.orchestrator_api_paths.entity_url(entity_type, entity_id)

    headers: dict[str, str] = {}
    token = await token_manager.get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=30)

        if response.status_code in (401, 403) and token_manager.auth_enabled:
            new_token = await token_manager.refresh_token()
            headers["Authorization"] = f"Bearer {new_token}"
            response = await client.get(url, headers=headers, timeout=30)

    if response.status_code == 404:
        raise ModelRetry(f"No {entity_type.value} found with ID {entity_id}.")
    response.raise_for_status()

    detailed = response.json()

    description = f"Fetched details for {entity_type.value} {entity_id}"

    ctx.deps.state.memory.record_tool_step(
        ToolStep(
            step_type="fetch_entity_details",
            description=description,
            context={"entity_id": entity_id},
        )
    )

    detailed_json = json.dumps(detailed, indent=2, default=str)

    artifact = DataArtifact(
        description=description,
        entity_id=entity_id,
        entity_type=entity_type.value,
    )

    return ToolReturn(return_value=detailed_json, metadata=artifact)


@result_actions_toolset.tool
async def prepare_export(
    ctx: RunContext[StateDeps[SearchState]],
    query_id: UUID | None = None,
) -> ToolReturn:
    """Prepares export URL for a search query.

    Args:
        ctx: Runtime context for agent (injected).
        query_id: Optional. Defaults to the most recent query. Only pass this to reference a specific historical query.

    Returns:
        ToolReturn with ExportData and ExportArtifact metadata.
    """
    query_id = query_id or ctx.deps.state.query_id
    if query_id is None:
        raise ModelRetry("No query available. Run a search first.")

    logger.debug(
        "Prepared query for export",
        query_id=str(query_id),
    )

    download_url = agent_settings.orchestrator_api_paths.export_url(str(query_id))

    export_data = ExportData(
        query_id=str(query_id),
        download_url=download_url,
        message="Export ready for download.",
    )

    description = f"Prepared export for query {query_id}"

    # Record tool step
    ctx.deps.state.memory.record_tool_step(
        ToolStep(
            step_type="prepare_export",
            description=description,
            context={"query_id": query_id},
        )
    )

    logger.debug("Export prepared", query_id=export_data.query_id)

    artifact = ExportArtifact(
        description=description,
        query_id=str(query_id),
        download_url=download_url,
    )

    return ToolReturn(return_value=export_data, metadata=artifact)
