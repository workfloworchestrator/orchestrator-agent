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

from orchestrator.search.core.types import EntityType
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class OrchestratorAPIPaths(BaseSettings):
    """Path templates for orchestrator-core API endpoints."""

    SUBSCRIPTION: str = "/api/subscriptions/domain-model/{entity_id}"
    PRODUCT: str = "/api/products/{entity_id}"
    WORKFLOW: str = "/api/workflows/{entity_id}"
    PROCESS: str = "/api/processes/{entity_id}"
    EXPORT: str = "/api/search/queries/{query_id}/export"

    _entity_map: dict[EntityType, str] = {
        EntityType.SUBSCRIPTION: "SUBSCRIPTION",
        EntityType.PRODUCT: "PRODUCT",
        EntityType.WORKFLOW: "WORKFLOW",
        EntityType.PROCESS: "PROCESS",
    }

    def entity_url(self, entity_type: EntityType, entity_id: str) -> str:
        """Resolve full URL for an entity type."""
        attr = self._entity_map[entity_type]
        path = getattr(self, attr).format(entity_id=entity_id)
        return f"{agent_settings.ORCHESTRATOR_API_URL}{path}"

    def export_url(self, query_id: str) -> str:
        """Resolve full URL for a query export."""
        return f"{agent_settings.ORCHESTRATOR_API_URL}{self.EXPORT.format(query_id=query_id)}"


class AgentSettings(BaseSettings):
    """Settings for the standalone orchestrator agent."""

    DATABASE_URI: str = Field(default="", description="PostgreSQL connection URI for WFO database")
    BASE_URL: str = Field(default="http://localhost:8000", description="Public URL of this agent service")
    ORCHESTRATOR_API_URL: str = Field(default="http://localhost:8080", description="URL of the orchestrator-core API")
    AGENT_MODEL: str = Field(default="openai:gpt-4o", description="LLM model for the agent")
    AGENT_DEBUG: bool = Field(default=False, description="Enable debug logging for agent execution")
    orchestrator_api_paths: OrchestratorAPIPaths = Field(default_factory=OrchestratorAPIPaths)

    OAUTH2_ACTIVE: bool = Field(
        default=False, description="Toggle to enable authenticated requests to the orchestrator"
    )
    OAUTH2_TOKEN_ENDPOINT: str | None = Field(default=None, description="OAuth2 token endpoint URL")
    OAUTH2_CLIENT_ID: str | None = Field(default=None, description="OAuth2 client ID")
    OAUTH2_CLIENT_SECRET: str | None = Field(default=None, description="OAuth2 client secret")

    @model_validator(mode="after")
    def _validate_oauth2_settings(self) -> "AgentSettings":
        if self.OAUTH2_ACTIVE:
            missing = [
                name
                for name in ("OAUTH2_TOKEN_ENDPOINT", "OAUTH2_CLIENT_ID", "OAUTH2_CLIENT_SECRET")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"OAUTH2_ACTIVE is True but the following settings are not set: {', '.join(missing)}")
        return self


agent_settings = AgentSettings()
