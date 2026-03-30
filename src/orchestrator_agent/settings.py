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

from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.search.core.types import EntityType
from pydantic import Field
from pydantic_settings import BaseSettings

if TYPE_CHECKING:
    from pydantic_ai.models import Model


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
    BASE_URL: str = Field(default="http://localhost:8080", description="Public URL of this agent service")
    ORCHESTRATOR_API_URL: str = Field(default="http://localhost:8080", description="URL of the orchestrator-core API")
    AGENT_MODEL: str = Field(default="openai:gpt-4o", description="LLM model for the agent")
    AGENT_API_BASE: str | None = Field(
        default=None, description="Custom base URL for the LLM provider (OpenAI-compatible or Azure endpoint)"
    )
    AGENT_API_KEY: str | None = Field(default=None, description="API key for the LLM provider")
    AGENT_API_VERSION: str | None = Field(
        default=None, description="API version for Azure OpenAI (e.g. 2024-12-01-preview)"
    )
    AGENT_DEBUG: bool = Field(default=False, description="Enable debug logging for agent execution")
    orchestrator_api_paths: OrchestratorAPIPaths = Field(default_factory=OrchestratorAPIPaths)

    def create_model(self) -> str | Model:
        """Create a pydantic-ai model from settings.

        Returns the plain model string when no custom endpoint/key/version is configured.
        When AGENT_API_BASE, AGENT_API_KEY, or AGENT_API_VERSION is set, constructs an
        OpenAIChatModel with the appropriate provider (AzureProvider for ``azure:`` prefixed
        models, OpenAIProvider otherwise).
        """
        if self.AGENT_API_BASE is None and self.AGENT_API_KEY is None and self.AGENT_API_VERSION is None:
            return self.AGENT_MODEL

        from pydantic_ai.models.openai import OpenAIChatModel

        model_name = self.AGENT_MODEL
        provider_prefix = None
        if ":" in model_name:
            provider_prefix, model_name = model_name.split(":", 1)

        if provider_prefix == "azure" or self.AGENT_API_VERSION is not None:
            from pydantic_ai.providers.azure import AzureProvider

            provider = AzureProvider(
                azure_endpoint=self.AGENT_API_BASE,
                api_version=self.AGENT_API_VERSION,
                api_key=self.AGENT_API_KEY,
            )
        else:
            from pydantic_ai.providers.openai import OpenAIProvider

            provider = OpenAIProvider(base_url=self.AGENT_API_BASE, api_key=self.AGENT_API_KEY)  # type: ignore[assignment]

        return OpenAIChatModel(model_name, provider=provider)


agent_settings = AgentSettings()
