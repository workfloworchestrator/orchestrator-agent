"""Eval-harness settings: pydantic-settings with the standard precedence
(real environment variables > evals/.env > the defaults below).

The agent's own ``AgentSettings`` reads the process environment at import, so
``apply_to_environment()`` must run before ``orchestrator_agent`` is imported —
the harness resolves the values, the production settings class consumes them.
The defaults describe the bundled eval environment (docker-compose.yml): the
core's MCP address is decided by that file, and its core runs authless.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

EVALS_DIR = Path(__file__).parent


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=EVALS_DIR / ".env", extra="ignore", frozen=True)

    # The orchestrator-core MCP server to evaluate against (docker-compose.yml).
    wfo_core_mcp_url: str = "http://localhost:8086/mcp"
    # The bundled core runs authless; outbound OAuth2 would demand a token URL.
    oauth2_active: bool = False

    # Pin the model: scores are only comparable across runs on the same model.
    agent_model: str = "openai:gpt-4o"
    agent_api_key: str | None = None

    dataset_file: Path = EVALS_DIR / "wfo_search_dataset.yaml"

    def apply_to_environment(self) -> None:
        """Publish the resolved values for the agent's ``AgentSettings`` to read."""
        os.environ["WFO_CORE_MCP_URL"] = self.wfo_core_mcp_url
        os.environ["OAUTH2_ACTIVE"] = str(self.oauth2_active)
        os.environ["AGENT_MODEL"] = self.agent_model
        if self.agent_api_key:
            os.environ["AGENT_API_KEY"] = self.agent_api_key


eval_settings = EvalSettings()
