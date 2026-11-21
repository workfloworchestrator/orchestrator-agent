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
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EVALS_DIR = Path(__file__).parent


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=EVALS_DIR / ".env", extra="ignore", frozen=True)

    # The orchestrator-core MCP server to evaluate against (docker-compose.yml).
    wfo_core_mcp_url: str = "http://localhost:8086/mcp"
    # The bundled core runs authless; outbound OAuth2 would demand a token URL.
    oauth2_active: bool = False

    dataset_file: Path = EVALS_DIR / "wfo_search_dataset.yaml"
    # The models to evaluate, one ``EvalModel`` entry each. An entry runs when the variables it names are set;
    # scores are only comparable across runs on the same model.
    models_file: Path = EVALS_DIR.parent / "eval-models.json"

    def apply_to_environment(self) -> None:
        """Publish the resolved values for the agent's ``AgentSettings`` to read."""
        os.environ["WFO_CORE_MCP_URL"] = self.wfo_core_mcp_url
        os.environ["OAUTH2_ACTIVE"] = str(self.oauth2_active)


class EvalModel(BaseModel):
    """One entry of the models file: a model and the names of the variables that hold its credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ``provider:model``, or the bare model name on an OpenAI-compatible endpoint (``api_base``).
    model: str
    # The NAME of the variable that holds the key, never the key: the file is committed.
    api_key: str
    # The name of the variable that holds the endpoint's URL; without it the provider's own URL is used.
    api_base: str | None = None
    # pydantic-ai model settings sent with every request to this model, e.g. {"temperature": 0.6}.
    settings: dict[str, Any] = Field(default_factory=dict)

    def key(self) -> str | None:
        return variable(self.api_key)

    def base(self) -> str | None:
        return variable(self.api_base) if self.api_base else None

    def is_configured(self) -> bool:
        """Whether every variable the entry names is set."""
        return bool(self.key()) and (self.api_base is None or bool(self.base()))


_DOTENV = dotenv_values(EVALS_DIR / ".env")


def variable(name: str) -> str | None:
    """A variable an entry names: from the environment, else from ``evals/.env``; None when unset or empty."""
    return os.environ.get(name) or _DOTENV.get(name) or None


eval_settings = EvalSettings()
