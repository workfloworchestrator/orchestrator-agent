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

from unittest.mock import patch

import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider

from orchestrator_agent.settings import AgentSettings


def test_create_model_returns_string_when_no_custom_config():
    settings = AgentSettings(AGENT_MODEL="openai:gpt-4o")
    result = settings.create_model()
    assert result == "openai:gpt-4o"
    assert isinstance(result, str)


def test_create_model_returns_openai_model_when_both_set():
    settings = AgentSettings(
        AGENT_MODEL="openai:gpt-4o",
        AGENT_API_BASE="https://my-proxy.example.com/v1",
        AGENT_API_KEY="sk-custom-key",
    )
    result = settings.create_model()
    assert isinstance(result, OpenAIChatModel)


def test_create_model_returns_openai_model_when_only_api_key():
    settings = AgentSettings(
        AGENT_MODEL="openai:gpt-4o",
        AGENT_API_KEY="sk-custom-key",
    )
    result = settings.create_model()
    assert isinstance(result, OpenAIChatModel)


@patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"})
def test_create_model_returns_openai_model_when_only_api_base():
    settings = AgentSettings(
        AGENT_MODEL="openai:gpt-4o",
        AGENT_API_BASE="http://localhost:11434/v1",
    )
    result = settings.create_model()
    assert isinstance(result, OpenAIChatModel)


@pytest.mark.parametrize(
    "model_input,expected_name",
    [
        ("openai:gpt-4o", "gpt-4o"),
        ("gpt-4o", "gpt-4o"),
    ],
)
def test_create_model_strips_provider_prefix(model_input: str, expected_name: str) -> None:
    settings = AgentSettings(
        AGENT_MODEL=model_input,
        AGENT_API_BASE="https://example.com/v1",
        AGENT_API_KEY="sk-test",
    )
    result = settings.create_model()
    assert isinstance(result, OpenAIChatModel)
    assert result.model_name == expected_name


def test_create_model_azure_prefix_uses_azure_provider():
    settings = AgentSettings(
        AGENT_MODEL="azure:gpt-4o",
        AGENT_API_BASE="https://my-resource.openai.azure.com/",
        AGENT_API_KEY="azure-key",
        AGENT_API_VERSION="2024-12-01-preview",
    )
    result = settings.create_model()
    assert isinstance(result, OpenAIChatModel)
    assert result.model_name == "gpt-4o"
    assert isinstance(result._provider, AzureProvider)


def test_create_model_api_version_triggers_azure_provider():
    settings = AgentSettings(
        AGENT_MODEL="gpt-4o",
        AGENT_API_BASE="https://my-resource.openai.azure.com/",
        AGENT_API_KEY="azure-key",
        AGENT_API_VERSION="2024-12-01-preview",
    )
    result = settings.create_model()
    assert isinstance(result, OpenAIChatModel)
    assert isinstance(result._provider, AzureProvider)


def test_create_model_openai_provider_when_no_azure():
    settings = AgentSettings(
        AGENT_MODEL="openai:gpt-4o",
        AGENT_API_BASE="https://proxy.example.com/v1",
        AGENT_API_KEY="sk-test",
    )
    result = settings.create_model()
    assert isinstance(result, OpenAIChatModel)
    assert isinstance(result._provider, OpenAIProvider)


def test_oauth2_outbound_active_defaults_to_none():
    settings = AgentSettings()
    assert settings.OAUTH2_OUTBOUND_ACTIVE is None


def test_oauth2_outbound_active_accepts_explicit_bool():
    assert AgentSettings(OAUTH2_OUTBOUND_ACTIVE=True).OAUTH2_OUTBOUND_ACTIVE is True
    assert AgentSettings(OAUTH2_OUTBOUND_ACTIVE=False).OAUTH2_OUTBOUND_ACTIVE is False


def test_agent_domain_context_defaults_to_empty():
    assert AgentSettings().AGENT_DOMAIN_CONTEXT == ""


def test_agent_domain_context_accepts_value():
    assert AgentSettings(AGENT_DOMAIN_CONTEXT="circuit codes map to imsCircuitId").AGENT_DOMAIN_CONTEXT == (
        "circuit codes map to imsCircuitId"
    )


def test_langfuse_disabled_by_default():
    settings = AgentSettings()
    assert settings.LANGFUSE_ENABLED is False


@patch.dict("os.environ", {"LANGFUSE_ENABLED": "true"})
def test_langfuse_enabled_from_env():
    settings = AgentSettings()
    assert settings.LANGFUSE_ENABLED is True
