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

from pydantic_ai.models.openai import OpenAIChatModel

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


def test_create_model_strips_provider_prefix():
    settings = AgentSettings(
        AGENT_MODEL="openai:gpt-4o",
        AGENT_API_BASE="https://example.com/v1",
        AGENT_API_KEY="sk-test",
    )
    result = settings.create_model()
    assert isinstance(result, OpenAIChatModel)
    assert result.model_name == "gpt-4o"


def test_create_model_no_prefix():
    settings = AgentSettings(
        AGENT_MODEL="gpt-4o",
        AGENT_API_BASE="https://example.com/v1",
        AGENT_API_KEY="sk-test",
    )
    result = settings.create_model()
    assert isinstance(result, OpenAIChatModel)
    assert result.model_name == "gpt-4o"
