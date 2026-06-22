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

from functools import cache

from fastapi import Request

from orchestrator_agent.agent import WFOAgent, build_agent


@cache
def get_agent(request: Request) -> WFOAgent:
    """Dependency to provide the agent instance.

    The agent is built once and cached for the lifetime of the application. MCP
    sessions are opened per-run via ``async with agent:`` in the adapters.
    """
    from orchestrator_agent.settings import agent_settings

    return build_agent(agent_settings.create_model())
