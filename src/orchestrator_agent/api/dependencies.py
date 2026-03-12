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

from orchestrator_agent.agent import AgentAdapter


@cache
def get_agent(request: Request) -> AgentAdapter:
    """Dependency to provide the agent instance.

    The agent is built once and cached for the lifetime of the application.
    """
    from orchestrator.llm_settings import llm_settings

    model = llm_settings.AGENT_MODEL
    debug = llm_settings.AGENT_DEBUG

    return AgentAdapter(model, debug=debug)
