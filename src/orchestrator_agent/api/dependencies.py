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

from fastapi import Request

from orchestrator_agent.agent import WFOAgent, build_agent


def get_agent(request: Request) -> WFOAgent:
    """Dependency to provide the agent instance the AG-UI route runs.

    The app builds one agent at startup and shares it across its protocols (A2A, chat completions,
    AG-UI): the agent with the workflow form-fill skill, so a form is filled the same way on every
    route. Without that (an app started without the lifespan), it is built here once. MCP sessions are
    opened per-run via ``async with agent:`` in the adapters.
    """
    agent: WFOAgent | None = getattr(request.app.state, "agent", None)
    if agent is None:
        from orchestrator_agent.form_fill import build_form_fill_skill
        from orchestrator_agent.settings import agent_settings

        model = agent_settings.create_model()
        agent = build_agent(model, form_fill=build_form_fill_skill(model))
        request.app.state.agent = agent
    return agent


def bearer_token(request: Request) -> str | None:
    """The person's bearer token, as the request carries it; None when it carries none."""
    auth_header = request.headers.get("authorization", "")
    return auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else None
