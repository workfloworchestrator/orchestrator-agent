# Copyright 2019-2026 SURF, GÉANT.
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

"""The form-fill skill as a pydantic-ai capability.

The skill runs *inside* the agent run, through the framework's own seams:

- ``get_toolset`` contributes the ``start_workflow_form`` handoff tool the model calls to open a form.
- ``before_model_request`` runs the skill before the first model request of a run; when it claims the
  message (a form is open) the model is skipped and the contract reply becomes
  the run's output, recorded in message history like any model answer. On the request that follows a
  handoff, the skill walks the first pages the same way.

The reply is also left on the state (``SearchState.form_reply``, transient) so the transport delivers
exactly it, in place of the model's prose.
The skill calls core through the agent's own MCP session, so a turn costs no second session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.ui import StateDeps

from orchestrator_agent.form_fill.handoff import build_handoff_toolset
from orchestrator_agent.state import Reply, SearchState

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset

    from orchestrator_agent.form_fill.skill import FormFillSkill

logger = structlog.get_logger(__name__)

FormDeps = StateDeps[SearchState]


class FormFillCapability(AbstractCapability[FormDeps]):
    """Runs the form-fill skill in front of the model, and walks a form the model hands off."""

    def __init__(self, skill: FormFillSkill, core_toolset: MCPToolset[Any]) -> None:
        self.skill = skill
        self.core_toolset = core_toolset
        self.handoff_toolset = build_handoff_toolset(skill, core_toolset.direct_call_tool)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return "form_fill"

    def get_toolset(self) -> FunctionToolset[FormDeps]:
        return self.handoff_toolset

    async def before_model_request(
        self, ctx: RunContext[FormDeps], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        state = ctx.deps.state
        if ctx.run_step == 1:
            state.form_reply = None
            reply = await self._claim(state)
        elif state.form_fill is not None and state.form_fill.status == "opening":
            reply = await self._open(state)
        else:
            return request_context
        if reply is None:
            return request_context
        state.form_reply = reply
        raise SkipModelRequest(ModelResponse(parts=[TextPart(content=reply.text)]))

    async def _claim(self, state: SearchState) -> Reply | None:
        """Offer the message to the skill; its reply, or None to let the model answer.

        A failure in the skill leaves the session as it was: the skill is an add-on, and a core (or
        engine) hiccup must not fail the request.
        """
        if not self.skill.wants(state):
            return None
        backup = state.form_fill.model_copy(deep=True) if state.form_fill is not None else None
        try:
            return await self.skill.handle(state.user_input, state, self.core_toolset.direct_call_tool)
        except Exception:
            logger.exception("Form-fill skill failed; answering with the model instead")
            state.form_fill = backup
            return None

    async def _open(self, state: SearchState) -> Reply | None:
        """Walk the form the model just handed off; None (and the model's own answer stands) if that fails."""
        try:
            return await self.skill.open(state, self.core_toolset.direct_call_tool)
        except Exception:
            logger.exception("Form-fill handoff failed; answering with the model's text instead")
            state.form_fill = None
            return None


__all__ = ["FormFillCapability"]
