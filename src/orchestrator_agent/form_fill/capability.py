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

- ``get_toolset`` contributes the ``start_workflow_form`` handoff and, while a form is open, the page tools
  the model fills it with (``tools``; built per run step, so ``submit_page`` takes the current page).
- ``get_instructions`` renders, per request, the rules of filling and where the form stands.
- ``before_run`` carries out the person's decision on a pending approval, before anything is rendered for
  the model: a start, a cancel, or the form reopened with what they said.
- ``before_model_request`` answers in the model's place where the skill has the answer — the outcome of
  that decision, and the approval as soon as core says the form is complete: the model is skipped and the
  skill's reply is the run's output, recorded in message history like any model answer. The reply is also left on the state (``SearchState.form_reply``, transient): the transport turns
  it into the client's approval (the only structured stop a form has).

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
from pydantic_ai.toolsets import CombinedToolset

from orchestrator_agent.form_fill.handoff import FormDeps, build_handoff_toolset
from orchestrator_agent.form_fill.instructions import render
from orchestrator_agent.form_fill.tools import PageToolset
from orchestrator_agent.state import Reply, SearchState

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset

    from orchestrator_agent.form_fill.skill import FormFillSkill

logger = structlog.get_logger(__name__)


class FormFillCapability(AbstractCapability[FormDeps]):
    """The form's tools and instructions for the model; the approval and the person's decision in front of it."""

    def __init__(self, skill: FormFillSkill, core_toolset: MCPToolset[Any]) -> None:
        self.skill = skill
        self.core_toolset = core_toolset
        call_tool = core_toolset.direct_call_tool
        self.toolset: CombinedToolset[FormDeps] = CombinedToolset(
            [build_handoff_toolset(skill, call_tool), PageToolset(skill, call_tool)]
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return "form_fill"

    def get_toolset(self) -> CombinedToolset[FormDeps]:
        return self.toolset

    def get_instructions(self) -> Any:
        return self._instructions

    def _instructions(self, ctx: RunContext[FormDeps]) -> str:
        """The form's instructions while one is open; nothing otherwise."""
        session = ctx.deps.state.form_fill
        if session is None or session.status == "done":
            return ""
        return render(session, hitl=ctx.deps.state.hitl)

    async def before_run(self, ctx: RunContext[FormDeps]) -> None:
        """The person's decision on a pending approval, carried out before the run renders anything.

        The form's instructions and tools are read off the session per request, so a form the decision
        reopened must be reopened by then. A reply (the start's outcome, a cancel, the approval again) is
        left on the state for the first model request to answer with.
        """
        state = ctx.deps.state
        state.form_reply = await self._claim(state)

    async def before_model_request(
        self, ctx: RunContext[FormDeps], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        state = ctx.deps.state
        if ctx.run_step == 1 and state.form_reply is not None:
            return self._answer(state, state.form_reply)
        if (reply := self._completed(state)) is not None:
            return self._answer(state, reply)
        return request_context

    @staticmethod
    def _answer(state: SearchState, reply: Reply) -> ModelRequestContext:
        """The skill's reply in place of the model's: the run's output, and the transport's stop."""
        state.form_reply = reply
        raise SkipModelRequest(ModelResponse(parts=[TextPart(content=reply.text)]))

    async def _claim(self, state: SearchState) -> Reply | None:
        """The person's decision on the pending approval, carried out; None when the turn is the model's.

        A failure in the skill leaves the session as it was: the skill is an add-on, and a core (or
        engine) hiccup must not fail the request.
        """
        if state.form_fill is None:
            return None
        backup = state.form_fill.model_copy(deep=True)
        try:
            return await self.skill.handle(state, self.core_toolset.direct_call_tool)
        except Exception:
            logger.exception("Form-fill skill failed; answering with the model instead")
            state.form_fill = backup
            return None

    def _completed(self, state: SearchState) -> Reply | None:
        """The form is complete and no approval is out yet: the start to approve.

        For a caller that cannot approve, every page validated and nothing started instead; the form closes.
        """
        session = state.form_fill
        if session is None or session.status != "confirming" or session.pending:
            return None
        if state.hitl:
            return self.skill.summary(session)
        reply = self.skill.prepared(session)  # closes the form
        state.form_fill = None
        return reply


__all__ = ["FormFillCapability"]
