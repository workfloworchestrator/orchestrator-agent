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

The start is the framework's tool approval, as documented: the model calls ``start_workflow`` when the form
is complete, the call stops the run until the person approved (``ApprovalRequired``), and their decision
comes back as that call's result, which the model reads like any tool result.

The skill calls core through the agent's own MCP session, so a turn costs no second session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import CombinedToolset

from orchestrator_agent.form_fill.instructions import render
from orchestrator_agent.form_fill.tools import FormDeps, PageToolset, build_handoff_toolset
from orchestrator_agent.state import FormPhase

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset

    from orchestrator_agent.form_fill.skill import FormFillSkill


class FormFillCapability(AbstractCapability[FormDeps]):
    """The form's tools and instructions for the model."""

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
        if session is None or session.status == FormPhase.DONE:
            return ""
        return render(session, hitl=ctx.deps.state.hitl)


__all__ = ["FormFillCapability"]
