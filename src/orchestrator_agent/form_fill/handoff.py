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

"""The model's handoff to the form-fill skill.

Deciding *that* a message asks to start a workflow, and *which* one, is a judgment call; the model makes
it by calling ``start_workflow_form`` with the key it picked from ``list_workflows``. The tool checks the
key against core's catalogue — a key that is not there is a tool retry, so the model corrects itself —
and opens the form: core's first page, with a passed subscription submitted by code. The tool's result is
the page to fill, and from then on the model fills the form with the page tools (``tools``) under the
form's instructions (``instructions``). The start itself is never the model's: the skill raises the
approval when core says the form is complete.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.ui import StateDeps

from orchestrator_agent.form_fill.pages import CallTool
from orchestrator_agent.form_fill.skill import FormFillSkill
from orchestrator_agent.state import SearchState
from orchestrator_agent.tool_names import START_WORKFLOW_FORM_TOOL

FormDeps = StateDeps[SearchState]


def build_handoff_toolset(skill: FormFillSkill, call_tool: CallTool) -> FunctionToolset[FormDeps]:
    """The ``start_workflow_form`` tool, checking the key against core's catalogue (through ``skill``)."""
    toolset: FunctionToolset[FormDeps] = FunctionToolset()

    @toolset.tool(name=START_WORKFLOW_FORM_TOOL)
    async def start_workflow_form(
        ctx: RunContext[FormDeps], workflow_key: str, subscription_id: str | None = None
    ) -> dict[str, Any]:
        """Open the input form of a workflow, to fill it with the person.

        Pick ``workflow_key`` from ``list_workflows`` — exactly as listed — by target (create / modify /
        terminate) and the kind of thing the caller names. For a workflow on an existing subscription pass
        its ``subscription_id`` (from the request or earlier in the conversation): it fills the form's
        first page. The result is the first page to fill; from then on fill the form with the page tools
        (``submit_page``, ``search_options``, ...) as the form's instructions say. A new form replaces an
        open one.
        """
        if workflow_key not in await skill.workflows(call_tool):
            raise ModelRetry(f"Unknown workflow key {workflow_key!r}: pass a key exactly as list_workflows returns it.")
        return await skill.open(ctx.deps.state, call_tool, workflow_key, subscription_id)

    return toolset


__all__ = ["FormDeps", "build_handoff_toolset"]
