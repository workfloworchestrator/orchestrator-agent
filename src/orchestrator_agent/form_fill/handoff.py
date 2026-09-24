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
key against core's catalogue — a key that is not there is a tool retry, so the model corrects itself — and
only marks the session as opening; it never touches the form. After the model's run the A2A executor lets
the skill walk the first pages (``FormFillSkill.open``) and replaces the model's text with the contract
reply. From then on the skill claims every message of the conversation until the form is started or
cancelled.
"""

from __future__ import annotations

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.ui import StateDeps

from orchestrator_agent.form_fill.contract import SUBSCRIPTION_ID
from orchestrator_agent.form_fill.skill import CallTool, FormFillSkill
from orchestrator_agent.state import FormFillSession, SearchState
from orchestrator_agent.tool_names import START_WORKFLOW_FORM_TOOL

FormDeps = StateDeps[SearchState]


def build_handoff_toolset(skill: FormFillSkill, call_tool: CallTool) -> FunctionToolset[FormDeps]:
    """The ``start_workflow_form`` tool, checking the key against core's catalogue (through ``skill``)."""
    toolset: FunctionToolset[FormDeps] = FunctionToolset()

    @toolset.tool(name=START_WORKFLOW_FORM_TOOL)
    async def start_workflow_form(
        ctx: RunContext[FormDeps], workflow_key: str, subscription_id: str | None = None
    ) -> str:
        """Start the input form of a workflow for the caller's request.

        Pick ``workflow_key`` from ``list_workflows`` — exactly as listed — by target (create / modify /
        terminate) and the kind of thing the caller names; every value the workflow needs is asked by the
        form itself, never by you. For a workflow on an existing subscription pass its ``subscription_id``
        (from the request or earlier in the conversation). The form-fill skill then takes over: it walks the
        form, asks the caller for the values it needs and for confirmation, and starts the workflow. Its
        message replaces your answer, so after calling this reply with one short line and nothing else.
        """
        if workflow_key not in await skill.workflows(call_tool):
            raise ModelRetry(f"Unknown workflow key {workflow_key!r}: pass a key exactly as list_workflows returns it.")
        state = ctx.deps.state
        state.form_fill = FormFillSession(
            workflow_key=workflow_key,
            status="opening",
            request=state.user_input,
            values={SUBSCRIPTION_ID: subscription_id} if subscription_id else {},
        )
        return f"Form for `{workflow_key}` opened; the form-fill skill replies to the caller from here."

    return toolset


__all__ = ["FormDeps", "build_handoff_toolset"]
