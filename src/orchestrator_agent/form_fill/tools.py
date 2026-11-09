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

"""The page tools: what the model fills the current page of an open form with.

The tool set is the state. While a form is gathering, ``submit_page`` takes the current page's fields as its
parameters (core's own schema for the page, ``pages.tool_schema``) and ``search_options`` finds a long
list's values; once a page is accepted ``restart_from_page`` can bring it back; ``cancel_form`` ends the
form. Without an open form there are no tools. The tools are built anew for every run step from the
session, so the parameters follow the page. Nothing here validates a value: core judges a page, and a
searched field only takes a value a search returned (``skill``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.ui import StateDeps
from pydantic_core import SchemaValidator, core_schema

from orchestrator_agent.form_fill import session as fs
from orchestrator_agent.form_fill.pages import CallTool, tool_schema
from orchestrator_agent.state import SearchState
from orchestrator_agent.tool_names import (
    CANCEL_FORM_TOOL,
    RESTART_FROM_PAGE_TOOL,
    SEARCH_OPTIONS_TOOL,
    SUBMIT_PAGE_TOOL,
)

if TYPE_CHECKING:
    from orchestrator_agent.form_fill.skill import FormFillSkill

FormDeps = StateDeps[SearchState]

# The arguments as the model sent them: core judges a page, nothing here does.
ANY_OBJECT = SchemaValidator(core_schema.dict_schema())

MAX_CALLS = 12  # tool calls on one page before the model is told to stop and ask the person
BUDGET_SPENT = (
    "Too many tool calls on this page without progress. Stop, and ask the person in plain words what to do "
    "next (or cancel the form if they no longer want it)."
)

SUBMIT_DESCRIPTION = (
    "Submit the current page of the form. The parameters are the page's fields: fill them from what the person "
    "said, translated into the values the schema takes, nothing invented. A required field must be given; an "
    "optional one left out keeps its default. The orchestrator validates the page and answers: 'accepted' with "
    "the next page (this tool's parameters then become that page's fields), 'rejected' with what is missing or "
    "wrong (fix it from what the person said, or ask them), or 'complete' (the person is then asked to approve "
    "the start; say nothing more)."
)
SEARCH_DESCRIPTION = (
    "Find the options of a field of the current page that some words fit: for a field whose description says "
    "to search, and for any option list too long to read. Returns options as value and label; such a field "
    "only takes a value this tool returned. Empty words give the first options."
)
RESTART_DESCRIPTION = (
    "Change an answer on an earlier, accepted page: that page becomes the current page again and what came "
    "after it is dropped (those pages are filled again, since the form may change with the new value). "
    "Pages are numbered from 0; the accepted pages are listed in the instructions."
)
CANCEL_DESCRIPTION = (
    "End the form without starting anything: the person no longer wants the workflow, or asks for something "
    "else instead."
)
SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "field": {"type": "string", "description": "The field, by its name on the current page."},
        "words": {"type": "string", "description": "What the person said for it, or part of a name."},
    },
    "required": ["field", "words"],
    "additionalProperties": False,
}
RESTART_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"page": {"type": "integer", "minimum": 0, "description": "The accepted page to fill again."}},
    "required": ["page"],
    "additionalProperties": False,
}
EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


class PageToolset(AbstractToolset[FormDeps]):
    """The page tools of the open form, built per run step from the session (``FormFillSession``)."""

    def __init__(self, skill: FormFillSkill, call_tool: CallTool) -> None:
        self.skill = skill
        self.core = call_tool  # core's MCP tools, as the skill calls them

    @property
    def id(self) -> str:
        return "form_fill"

    @property
    def label(self) -> str:
        return "the workflow form's page tools"

    def _tool(self, name: str, description: str, schema: dict[str, Any]) -> ToolsetTool[FormDeps]:
        definition = ToolDefinition(name=name, description=description, parameters_json_schema=schema)
        return ToolsetTool(toolset=self, tool_def=definition, max_retries=1, args_validator=ANY_OBJECT)

    async def get_tools(self, ctx: RunContext[FormDeps]) -> dict[str, ToolsetTool[FormDeps]]:
        session = ctx.deps.state.form_fill
        if session is None or session.status == "done":
            return {}
        tools: dict[str, ToolsetTool[FormDeps]] = {}
        page = fs.current(session)
        if session.status == "gathering" and page is not None:
            tools[SUBMIT_PAGE_TOOL] = self._tool(SUBMIT_PAGE_TOOL, SUBMIT_DESCRIPTION, tool_schema(page))
            tools[SEARCH_OPTIONS_TOOL] = self._tool(SEARCH_OPTIONS_TOOL, SEARCH_DESCRIPTION, SEARCH_SCHEMA)
        if session.page_inputs:
            tools[RESTART_FROM_PAGE_TOOL] = self._tool(RESTART_FROM_PAGE_TOOL, RESTART_DESCRIPTION, RESTART_SCHEMA)
        tools[CANCEL_FORM_TOOL] = self._tool(CANCEL_FORM_TOOL, CANCEL_DESCRIPTION, EMPTY_SCHEMA)
        return tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[FormDeps], tool: ToolsetTool[FormDeps]
    ) -> Any:
        state = ctx.deps.state
        session = state.form_fill
        if session is None:
            return {"error": "No form is open."}
        if name == CANCEL_FORM_TOOL:
            return self.skill.cancel(state)
        fs.counted(session)
        if session.calls > MAX_CALLS:
            return {"error": BUDGET_SPENT}
        if name == SUBMIT_PAGE_TOOL:
            return await self.skill.submit(session, self.core, tool_args)
        if name == SEARCH_OPTIONS_TOOL:
            return await self.skill.search(
                session, self.core, str(tool_args.get("field", "")), str(tool_args.get("words", ""))
            )
        if name == RESTART_FROM_PAGE_TOOL:
            return await self.skill.restart(session, self.core, tool_args.get("page"))
        return {"error": f"Unknown tool {name}."}


__all__ = ["BUDGET_SPENT", "MAX_CALLS", "FormDeps", "PageToolset"]
