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

"""A workflow form as the chat shows it: one AG-UI activity message that groups the tool calls it is filled with.

The model fills a form with tool calls (``form_fill.tools``). They reach the client like any other, and the
agent sends the form beside them as an activity message it replaces whenever the form moves on: where the
form stands, and which calls are its own. Which calls belong to which form is decided here, from the
thread's messages, so a client groups by the ids it is given and knows no tool by name.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from orchestrator_agent.form_fill import session as fs
from orchestrator_agent.form_fill.pages import labels
from orchestrator_agent.state import FormFillSession, FormPhase
from orchestrator_agent.tool_names import (
    CANCEL_FORM_TOOL,
    RESTART_FROM_PAGE_TOOL,
    SEARCH_OPTIONS_TOOL,
    START_WORKFLOW_FORM_TOOL,
    START_WORKFLOW_TOOL,
    SUBMIT_PAGE_TOOL,
)

FORM_ACTIVITY = "workflow_form"  # the ``activityType`` of the message
FORM_TOOLS = frozenset(
    {
        START_WORKFLOW_FORM_TOOL,
        SUBMIT_PAGE_TOOL,
        SEARCH_OPTIONS_TOOL,
        RESTART_FROM_PAGE_TOOL,
        CANCEL_FORM_TOOL,
        START_WORKFLOW_TOOL,
    }
)


class FormActivity(BaseModel):
    """The content of a ``workflow_form`` activity message: where a workflow form stands."""

    workflow_key: str
    phase: FormPhase  # gathering: being filled; confirming: waits for approval; done: closed, nothing to show
    page: int | None = None  # the page being filled, from 0
    title: str | None = None  # that page's title, when the form gives it one
    values: dict[str, Any] = {}  # what the accepted pages hold, each value as the form shows it
    calls: list[str] = []  # the ids of the tool calls the form was filled with, in order


def form_activity(session: FormFillSession, calls: Sequence[ToolCallPart]) -> FormActivity:
    """The open form as the chat shows it, with the calls that are its own."""
    given = fs.given(session)
    page = fs.current(session)
    return FormActivity(
        workflow_key=session.workflow_key,
        phase=session.status,
        page=page.page if page is not None else None,
        title=page.title if page is not None else None,
        values={**given, **labels(session, given)},
        calls=[call.tool_call_id for call in calls],
    )


def closed_form(calls: Sequence[ToolCallPart], workflow_key: str | None = None) -> FormActivity:
    """A form that is over: its calls, and the workflow the model opened it for."""
    opened = [call.args_as_dict().get("workflow_key") for call in calls if call.tool_name == START_WORKFLOW_FORM_TOOL]
    key = workflow_key or next((str(key) for key in reversed(opened) if key), "")
    return FormActivity(workflow_key=key, phase=FormPhase.DONE, calls=[call.tool_call_id for call in calls])


def form_calls(messages: Iterable[ModelMessage]) -> list[ToolCallPart]:
    """The form tool calls among a thread's messages, in order."""
    parts = (part for message in messages if isinstance(message, ModelResponse) for part in message.parts)
    return [part for part in parts if isinstance(part, ToolCallPart) and part.tool_name in FORM_TOOLS]


def form_groups(calls: Sequence[ToolCallPart]) -> list[list[ToolCallPart]]:
    """The calls of each form: a form begins at ``start_workflow_form``, a start that was retried stays with it."""
    groups: list[list[ToolCallPart]] = []
    for index, call in enumerate(calls):
        retried = index > 0 and calls[index - 1].tool_name == START_WORKFLOW_FORM_TOOL
        if groups and (call.tool_name != START_WORKFLOW_FORM_TOOL or retried):
            groups[-1].append(call)
        else:
            groups.append([call])
    return groups


__all__ = [
    "FORM_ACTIVITY",
    "FORM_TOOLS",
    "FormActivity",
    "closed_form",
    "form_activity",
    "form_calls",
    "form_groups",
]
