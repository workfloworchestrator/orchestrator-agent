# Copyright 2019-2026 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""kagent's HITL extension: how a kagent parent is shown the start of a workflow to approve.

kagent (1.0) uses another agent as a tool and activates ``https://kagent.dev/extensions/hitl/v1`` on the call
(the ``A2A-Extensions`` header). A task that ends in ``input-required`` with a ``tool_approval_request`` under
that URI in its status message's metadata makes the parent pause and show a person the call to approve; their
decision comes back as a message on the same task with a ``tool_approval_response``. A run that stops on
``start_workflow`` goes out as such a request, under the id of the call that waits, and the response becomes
that call's result. The wire shapes mirror ``go/api/a2a/hitl.go`` in kagent-dev/kagent (v1.0.1). Everything
here is kagent's contract; what a decision means for the form is shared (``form_fill.start``).

kagent shows one pause per call of this agent: a start asked for again while answering a response (after a
rejection with a reason) is not shown until the parent calls again, which is why a message without a
response shows the waiting start again instead of running anything.
"""

from __future__ import annotations

from typing import Any, Literal

from a2a.server.agent_execution import RequestContext
from google.protobuf import json_format
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import DeferredToolResults
from pydantic_ai.messages import ToolCallPart

from orchestrator_agent.form_fill.start import answer, question
from orchestrator_agent.state import ApprovalStop
from orchestrator_agent.tool_names import START_WORKFLOW_TOOL

HITL_EXTENSION_URI = "https://kagent.dev/extensions/hitl/v1"
HITL_EXTENSION_DESCRIPTION = (
    "Pauses in input-required with a tool_approval_request (the workflow start to confirm); resume the task "
    "with the matching tool_approval_response."
)


class HITLTool(BaseModel):
    id: str
    call_id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolApprovalRequest(BaseModel):
    type: Literal["tool_approval_request"] = "tool_approval_request"
    hint: str = ""
    tools: list[HITLTool]


class ToolApproval(BaseModel):
    id: str
    approved: bool
    rejection_reason: str = ""


class ToolApprovalResponse(BaseModel):
    type: Literal["tool_approval_response"] = "tool_approval_response"
    approvals: list[ToolApproval]


def activated(context: RequestContext) -> bool:
    """Whether the caller activated kagent's extension: only then can it show a person the start to approve."""
    return HITL_EXTENSION_URI in context.requested_extensions


def request(call: ToolCallPart, stop: ApprovalStop) -> dict[str, Any]:
    """The metadata of the ``input-required`` message that shows the start to approve."""
    tool = HITLTool(
        id=call.tool_call_id,
        call_id=call.tool_call_id,
        name=START_WORKFLOW_TOOL,
        args={"workflow_key": stop.workflow_key, "values": stop.values},
    )
    return {HITL_EXTENSION_URI: ToolApprovalRequest(hint=question(stop), tools=[tool]).model_dump(mode="json")}


def decision(call: ToolCallPart, context: RequestContext) -> tuple[DeferredToolResults | None, bool]:
    """The person's decision on the waiting start as the call's result, and whether it rejects the start outright.

    A rejection with a reason declines the start for now with those words. A message that carries no
    decision on this call is no decision.
    """
    metadata = json_format.MessageToDict(context.message.metadata) if context.message is not None else {}
    try:
        response = ToolApprovalResponse.model_validate(metadata.get(HITL_EXTENSION_URI))
    except ValidationError:
        return None, False
    decided = next((approval for approval in response.approvals if approval.id == call.tool_call_id), None)
    if decided is None:
        return None, False
    return answer(call, decided.approved, decided.rejection_reason.strip())


__all__ = [
    "HITL_EXTENSION_DESCRIPTION",
    "HITL_EXTENSION_URI",
    "HITLTool",
    "ToolApproval",
    "ToolApprovalRequest",
    "ToolApprovalResponse",
    "activated",
    "decision",
    "request",
]
