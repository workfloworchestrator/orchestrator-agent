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

"""kagent's human-in-the-loop A2A extension, as spoken by a remote agent.

kagent (>= 1.0.0-alpha1) uses another agent as a tool through ``remote_a2a_tool``. When our task ends in
``input-required`` and its status message carries one of these payloads under the extension URI in
``message.metadata``, the kagent parent pauses and shows the human our questions (or an Approve/Reject
for the call we want confirmed) natively; the answer comes back as a message on the *same task* with the
matching response payload. Without a valid payload the parent reports "requested input without a valid
HITL extension" — so the payload is mandatory, not decorative.

Wire shapes mirror ``go/api/a2a/hitl.go`` in kagent-dev/kagent (json tags), validated there only for a
matching ``id`` and one answer per question.

This module is pure: it turns a skill ``Reply`` into a request payload, remembers what was asked
(``PendingAsk``), and turns the human's answers back into the data the skill reads (a JSON object keyed
by field name; a picked chip is a label, the value behind it is what travels). An approval is a
``ToolApproval``: a structured decision, never words.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from orchestrator_agent.state import Approval, AskField

HITL_EXTENSION_URI = "https://kagent.dev/extensions/hitl/v1"
HITL_EXTENSION_DESCRIPTION = (
    "Pauses in input-required with an ask_user_request (the form values still needed, with their allowed "
    "options) or a tool_approval_request (the workflow start to confirm); resume the task with the matching "
    "ask_user_response / tool_approval_response."
)


class HITLQuestion(BaseModel):
    question: str
    choices: list[str] = Field(default_factory=list)
    multiple: bool = False


class AskUserRequest(BaseModel):
    type: Literal["ask_user_request"] = "ask_user_request"
    id: str
    questions: list[HITLQuestion]


class AskUserAnswer(BaseModel):
    answer: list[str] = Field(default_factory=list)


class AskUserResponse(BaseModel):
    type: Literal["ask_user_response"] = "ask_user_response"
    id: str
    answers: list[AskUserAnswer] = Field(default_factory=list)


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


HITLResponse = AskUserResponse | ToolApprovalResponse
_RESPONSE: TypeAdapter[HITLResponse] = TypeAdapter(Annotated[HITLResponse, Field(discriminator="type")])


class PendingQuestion(BaseModel):
    """One question of a pending ask: which field it fills, whether several picks were invited, its chips."""

    field: str | None = None  # None = free text, passed through
    multiple: bool = False
    options: dict[str, str] | None = None  # chip label -> form value


class PendingAsk(BaseModel):
    """What the last ``input-required`` asked, so the response can be mapped back (stored on the session)."""

    id: str
    kind: Literal["ask", "approval"]
    questions: list[PendingQuestion] = Field(default_factory=list)


# --- outbound: a skill stop -> the extension payload ------------------------------------------------


def ask_request(request_id: str, ask: Sequence[AskField]) -> tuple[AskUserRequest, PendingAsk]:
    """An ``ask_user_request`` for the skill's ``AskField``s, plus what to remember for the answer."""
    questions = [HITLQuestion(question=f.question, choices=list(f.choices), multiple=f.multiple) for f in ask]
    pending = PendingAsk(
        id=request_id,
        kind="ask",
        questions=[
            PendingQuestion(
                field=f.name,
                multiple=f.multiple,
                options=dict(zip(f.choices, f.values, strict=True)) if f.values else None,
            )
            for f in ask
        ],
    )
    return AskUserRequest(id=request_id, questions=questions), pending


def approval_request(request_id: str, approval: Approval) -> tuple[ToolApprovalRequest, PendingAsk]:
    """A ``tool_approval_request`` for the workflow start the skill wants confirmed."""
    tool = HITLTool(id=request_id, call_id=request_id, name=approval.tool_name, args=dict(approval.args))
    return ToolApprovalRequest(hint=approval.hint, tools=[tool]), PendingAsk(id=request_id, kind="approval")


def payload_metadata(payload: BaseModel) -> dict[str, Any]:
    """Message metadata carrying the payload under the extension URI (the message also lists the URI)."""
    return {HITL_EXTENSION_URI: payload.model_dump(mode="json", exclude_none=True)}


# --- inbound: the human's response -> the skill's text contract -----------------------------------


def parse_response(metadata: Mapping[str, Any] | None) -> HITLResponse | None:
    """The human's response carried by a resume message, or None (absent, a request, or malformed)."""
    raw = (metadata or {}).get(HITL_EXTENSION_URI)
    if not isinstance(raw, Mapping):
        return None
    try:
        return _RESPONSE.validate_python(raw)
    except ValidationError:
        return None


class Answers(NamedTuple):
    """A human's answers to a pending ask, as the skill uses them."""

    values: dict[
        str, Any
    ]  # field -> value: a chip mapped to its value, free text as typed; unanswered questions left out
    text: str  # what was typed for a free-text question with no field (read by the interpreter), else ""


def answers_as_values(pending: PendingAsk, response: AskUserResponse) -> Answers | None:
    """The human's answers as data; None if they don't match the ask."""
    if pending.kind != "ask" or response.id != pending.id or len(response.answers) != len(pending.questions):
        return None
    values: dict[str, Any] = {}
    free: list[str] = []
    for question, answer in zip(pending.questions, response.answers, strict=True):
        items = [a for a in answer.answer if a.strip()]
        if question.options:
            items = [question.options.get(item, item) for item in items]  # a chip is a label; the form wants the value
        if question.field is None:
            free.extend(items[:1])
            continue
        # A multi-select question's picks stay a list; a single-answer question takes its first answer. An
        # unanswered question stays out: nothing is sent for the field, so the form's default applies (and a
        # required one comes back rejected by core).
        if items:
            values[question.field] = items if question.multiple else items[0]
    return Answers(values, "\n".join(free))


def approval_decision(pending: PendingAsk, response: ToolApprovalResponse) -> ToolApproval | None:
    """The human's decision on our pending call, or None when the response is not about it."""
    if pending.kind != "approval":
        return None
    return next((approval for approval in response.approvals if approval.id == pending.id), None)


__all__ = [
    "HITL_EXTENSION_DESCRIPTION",
    "HITL_EXTENSION_URI",
    "AskUserAnswer",
    "AskUserRequest",
    "AskUserResponse",
    "HITLQuestion",
    "HITLResponse",
    "HITLTool",
    "PendingAsk",
    "PendingQuestion",
    "ToolApproval",
    "ToolApprovalRequest",
    "ToolApprovalResponse",
    "Answers",
    "answers_as_values",
    "approval_decision",
    "approval_request",
    "ask_request",
    "parse_response",
    "payload_metadata",
]
