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
(``PendingAsk``), and turns the human's response back into the contract the skill already reads (a
JSON object keyed by field name, ``yes`` / ``no``) — so the skill logic is identical with or without
the extension. A picked chip is a label; the value behind it is what travels.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from orchestrator_agent.form_fill.contract import NO, YES, raw_pairs
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


class PendingAsk(BaseModel):
    """What the last ``input-required`` asked, so the response can be mapped back (stored on the session)."""

    id: str
    kind: Literal["ask", "approval"]
    fields: list[str | None] = Field(default_factory=list)  # per question: the form field, or None = free text
    multiple: list[bool] = Field(default_factory=list)  # per question: whether several answers were invited
    options: list[dict[str, str] | None] = Field(default_factory=list)  # per question: chip label -> form value


# --- outbound: a skill stop -> the extension payload ------------------------------------------------


def ask_request(request_id: str, ask: Sequence[AskField]) -> tuple[AskUserRequest, PendingAsk]:
    """An ``ask_user_request`` for the skill's ``AskField``s, plus what to remember for the answer."""
    questions = [HITLQuestion(question=f.question, choices=list(f.choices), multiple=f.multiple) for f in ask]
    pending = PendingAsk(
        id=request_id,
        kind="ask",
        fields=[f.name for f in ask],
        multiple=[f.multiple for f in ask],
        options=[dict(zip(f.choices, f.values, strict=True)) if f.values else None for f in ask],
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


def answers_as_text(pending: PendingAsk, response: AskUserResponse) -> str | None:
    """The answers as the contract reads them; None if they don't match the ask.

    A JSON object keyed by field name, followed by any free-text question's answer verbatim.
    """
    if pending.kind != "ask" or response.id != pending.id or len(response.answers) != len(pending.fields):
        return None
    multiple = pending.multiple or [False] * len(pending.fields)
    options = pending.options or [None] * len(pending.fields)
    values: dict[str, Any] = {}
    free: list[str] = []
    for name, many, labels, answer in zip(pending.fields, multiple, options, response.answers, strict=True):
        items = [a.strip() for a in answer.answer if a and a.strip()]
        if labels:
            items = [labels.get(item, item) for item in items]  # a chip is a label; the form wants the value
        if name is None:
            free.extend(items[:1])
            continue
        # A multi-select question's picks stay a list; a single-answer question takes its first answer. An
        # empty answer still travels: it tells the skill the reply is about the form (the page's defaults
        # then apply) instead of being judged as an unrelated message.
        values[name] = items if many else (items[0] if items else "")
    return "\n".join(([json.dumps(values)] if values else []) + free)


def approval_decision(pending: PendingAsk, response: ToolApprovalResponse) -> tuple[bool, str] | None:
    """(approved, rejection reason) for our pending call, or None when the response is not about it."""
    if pending.kind != "approval":
        return None
    for approval in response.approvals:
        if approval.id == pending.id:
            return approval.approved, approval.rejection_reason
    return None


def approval_as_text(pending: PendingAsk, response: ToolApprovalResponse) -> str | None:
    """The human's decision as the contract reads it: ``yes``, a JSON object of corrected values, or ``no``."""
    decision = approval_decision(pending, response)
    if decision is None:
        return None
    approved, reason = decision
    if approved:
        return YES
    return reason if raw_pairs(reason) else NO


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
    "ToolApproval",
    "ToolApprovalRequest",
    "ToolApprovalResponse",
    "answers_as_text",
    "approval_as_text",
    "approval_decision",
    "approval_request",
    "ask_request",
    "parse_response",
    "payload_metadata",
]
