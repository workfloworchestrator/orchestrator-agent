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
``input-required`` and its status message carries a ``tool_approval_request`` under the extension URI in
``message.metadata``, the kagent parent pauses and shows the human an Approve/Reject for the call we want
confirmed natively; the answer comes back as a message on the *same task* with the matching
``tool_approval_response``. Without a valid payload the parent reports "requested input without a valid
HITL extension" — so the payload is mandatory, not decorative. A workflow form has one such stop, the
start to approve; its pages are filled by the model in the conversation the parent relays.

Wire shapes mirror ``go/api/a2a/hitl.go`` in kagent-dev/kagent (json tags; unchanged from 1.0.0-alpha3 to
alpha7), validated there only for a matching ``id``. kagent's own models cannot be a dependency:
``kagent-core`` needs ``kagent-proto``, which is not published.

The first half of this module is the wire; the second half is the A2A turn around it — the response a
message carries as the skill's input (a rejection's reason is what the model acts on), the skill's stop as
the pause — and ``KagentHitl`` is both as the transport the A2A adapter picks for a caller that activated
the extension.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any, Literal

import structlog
from a2a.server.agent_execution import RequestContext
from google.protobuf import json_format
from pydantic import BaseModel, Field, ValidationError

from orchestrator_agent.form_fill.pending import PendingStop, pending_approval
from orchestrator_agent.form_fill.session import paused
from orchestrator_agent.state import Approval, Decision, FormFillSession, FormInput, Reply

logger = structlog.get_logger(__name__)

HITL_EXTENSION_URI = "https://kagent.dev/extensions/hitl/v1"
HITL_EXTENSION_DESCRIPTION = (
    "Pauses in input-required with a tool_approval_request (the workflow start to confirm, with every value "
    "the form validated); resume the task with the matching tool_approval_response. A rejection's reason "
    "reopens the form for the agent to act on."
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


# --- outbound: the skill's stop -> the extension payload ------------------------------------------------


def approval_request(request_id: str, approval: Approval) -> tuple[ToolApprovalRequest, PendingStop]:
    """A ``tool_approval_request`` for the workflow start the skill wants confirmed."""
    tool = HITLTool(id=request_id, call_id=request_id, name=approval.tool_name, args=dict(approval.args))
    return ToolApprovalRequest(hint=approval.hint, tools=[tool]), pending_approval(request_id)


def payload_metadata(payload: BaseModel) -> dict[str, Any]:
    """Message metadata carrying the payload under the extension URI (the message also lists the URI)."""
    return {HITL_EXTENSION_URI: payload.model_dump(mode="json", exclude_none=True)}


# --- inbound: the human's response -> the decision ------------------------------------------------------


def parse_response(metadata: Mapping[str, Any] | None) -> ToolApprovalResponse | None:
    """The human's response carried by a resume message, or None (absent, a request, or malformed)."""
    raw = (metadata or {}).get(HITL_EXTENSION_URI)
    if not isinstance(raw, Mapping):
        return None
    try:
        return ToolApprovalResponse.model_validate(raw)
    except ValidationError:
        return None


def approval_decision(pending: PendingStop, response: ToolApprovalResponse) -> ToolApproval | None:
    """The human's decision on our pending call, or None when the response is not about it."""
    return next((approval for approval in response.approvals if approval.id == pending.id), None)


# --- the A2A turn: the response a message carries in, the skill's stop out as a pause -------------------
# The form-fill skill runs inside the agent (``form_fill.capability``) and knows no wire format. Before the
# run, the human's response carried by the message becomes the skill's input (``human_input``); after the
# run, the skill's stop becomes the pause payload and is remembered on the session so the next response can
# be mapped back (``pause``). The executor touches no form state itself.


def response_in(context: RequestContext) -> ToolApprovalResponse | None:
    """The human's response carried by this message's metadata, if it is one (the SDK hands it over as a struct)."""
    if context.message is None:
        return None
    return parse_response(json_format.MessageToDict(context.message.metadata))


def carries_response(context: RequestContext) -> bool:
    """Whether this message was sent as a response to a pause at all, readable or not."""
    message = context.message
    return message is not None and HITL_EXTENSION_URI in json_format.MessageToDict(message.metadata)


def human_input(
    session: FormFillSession | None, response: ToolApprovalResponse | None, *, attempted: bool = False
) -> FormInput | None:
    """The human's response as the skill's input; None when the message carries none (or no form is paused).

    An approval is a start decision, a rejection a cancel decision — with the reason the human gave, for
    the model to act on. A response that does not match the pending stop — or that was ``attempted`` but
    cannot be read — still means the human is with the form, so it is an empty input: the stop is shown again.
    """
    if session is None or not session.pending:
        return None
    if response is None:
        return FormInput() if attempted else None
    pending = PendingStop.model_validate(session.pending)
    approval = approval_decision(pending, response)
    if approval is not None:
        if approval.approved:
            return FormInput(decision=Decision.START)
        return FormInput(decision=Decision.CANCEL, reason=approval.rejection_reason.strip() or None)
    logger.warning("HITL response does not match the pending stop; showing it again", pending_id=pending.id)
    return FormInput()


def pause(reply: Reply, session: FormFillSession | None, *, answered: bool) -> ToolApprovalRequest | None:
    """The reply's stop as a pause payload, remembered on the session; None when the reply is no stop.

    ``answered`` says this turn carried a response: the pause then goes out on a resumed call, which a
    parent runtime does not show (it pauses once per call), so it is marked ``unseen`` for the skill to
    show again on the parent's next call. It records on the session, so it must run before the state is
    persisted.
    """
    if session is None or reply.approval is None:
        return None
    payload, pending = approval_request(uuid.uuid4().hex, reply.approval)
    paused(session, pending, unseen=answered)
    return payload


class KagentHitl:
    """kagent as the client: a human-in-the-loop transport over A2A (``adapters.hitl.HumanInTheLoop``)."""

    def shows_stops(self, request: RequestContext) -> bool:
        """The caller activated the extension: it asked for native pauses."""
        return HITL_EXTENSION_URI in request.requested_extensions

    def read(self, session: FormFillSession | None, request: RequestContext) -> FormInput | None:
        return human_input(session, response_in(request), attempted=carries_response(request))

    def pause(self, reply: Reply, session: FormFillSession | None, request: RequestContext) -> dict[str, Any] | None:
        payload = pause(reply, session, answered=response_in(request) is not None)
        return payload_metadata(payload) if payload is not None else None

    def words(self, reply: Reply) -> str:
        """The skill's own text: one JSON object, which a calling agent reads reliably."""
        return reply.text


__all__ = [
    "HITL_EXTENSION_DESCRIPTION",
    "HITL_EXTENSION_URI",
    "HITLTool",
    "KagentHitl",
    "ToolApproval",
    "ToolApprovalRequest",
    "ToolApprovalResponse",
    "approval_decision",
    "approval_request",
    "carries_response",
    "human_input",
    "parse_response",
    "pause",
    "payload_metadata",
    "response_in",
]
