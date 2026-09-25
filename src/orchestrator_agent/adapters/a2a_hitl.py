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

"""The form-fill side of an A2A turn, kagent's human-in-the-loop extension included.

The form-fill skill itself runs inside the agent (``form_fill.capability``); what is left for the
transport is in ``FormTurn``: before the run, which form session the skill continues and what text it
reads (a human's response through the extension becomes the contract: a JSON object keyed by field name,
``yes`` / ``no``); after the run, what the skill's reply becomes on the wire (``FormStop``: the contract
text, and the pause payload for a caller with the extension) and which task a paused form belongs to.
The executor touches no form state itself.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import structlog
from a2a.server.agent_execution import RequestContext
from google.protobuf import json_format

from orchestrator_agent.form_fill.hitl import (
    AskUserRequest,
    HITLResponse,
    PendingAsk,
    ToolApprovalRequest,
    ToolApprovalResponse,
    answers_as_text,
    approval_as_text,
    approval_request,
    ask_request,
    parse_response,
)
from orchestrator_agent.state import FormFillSession, Reply, SearchState

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FormStop:
    """What the form-fill skill's reply of this turn becomes on the wire."""

    text: str  # the contract text: the answer, in place of the run's prose and rendered blocks
    payload: AskUserRequest | ToolApprovalRequest | None  # the pause payload, for a caller with the extension


@dataclass
class FormTurn:
    """One A2A turn's form-fill side, so the executor edits no form state itself.

    ``begin`` decides which form session (if any) the skill continues and what text it reads; ``finish``
    turns the skill's reply into a ``FormStop`` and binds a pause to this task. A form paused in another
    task is kept aside and put back when this turn did not start one of its own.
    """

    hitl: bool
    task_id: str
    session: FormFillSession | None
    text: str  # what the skill (and the handoff tool) reads: the human's response mapped, or the message itself
    foreign: FormFillSession | None = field(default=None, repr=False)

    @classmethod
    def begin(
        cls, prior_state: SearchState | None, context: RequestContext, *, hitl: bool, task_id: str, user_input: str
    ) -> FormTurn:
        session, foreign = prior_session(prior_state, hitl, task_id)
        response = response_in(context) if hitl else None
        text = user_input if response is None else resume_text(session, response, user_input)
        return cls(hitl=hitl, task_id=task_id, session=session, text=text, foreign=foreign)

    def finish(self, state: SearchState) -> FormStop | None:
        """The reply as a stop, or None when the skill did not reply this turn. Runs before the state is persisted."""
        reply = state.form_reply
        if reply is None:
            if state.form_fill is None:
                state.form_fill = self.foreign  # another task's paused form stays where it was
            return None
        if self.hitl and state.form_fill is not None:
            state.form_fill.task_id = self.task_id
        return FormStop(text=reply.text, payload=stop_payload(reply, state) if self.hitl else None)


def response_in(context: RequestContext) -> HITLResponse | None:
    """The human's response carried by this message's metadata, if it is one (the SDK hands it over as a struct)."""
    if context.message is None:
        return None
    return parse_response(json_format.MessageToDict(context.message.metadata))


def prior_session(
    prior_state: SearchState | None, hitl: bool, task_id: str
) -> tuple[FormFillSession | None, FormFillSession | None]:
    """(the form session this turn continues, the one it must leave alone).

    A human-in-the-loop form lives in one task that pauses and resumes, so a session of another task is
    not continued — but kept, for that task.
    """
    session = prior_state.form_fill if prior_state is not None else None
    if hitl and session is not None and session.task_id not in (None, task_id):
        return None, session
    return session, None


def resume_text(session: FormFillSession | None, response: HITLResponse, user_input: str) -> str:
    """The text the skill reads for the human's response to the pending stop; the message itself if none is pending."""
    if session is None or not session.hitl_request:
        return user_input
    pending = PendingAsk.model_validate(session.hitl_request)
    if isinstance(response, ToolApprovalResponse):
        mapped = approval_as_text(pending, response)
    else:
        mapped = answers_as_text(pending, response)
    if mapped is None:
        logger.warning("HITL response does not match the pending request; re-asking", pending_id=pending.id)
        return user_input
    return mapped


def stop_payload(reply: Reply, state: SearchState) -> AskUserRequest | ToolApprovalRequest | None:
    """The pause payload for a stop (ask or approval), remembered on the session; None when the reply is final.

    It records the pending request on the session, so it must run before the state is persisted.
    """
    session = state.form_fill
    if session is None:
        return None
    if reply.ask is None and reply.approval is None:
        session.hitl_request = None
        return None
    request_id = uuid.uuid4().hex
    payload: AskUserRequest | ToolApprovalRequest
    if reply.approval is not None:
        payload, pending = approval_request(request_id, reply.approval)
    else:
        payload, pending = ask_request(request_id, reply.ask or [])
    session.hitl_request = pending.model_dump(mode="json")
    return payload


__all__ = ["FormStop", "FormTurn", "prior_session", "response_in", "resume_text", "stop_payload"]
