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
from typing import Any, NamedTuple

import structlog
from a2a.server.agent_execution import RequestContext
from google.protobuf import json_format

from orchestrator_agent.adapters.kagent_hitl import (
    AskUserRequest,
    HITLResponse,
    PendingAsk,
    ToolApprovalRequest,
    ToolApprovalResponse,
    answers_as_values,
    approval_decision,
    approval_request,
    ask_request,
    parse_response,
)
from orchestrator_agent.state import Decision, FormFillSession, Reply, SearchState, values_in

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
    text: str  # what the skill (and the handoff tool) reads when the turn carries no data
    decision: Decision | None = None  # the human's approval, as a decision the skill acts on
    values: dict[str, Any] | None = None  # the human's answers (or a JSON correction), as data
    foreign: FormFillSession | None = field(default=None, repr=False)

    @classmethod
    def begin(
        cls, prior_state: SearchState | None, context: RequestContext, *, hitl: bool, task_id: str, user_input: str
    ) -> FormTurn:
        session, foreign = prior_session(prior_state, hitl, task_id)
        response = response_in(context) if hitl else None
        resumed = Resumed(user_input, None, None) if response is None else resume(session, response, user_input)
        return cls(
            hitl=hitl,
            task_id=task_id,
            session=session,
            text=resumed.text,
            decision=resumed.decision,
            values=resumed.values,
            foreign=foreign,
        )

    def install(self, state: SearchState) -> str:
        """Put this turn's session, decision and values on the run state; returns the text the skill reads."""
        state.form_fill, state.user_input = self.session, self.text
        state.form_decision, state.form_values = self.decision, self.values
        return self.text

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


class Resumed(NamedTuple):
    """What the skill reads for the human's response to the pending stop."""

    text: str  # the message, or what was typed for a free question
    decision: Decision | None  # an approval, or a plain rejection
    values: dict[str, Any] | None  # the answers as data, or a JSON rejection reason (a correction)


def resume(session: FormFillSession | None, response: HITLResponse, user_input: str) -> Resumed:
    """The human's response mapped for the skill.

    Answers become data (a chip is its value; an empty answer sends nothing, so the form's default applies); an
    approval is a start decision; a rejection whose reason is a JSON object is a correction (data), any other
    rejection a cancel decision. A response that does not match the pending request leaves the message as it
    is (the stop is re-asked).
    """
    if session is None or not session.hitl_request:
        return Resumed(user_input, None, None)
    pending = PendingAsk.model_validate(session.hitl_request)
    if isinstance(response, ToolApprovalResponse):
        approval = approval_decision(pending, response)
        if approval is None:
            logger.warning("HITL response does not match the pending request; re-asking", pending_id=pending.id)
            return Resumed(user_input, None, None)
        if approval.approved:
            return Resumed(user_input, Decision.START, None)
        if (correction := values_in(approval.rejection_reason)) is not None:
            return Resumed(user_input, None, correction)
        return Resumed(user_input, Decision.CANCEL, None)
    answers = answers_as_values(pending, response)
    if answers is None:
        logger.warning("HITL response does not match the pending request; re-asking", pending_id=pending.id)
        return Resumed(user_input, None, None)
    if answers.text and not answers.values:  # only words for a free question: the interpreter reads them
        return Resumed(answers.text, None, None)
    return Resumed(user_input, None, answers.values)  # the answers as data, {} when every question was left empty


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


__all__ = ["FormStop", "FormTurn", "Resumed", "prior_session", "response_in", "resume", "stop_payload"]
