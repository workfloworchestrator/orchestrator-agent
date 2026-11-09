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

"""AG-UI interrupts, as the human-in-the-loop transport of a client that shows them (the chat popup).

AG-UI (protocol 0.1.19 and later; the JS client from 1.0) lets a run end waiting for a person: ``RUN_FINISHED``
carries an interrupt outcome with one ``Interrupt`` per thing to answer (an id, a reason, a message, the
schema of the answer, and metadata), and the client answers by starting the next run with a ``resume``
entry per interrupt (``ResumeEntry``: the interrupt's id, ``resolved`` with a payload, or ``cancelled``).
A workflow form has one such stop, the start to approve: it goes out as an interrupt with AG-UI's own
reason for a gated call, ``tool_call``, showing the call as it will be made, and the resume entry that
answers it comes back as the skill's input — the decision, and what the person said when they did not
approve, which the model then acts on. The pages before it are filled in conversation, with no interrupt.

What the stop asked is remembered on the session (``form_fill.pending``); this module keeps nothing of its own.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

import structlog
from ag_ui.core import Interrupt, ResumeEntry, RunAgentInput
from pydantic import BaseModel, StrictBool, ValidationError

from orchestrator_agent.adapters.words import shown
from orchestrator_agent.form_fill.pending import PendingStop, pending_approval, pending_of
from orchestrator_agent.form_fill.session import paused
from orchestrator_agent.state import Approval, Decision, FormFillSession, FormInput, Reply

logger = structlog.get_logger(__name__)

APPROVAL_REASON = "tool_call"  # an interrupt that carries the start to approve: AG-UI's reason for a gated call
INTERRUPTS_VERSION = (0, 1, 19)  # the AG-UI release that added the interrupt lifecycle
_VERSION = re.compile(r"^\s*v?(\d+(?:\.\d+)*)")


class ApprovalAnswer(BaseModel):
    """The payload that resolves a ``tool_call`` interrupt: the decision on the start, as AG-UI shapes an approval.

    A rejection may say why, or what to change: the form reopens and the model acts on it.
    """

    approved: StrictBool  # a real JSON boolean, as pydantic-ai reads an AG-UI approval; "yes" is no decision
    reason: str | None = None


class ToolCall(BaseModel):
    """The call the agent wants approved, as it will be made."""

    name: str
    args: dict[str, Any]


class ApprovalStop(BaseModel):
    """The metadata of a ``tool_call`` interrupt: the workflow start to approve."""

    workflow_key: str
    tool: ToolCall
    form: dict[str, Any] | None  # the skill's reply as data (``FormReply``), for a client that renders it


def supports_interrupts(version: str | None) -> bool:
    """Whether a client that says it speaks this AG-UI version gets interrupts (and so can be shown a stop)."""
    match = _VERSION.match(version or "")
    return match is not None and tuple(int(part) for part in match.group(1).split(".")) >= INTERRUPTS_VERSION


# --- outbound: the skill's stop -> the interrupt it is shown as -------------------------------------------


def _form(reply: Reply) -> dict[str, Any] | None:
    """The skill's reply as data (``FormReply``), for a client that renders it; None when the text is not one."""
    try:
        data = json.loads(reply.text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def approval_interrupt(
    request_id: str, reply: Reply, approval: Approval, session: FormFillSession
) -> tuple[Interrupt, PendingStop]:
    """A ``tool_call`` interrupt for the workflow start the skill wants confirmed, shown as it will be made."""
    interrupt = Interrupt(
        id=request_id,
        reason=APPROVAL_REASON,
        tool_call_id=request_id,
        message=approval.hint,
        response_schema=ApprovalAnswer.model_json_schema(),
        metadata=ApprovalStop(
            workflow_key=session.workflow_key,
            tool=ToolCall(name=approval.tool_name, args=dict(approval.args)),
            form=_form(reply),
        ).model_dump(mode="json"),
    )
    return interrupt, pending_approval(request_id)


def pause(reply: Reply, session: FormFillSession | None) -> Interrupt | None:
    """The reply's stop as the interrupt that shows it, remembered on the session; None when the reply is no stop.

    It records on the session, so it must run before the state is persisted. The client shows the interrupt
    as soon as the run ends, so the stop is never ``unseen``.
    """
    if session is None or reply.approval is None:
        return None
    request_id = uuid.uuid4().hex
    interrupt, pending = approval_interrupt(request_id, reply, reply.approval, session)
    pending.shown = interrupt.model_dump(mode="json", by_alias=True, exclude_none=True)  # for a reopened chat
    paused(session, pending, unseen=False)
    return interrupt


# --- inbound: the run's resume entries -> the skill's input ---------------------------------------------------


def human_input(session: FormFillSession | None, resume: Sequence[ResumeEntry] | None) -> FormInput | None:
    """The person's response to the pending stop among the run's resume entries, as the skill's input.

    None when the run carries no response, or no stop is pending: the message is the model's. A cancelled
    entry is a cancel decision; a resolved approval a start, or a cancel with what the person said. A
    response that is not about the pending stop, or cannot be read, still means the person is with the
    form: an empty input shows the stop again.
    """
    pending = pending_of(session)
    if pending is None or not resume:
        return None
    entry = next((entry for entry in resume if entry.interrupt_id == pending.id), None)
    if entry is None:
        logger.warning("No resume entry answers the pending stop; showing it again", pending_id=pending.id)
        return FormInput()
    if entry.status == "cancelled":
        return FormInput(decision=Decision.CANCEL)
    return _decision(entry.payload)


def _decision(payload: Any) -> FormInput:
    try:
        answer = ApprovalAnswer.model_validate(payload)
    except ValidationError:
        logger.warning("The approval's resume payload cannot be read; showing the approval again")
        return FormInput()
    if answer.approved:
        return FormInput(decision=Decision.START)
    return FormInput(decision=Decision.CANCEL, reason=(answer.reason or "").strip() or None)


# --- the transport ---------------------------------------------------------------------------------------------


class AGUIInterrupts:
    """An AG-UI client that shows interrupts (the chat popup): a human-in-the-loop transport over AG-UI."""

    def shows_stops(self, request: RunAgentInput) -> bool:
        """The client speaks an AG-UI with interrupts: the protocol version it sends says so."""
        return supports_interrupts(request.protocol_version)

    def read(self, session: FormFillSession | None, request: RunAgentInput) -> FormInput | None:
        return human_input(session, request.resume)

    def pause(self, reply: Reply, session: FormFillSession | None, request: RunAgentInput) -> Interrupt | None:
        return pause(reply, session)

    def words(self, reply: Reply) -> str:
        """The reply as a person reads it in the chat; the interrupt carries the data."""
        return shown(reply)


__all__ = [
    "APPROVAL_REASON",
    "AGUIInterrupts",
    "ApprovalAnswer",
    "ApprovalStop",
    "ToolCall",
    "approval_interrupt",
    "human_input",
    "pause",
    "supports_interrupts",
]
