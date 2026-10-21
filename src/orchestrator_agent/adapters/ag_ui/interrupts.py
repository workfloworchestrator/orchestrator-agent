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
That is a stop of the form-fill skill and the person's response to it, so this module is the transport
for any AG-UI client that shows interrupts: a page's questions go out as one interrupt with reason
``ask_user``, the start to approve as one with AG-UI's own reason for a gated call, ``tool_call``, and the
resume entry that answers it comes back as the skill's input. No model reads either.

The questions travel in the interrupt's metadata as the skill asked them (each field's name, title, the
choices shown, whether several may be picked, whether it is required, core's problem with the last answer,
a hint), and the answer is per question: the positions of the choices picked, or the text typed. The
values behind the choices stay on the server (the session's ``pending``, from which the response is
mapped back), so a response can only pick what was shown. What the stop asked is remembered on the
session as the skill asked it; this module keeps nothing of its own.
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
from orchestrator_agent.form_fill.pending import (
    PendingAsk,
    answered_values,
    pending_approval,
    pending_ask,
    pending_of,
)
from orchestrator_agent.form_fill.session import paused
from orchestrator_agent.state import Approval, Decision, FormFillSession, FormInput, Reply

logger = structlog.get_logger(__name__)

ASK_REASON = "ask_user"  # an interrupt that carries the questions of a form page
APPROVAL_REASON = "tool_call"  # an interrupt that carries the start to approve: AG-UI's reason for a gated call
INTERRUPTS_VERSION = (0, 1, 19)  # the AG-UI release that added the interrupt lifecycle
_VERSION = re.compile(r"^\s*v?(\d+(?:\.\d+)*)")


class AskAnswers(BaseModel):
    """The payload that resolves an ``ask_user`` interrupt: one answer per question, in the order asked.

    An answer is the positions of the choices picked (several only for a question that takes several), or
    the text typed; an empty answer leaves the field to its default.
    """

    answers: list[list[int | str]]


class ApprovalAnswer(BaseModel):
    """The payload that resolves a ``tool_call`` interrupt: the decision on the start, as AG-UI shapes an approval."""

    approved: StrictBool  # a real JSON boolean, as pydantic-ai reads an AG-UI approval; "yes" is no decision
    reason: str | None = None


class AskedQuestion(BaseModel):
    """A question as the interrupt shows it: an ``AskField`` without the values behind its choices.

    An answer picks a choice by position, so the client never sees the form values.
    """

    name: str
    question: str
    choices: list[str] = []
    multiple: bool = False
    required: bool = True
    title: str = ""
    problem: str = ""
    hint: str = ""


class AskStop(BaseModel):
    """The metadata of an ``ask_user`` interrupt: the page's questions and the form they belong to."""

    workflow_key: str
    title: str
    questions: list[AskedQuestion]
    form: dict[str, Any] | None  # the skill's reply as data (``FormReply``), for a client that renders it


class ToolCall(BaseModel):
    """The call the agent wants approved, as it will be made."""

    name: str
    args: dict[str, Any]


class ApprovalStop(BaseModel):
    """The metadata of a ``tool_call`` interrupt: the workflow start to approve."""

    workflow_key: str
    tool: ToolCall
    form: dict[str, Any] | None


def supports_interrupts(version: str | None) -> bool:
    """Whether a client that says it speaks this AG-UI version gets interrupts (and so can be shown a stop)."""
    match = _VERSION.match(version or "")
    return match is not None and tuple(int(part) for part in match.group(1).split(".")) >= INTERRUPTS_VERSION


# --- outbound: a skill stop -> the interrupt it is shown as -------------------------------------------------


def _form(reply: Reply) -> dict[str, Any] | None:
    """The skill's reply as data (``FormReply``), for a client that renders it; None when the text is not one."""
    try:
        data = json.loads(reply.text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _title(session: FormFillSession) -> str:
    """What heads the questions: the title of the page being asked, or the workflow when the form gave it none."""
    return (session.pages[-1].title if session.pages else None) or session.workflow_key


def ask_interrupt(request_id: str, reply: Reply, session: FormFillSession) -> tuple[Interrupt, PendingAsk]:
    """An ``ask_user`` interrupt for the skill's questions, plus what to remember for the answer."""
    pending = pending_ask(request_id, reply.ask or [])
    # The questions as asked, without the values behind the choices (``AskedQuestion`` has no such field).
    questions = [AskedQuestion.model_validate(question) for question in pending.model_dump(mode="json")["questions"]]
    stop = AskStop(workflow_key=session.workflow_key, title=_title(session), questions=questions, form=_form(reply))
    interrupt = Interrupt(
        id=request_id,
        reason=ASK_REASON,
        message=_title(session),
        response_schema=AskAnswers.model_json_schema(),
        metadata=stop.model_dump(mode="json"),
    )
    return interrupt, pending


def approval_interrupt(
    request_id: str, reply: Reply, approval: Approval, session: FormFillSession
) -> tuple[Interrupt, PendingAsk]:
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
    """The reply's stop as the interrupt that shows it, remembered on the session; None when the reply ends the form.

    It records on the session, so it must run before the state is persisted. The client shows the interrupt
    as soon as the run ends, so the stop is never ``unseen``.
    """
    if session is None or (reply.ask is None and reply.approval is None):
        return None
    request_id = uuid.uuid4().hex
    if reply.approval is not None:
        interrupt, pending = approval_interrupt(request_id, reply, reply.approval, session)
    else:
        interrupt, pending = ask_interrupt(request_id, reply, session)
    pending.shown = interrupt.model_dump(mode="json", by_alias=True, exclude_none=True)  # for a reopened chat
    paused(session, pending, unseen=False)
    return interrupt


# --- inbound: the run's resume entries -> the skill's input ---------------------------------------------------


def human_input(session: FormFillSession | None, resume: Sequence[ResumeEntry] | None) -> FormInput | None:
    """The person's response to the pending stop among the run's resume entries, as the skill's input.

    None when the run carries no response, or no stop is pending: the message is the model's. A cancelled
    entry is a cancel decision; a resolved approval a start or a cancel; resolved answers the values of
    the fields asked. A response that is not about the pending stop, or cannot be read, still means the
    person is with the form: an empty input shows the stop again.
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
    return _decision(entry.payload) if pending.kind == "approval" else _answers(pending, entry.payload)


def _decision(payload: Any) -> FormInput:
    try:
        answer = ApprovalAnswer.model_validate(payload)
    except ValidationError:
        logger.warning("The approval's resume payload cannot be read; showing the approval again")
        return FormInput()
    return FormInput(decision=Decision.START if answer.approved else Decision.CANCEL)


def _answers(pending: PendingAsk, payload: Any) -> FormInput:
    try:
        answers = AskAnswers.model_validate(payload)
    except ValidationError:
        logger.warning("The questions' resume payload cannot be read; showing them again", pending_id=pending.id)
        return FormInput()
    values = answered_values(pending, answers.answers)
    return FormInput(values=values) if values is not None else FormInput()


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
    "ASK_REASON",
    "AGUIInterrupts",
    "ApprovalAnswer",
    "AskAnswers",
    "approval_interrupt",
    "ask_interrupt",
    "human_input",
    "pause",
    "supports_interrupts",
]
