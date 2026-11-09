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

"""LibreChat's ask-user tool, as spoken by an agent that LibreChat calls as its model.

LibreChat (>= v0.8.8) gives a chat on a model spec with ``askUserQuestion: true`` one function tool,
``ask_user_question``. A reply that calls it pauses LibreChat's run and shows the person a card of
questions with their options; their answers come back as the tool message of the next request,
``{"answers": {<question id>: <string>}}``, a picked option as its ``value`` verbatim. That makes the tool
a human-in-the-loop transport for a workflow form's one structured stop, the start to approve: the card
offers Approve and Reject, and the tool message is the person's decision. The form's pages are filled in
the conversation itself, by the model; no card is shown for them.

The approval's options carry values only this stop knows, so anything else in the answer is what the
person typed — and a person who types under the approval wants something read: it is a rejection with
their words as the reason, which the model acts on. Skipping the card rejects. What the stop asked is
remembered on the session (``form_fill.pending``); the card's call id and option values are derived from
the stop's id every time, so the reading of an answer arrives at the same ones.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any

import structlog
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call import Function
from pydantic import BaseModel, ValidationError

from orchestrator_agent.adapters.chat.request import ChatRequest
from orchestrator_agent.adapters.words import shown
from orchestrator_agent.form_fill.pending import PendingStop, pending_approval, pending_of
from orchestrator_agent.form_fill.session import paused
from orchestrator_agent.state import Approval, Decision, FormFillSession, FormInput, Reply

logger = structlog.get_logger(__name__)

CLIENT = "librechat"  # how LibreChat names itself (the ``X-Agent-Client`` header its endpoint is set to send)
ASK_TOOL = "ask_user_question"
NOT_SHOWN = "The form could not be shown"
SKIPPED = "The user chose not to answer this question."  # the answer to every question of a skipped card
REFUSED = "Error:"  # how the tool message starts when LibreChat did not accept the call

# The tool's limits, from its parameter schema (LibreChat v0.8.8).
MAX_QUESTION = 2000

APPROVE_LABEL = "Approve"
REJECT_LABEL = "Reject"
APPROVAL_HEADER = "Approval"
APPROVAL_NOTE = "Nothing is started until you approve. To change something, type it here instead."


def call_id(pending: PendingStop) -> str:
    """The tool call id of the stop's card; LibreChat returns it unchanged on the tool message."""
    return f"ask_{pending.id}"


# --- outbound: the skill's stop -> the call it is shown as ----------------------------------------------


def pause(reply: Reply, session: FormFillSession | None) -> PendingStop | None:
    """Remember the reply's stop on the session; None when the reply is no stop.

    It records on the session, so it must run before the state is persisted. A card is shown as soon as
    it is sent, so the stop is never ``unseen``.
    """
    if session is None or reply.approval is None:
        return None
    pending = pending_approval(uuid.uuid4().hex)
    paused(session, pending, unseen=False)
    return pending


def approval_card(pending: PendingStop, approval: Approval) -> dict[str, Any]:
    """The arguments of the call that shows the start to approve: approve or reject, as two options."""
    question = {
        "id": "q0",
        "header": APPROVAL_HEADER,
        "question": _cut(_plain(approval.hint), MAX_QUESTION),
        "description": APPROVAL_NOTE,
        "options": [
            {"label": APPROVE_LABEL, "value": _token(pending.id, "approve")},
            {"label": REJECT_LABEL, "value": _token(pending.id, "reject")},
        ],
    }
    return {"questions": [question]}


def _token(request_id: str, name: str) -> str:
    return f"{request_id}:{name}"


def _plain(text: str) -> str:
    return text.replace("`", "")


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- inbound: the tool results of a request -> the skill's input ---------------------------------------


def read(session: FormFillSession | None, results: Mapping[str, str]) -> FormInput | ChatCompletionMessage | None:
    """The person's response to the stop the session is paused at, from a request's tool results (by call id).

    LibreChat replays the calls of the run and their tool messages with every request, so the answer to
    the card is in ``results`` once it is given. A picked option is the decision; typed words are a
    rejection with those words as the reason. A response that cannot be read is an empty input: the stop
    is shown again. A message instead of an input answers the request without the skill: the error of a
    call LibreChat did not accept. None when no stop is pending.
    """
    pending = pending_of(session)
    if pending is None or session is None:
        return None
    content = results.get(call_id(pending))
    if content is None:
        return FormInput()
    if content.startswith(REFUSED):
        # Not shown again: the same call would be refused again. The form stays open and the person's
        # next message is read by the model.
        logger.error("LibreChat refused an ask-user call", error=content, pending_id=pending.id)
        return ChatCompletionMessage(role="assistant", content=f"{NOT_SHOWN}: {content}")
    answers = _answers(content)
    if answers is None:
        return FormInput()
    return _decision(pending, answers.get("q0", ""))


class _Answers(BaseModel):
    """The tool's result as LibreChat sends it: every question's answer, by question id."""

    answers: dict[str, str]


def _answers(content: str) -> dict[str, str] | None:
    """The answers of one tool result by question id; None when it is not the tool's result (empty: unanswered)."""
    try:
        return _Answers.model_validate_json(content).answers
    except ValidationError:
        return None


def _decision(pending: PendingStop, answer: str) -> FormInput:
    """The answer to the approval card: a start or a cancel when an option was picked, a rejection with what was typed.

    LibreChat joins the values of the picked options with ", " and puts typed text after them; an option
    value is a token of this stop, so the first part that is not one starts the typed text.
    """
    approve, reject = _token(pending.id, "approve"), _token(pending.id, "reject")
    if answer.strip() == SKIPPED:
        return FormInput(decision=Decision.CANCEL)
    picked: set[str] = set()
    rest = answer.strip()
    while rest:
        head, _, tail = rest.partition(", ")
        if head in (approve, reject):
            picked.add(head)
            rest = tail
        else:
            break
    if rest:
        return FormInput(decision=Decision.CANCEL, reason=rest)
    if approve in picked and reject not in picked:
        return FormInput(decision=Decision.START)
    if reject in picked:
        return FormInput(decision=Decision.CANCEL)
    return FormInput()


# --- the transport: all of the above, for a chat-completions request -------------------------------------


class LibreChatHitl:
    """LibreChat as the client: a human-in-the-loop transport over chat completions (``adapters.hitl.HumanInTheLoop``).

    What makes LibreChat show a stop is an assistant message that ends in a call of its ask-user tool.
    """

    def shows_stops(self, request: ChatRequest) -> bool:
        """Whether this chat can show a card: it can when it offers a tool.

        A chat on the model spec offers the ask-user tool; the endpoint used without it offers none, and
        LibreChat would hand a call back as a user message. The tool's name is not what is checked.
        """
        return bool(request.tools)

    def read(self, session: FormFillSession | None, request: ChatRequest) -> FormInput | ChatCompletionMessage | None:
        """The answer to the stop's card as the skill's input."""
        return read(session, request.tool_results) if request.answers_a_call else None

    def pause(
        self, reply: Reply, session: FormFillSession | None, request: ChatRequest
    ) -> ChatCompletionMessage | None:
        pending = pause(reply, session)
        if pending is None or reply.approval is None:
            return None
        return _asks(shown(reply), pending, approval_card(pending, reply.approval))

    def words(self, reply: Reply) -> str:
        return shown(reply)


def _asks(text: str | None, pending: PendingStop, arguments: dict[str, Any]) -> ChatCompletionMessage:
    """An assistant message that ends in the ask-user call of the stop's card."""
    call = ChatCompletionMessageFunctionToolCall(
        id=call_id(pending), type="function", function=Function(name=ASK_TOOL, arguments=json.dumps(arguments))
    )
    return ChatCompletionMessage(role="assistant", content=text, tool_calls=[call])


__all__ = [
    "ASK_TOOL",
    "CLIENT",
    "NOT_SHOWN",
    "SKIPPED",
    "LibreChatHitl",
    "approval_card",
    "call_id",
    "pause",
    "read",
    "shown",
]
