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
a human-in-the-loop transport for the form-fill skill: a stop of the skill goes out as a call, the tool
message is the person's response, and no model reads either.

What the stop asked is remembered on the session as the skill asked it (``form_fill.pending``); this
module keeps nothing of its own. How LibreChat is shown the stop is derived from that every time: the
cards (``ask_card``, ``approval_card``), the id of each call, and the value of each option come from the
stop's id and the position of the question and of the choice, so a later card and the reading of an
answer (``read``) arrive at the same ones. ``LibreChatHitl`` is the transport the chat-completions adapter
uses for a caller that says it is LibreChat. What the tool does not take is handled here:

- a call takes four questions: a stop with more is shown as several cards, one call after the other, and
  the skill gets the response once all of them are answered;
- a question takes twelve options: a field with more is asked as free text, its options listed;
- every question must be answered: an optional field gets a "Keep default" option;
- an answer is a string, and the card always takes typed text: an option's value is a token only this
  stop knows, so anything else is what the person typed — typing "Approve" approves nothing;
- skipping a card answers every question with a fixed sentence: an optional field then keeps its default,
  a required one (or the approval) ends the form.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import structlog
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call import Function
from pydantic import BaseModel, ValidationError

from orchestrator_agent.adapters.chat.request import ChatRequest
from orchestrator_agent.form_fill.pending import PendingAsk, answered_values, pending_approval, pending_ask, pending_of
from orchestrator_agent.state import (
    Approval,
    AskField,
    Decision,
    FormFillSession,
    FormInput,
    FormReply,
    Reply,
    SummaryTable,
)

logger = structlog.get_logger(__name__)

CLIENT = "librechat"  # how LibreChat names itself (the ``X-Agent-Client`` header its endpoint is set to send)
ASK_TOOL = "ask_user_question"
NOT_SHOWN = "The form could not be shown"
SKIPPED = "The user chose not to answer this question."  # the answer to every question of a skipped card
REFUSED = "Error:"  # how the tool message starts when LibreChat did not accept the call

# The tool's limits, from its parameter schema (LibreChat v0.8.8).
MAX_QUESTIONS = 4
MAX_OPTIONS = 12
MAX_HEADER = 80
MAX_QUESTION = 2000
MAX_DESCRIPTION = 4000
MAX_LABEL = 280

KEEP_LABEL = "Keep default"
APPROVE_LABEL = "Approve"
REJECT_LABEL = "Reject"
APPROVAL_HEADER = "Approval"
APPROVAL_NOTE = "Nothing is started until you approve. A typed answer is not read as a decision."
SUMMARY_CAPTION = "The workflow's own summary:"
UNTITLED = "unknown"  # pydantic-forms' title of a form page that was given none


def call_id(pending: PendingAsk, card: int) -> str:
    """The tool call id of one card of a stop; LibreChat returns it unchanged on the tool message."""
    return f"ask_{pending.id}_{card}"


# --- outbound: a skill stop -> the calls it is shown as ------------------------------------------------


def pause(reply: Reply, session: FormFillSession | None) -> PendingAsk | None:
    """Remember the reply's stop on the session; None when the reply ends the form.

    It records on the session, so it must run before the state is persisted. A card is shown as soon as
    it is sent, so the stop is never ``unseen``.
    """
    if session is None or (reply.ask is None and reply.approval is None):
        return None
    request_id = uuid.uuid4().hex
    pending = pending_approval(request_id) if reply.approval is not None else pending_ask(request_id, reply.ask or [])
    session.pending = pending.model_dump(mode="json")
    session.unseen = False
    return pending


def ask_card(pending: PendingAsk, session: FormFillSession, card: int) -> dict[str, Any]:
    """The arguments of the call that shows one card of a page: at most ``MAX_QUESTIONS`` of its questions."""
    header = _header(session)
    first = card * MAX_QUESTIONS
    questions = enumerate(pending.questions[first : first + MAX_QUESTIONS], first)
    return {"questions": [_question(pending.id, index, field, header) for index, field in questions]}


def approval_card(pending: PendingAsk, approval: Approval) -> dict[str, Any]:
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


def _cards(pending: PendingAsk) -> int:
    """How many calls show the stop: one for the approval, one per ``MAX_QUESTIONS`` questions of a page."""
    return 1 if pending.kind == "approval" else -(-len(pending.questions) // MAX_QUESTIONS)


def _header(session: FormFillSession) -> str:
    """What heads a card: the title of the page being asked, or the workflow when the form gave it none."""
    title = session.pages[-1].get("title") if session.pages else None
    return title if isinstance(title, str) and title and title != UNTITLED else session.workflow_key


def _question(request_id: str, index: int, field: AskField, header: str) -> dict[str, Any]:
    """One field as a question of the tool."""
    note = "required" if field.required else "optional"
    # A card is plain text: the field's title, name and whether it is required, without the markdown and the
    # "leave empty" of the skill's own wording (here an optional field is left as it is by picking an option).
    text = f"{field.title} ({field.name}, {note})" if field.title else _plain(field.question)
    question: dict[str, Any] = {
        "id": f"q{index}",
        "header": _cut(header, MAX_HEADER),
        "question": _cut(text, MAX_QUESTION),
    }
    described = [field.problem] if field.problem else []
    options = [{"label": _label(label), "value": token} for token, label in _option_values(request_id, index, field)]
    if field.choices and not options:
        described.append(f"Type one of its {len(field.choices)} options: " + "; ".join(field.choices))
    if not field.required:
        options.append({"label": KEEP_LABEL, "value": _token(request_id, f"{index}.keep")})
        described.append(f'Pick "{KEEP_LABEL}" to leave it as it is.')
    if described:
        question["description"] = _cut(" ".join(described), MAX_DESCRIPTION)
    if options:
        question["options"] = options
    if field.multiple and _listed(field):
        question["multiSelect"] = True
    return question


def _listed(field: AskField) -> bool:
    """Whether the field's choices are shown as options: they are when they fit, with "Keep default" if optional."""
    return bool(field.choices) and len(field.choices) + (0 if field.required else 1) <= MAX_OPTIONS


def _option_values(request_id: str, index: int, field: AskField) -> list[tuple[str, str]]:
    """The value of each option of a question, with the choice it stands for; none when they are not listed.

    A value is the stop's id with the position of the question and of the choice: known to this stop
    only, so never what a person typed, and the same whenever it is worked out again.
    """
    if not _listed(field):
        return []
    return [(_token(request_id, f"{index}.{position}"), label) for position, label in enumerate(field.choices)]


def _token(request_id: str, name: str) -> str:
    return f"{request_id}:{name}"


def _plain(text: str) -> str:
    return text.replace("`", "")


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _label(label: str) -> str:
    """A choice as an option label: within the tool's limit, and never empty."""
    return _cut(label, MAX_LABEL) or "(empty)"


# --- inbound: the tool results of a request -> the skill's input ---------------------------------------


def read(session: FormFillSession | None, results: Mapping[str, str]) -> FormInput | ChatCompletionMessage | None:
    """The person's response to the stop the session is paused at, from a request's tool results (by call id).

    LibreChat replays the calls of the run and their tool messages with every request, so the answers to
    all cards shown so far are in ``results``. Answers to a page are its fields' values: a picked option
    as the value behind it, typed text as typed. The approval is a decision only when one of its two
    options was picked. A response that cannot be read is an empty input: the stop is shown again.

    A message instead of an input answers the request without the skill: the next card of a page with
    more questions than one call takes, or the error of a call LibreChat did not accept. None when no
    stop is pending.
    """
    pending = pending_of(session)
    if pending is None or session is None:
        return None
    contents = _contents(pending, results)
    refusal = next((content for content in contents if content.startswith(REFUSED)), None)
    if refusal is not None:
        # Not shown again: the same call would be refused again. The form stays open and, like any
        # unanswered stop, ends with the person's next message.
        logger.error("LibreChat refused an ask-user call", error=refusal, pending_id=pending.id)
        return ChatCompletionMessage(role="assistant", content=f"{NOT_SHOWN}: {refusal}")
    answers = _all_answers(contents)
    if answers is None:
        return FormInput()
    if pending.kind == "approval":
        return _decision(pending, answers.get("q0", ""))
    return _page_answers(pending, session, len(contents), answers)


def _all_answers(contents: Sequence[str]) -> dict[str, str] | None:
    """The answers of every result by question id; None when there is none, or one is not the tool's result."""
    answers: dict[str, str] = {}
    for content in contents:
        given = _answers(content)
        if given is None:
            return None
        answers.update(given)
    return answers if contents else None


def _page_answers(
    pending: PendingAsk, session: FormFillSession, answered: int, answers: Mapping[str, str]
) -> FormInput | ChatCompletionMessage:
    """The answers to the ``answered`` cards of a page: its fields' values once all cards are in, else the next card."""
    asked = pending.questions[: answered * MAX_QUESTIONS]
    picks: list[list[str]] = []
    for index, field in enumerate(asked):
        answer = answers.get(f"q{index}", SKIPPED)
        if answer == SKIPPED:
            if field.required:
                return FormInput(decision=Decision.CANCEL)  # a required field the person will not answer
            picks.append([])
        else:
            picks.append(_picks(answer, pending.id, index, field))
    if len(asked) < len(pending.questions):
        return _asks(None, pending, answered, ask_card(pending, session, answered))
    values = answered_values(pending, picks)
    return FormInput(values=values) if values is not None else FormInput()


def _contents(pending: PendingAsk, results: Mapping[str, str]) -> list[str]:
    """The results of the stop's calls, in call order, up to the first call without one."""
    contents: list[str] = []
    for card in range(_cards(pending)):
        content = results.get(call_id(pending, card))
        if content is None:
            break
        contents.append(content)
    return contents


class _Answers(BaseModel):
    """The tool's result as LibreChat sends it: every question's answer, by question id."""

    answers: dict[str, str]


def _answers(content: str) -> dict[str, str] | None:
    """The answers of one tool result by question id; None when it is not the tool's result (empty: unanswered)."""
    try:
        return _Answers.model_validate_json(content).answers
    except ValidationError:
        return None


def _picks(answer: str, request_id: str, index: int, field: AskField) -> list[str]:
    """One answer as the choices picked and what was typed.

    LibreChat joins the values of the picked options with ", " and puts typed text after them; an option
    value is a token of this stop, so the first part that is not one starts the typed text. A typed answer
    that is one choice but for its case is that choice. "Keep default" stands for no answer.
    """
    choices = dict(_option_values(request_id, index, field))
    keep = _token(request_id, f"{index}.keep")
    picked: list[str] = []
    rest = answer.strip()
    while rest:
        head, _, tail = rest.partition(", ")
        if head == keep:
            rest = tail
        elif head in choices:
            picked.append(choices[head])
            rest = tail
        else:
            picked.append(_as_choice(rest, field.choices))
            break
    return picked


def _as_choice(typed: str, choices: Sequence[str]) -> str:
    named = [label for label in choices if label.casefold() == typed.casefold()]
    return named[0] if len(named) == 1 else typed


def _decision(pending: PendingAsk, answer: str) -> FormInput:
    """The answer to the approval card: a start or a cancel when an option was picked, nothing otherwise."""
    if answer == _token(pending.id, "approve"):
        return FormInput(decision=Decision.START)
    if answer in (_token(pending.id, "reject"), SKIPPED):
        return FormInput(decision=Decision.CANCEL)
    return FormInput()


# --- what the person reads -------------------------------------------------------------------------------


def shown(reply: Reply) -> str:
    """A reply as the person reads it: the text above the card of a stop, or the outcome when the form is over.

    The skill's own text is one JSON object, written for a calling agent; a person in a chat gets the same
    data as a line of markdown, and at the approval as a table of the values as they will be sent — always,
    as that is what is approved — with the workflow's own summary under it when its form has one.
    """
    form = _form(reply)
    if form is None:
        return reply.text
    key = f"`{form.workflow_key}`"
    if form.status == "gathering":
        titled = form.title and form.title != UNTITLED
        return f"**Workflow form {key}**" + (f" — {form.title}" if titled else "")
    if form.status == "confirming":
        tables = [line for table in form.summary for line in _summary_table(table)]
        summary = ["", SUMMARY_CAPTION, *tables] if tables else []
        return "\n".join([f"**Start workflow {key} with these values?**", *_values_table(form), *summary])
    if form.status == "started":
        outcome = f"Process id: `{form.process_id}`" if form.process_id else (form.reason or "")
        return f"Workflow {key} started. {outcome}".rstrip()
    if form.status == "cancelled":
        return f"The {key} form was cancelled; nothing was started."
    return f"The {key} form failed: {form.reason}"


def _summary_table(table: SummaryTable) -> list[str]:
    """One table of the workflow's own summary: a row per label, a column per item, headed when the form heads them."""
    columns = table.get("columns") or [[]]
    headers = [*table.get("headers", []), *[""] * len(columns)][: len(columns)]
    lines = ["", "| | " + " | ".join(_cell(header) for header in headers) + " |", "|---|" + "---|" * len(columns)]
    for row, label in enumerate(table.get("labels", [])):
        cells = [_cell(column[row]) if row < len(column) else "" for column in columns]
        lines.append(f"| {_cell(label)} | " + " | ".join(cells) + " |")
    return lines


def _values_table(form: FormReply) -> list[str]:
    """The values as they will be submitted (an id as the label the form has for it), and the defaults that apply."""
    lines: list[str] = []
    if form.values:
        lines += ["", "| Field | Value |", "|---|---|"]
        lines += [f"| {name} | {_cell(form.labels.get(name, value))} |" for name, value in form.values.items()]
    if form.defaults:
        defaults = ", ".join(f"{name} = {_cell(value)}" for name, value in form.defaults.items())
        lines += ["", f"Defaults that apply: {defaults}"]
    return lines


def _form(reply: Reply) -> FormReply | None:
    try:
        return FormReply.model_validate_json(reply.text)
    except ValidationError:
        return None


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text.replace("|", "\\|").replace("\n", " ")


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
        """The answers to the stop's cards as the skill's input; the next card while there are more to show."""
        return read(session, request.tool_results) if request.answers_a_call else None

    def pause(
        self, reply: Reply, session: FormFillSession | None, request: ChatRequest
    ) -> ChatCompletionMessage | None:
        pending = pause(reply, session)
        if pending is None or session is None:
            return None
        if reply.approval is not None:
            return _asks(shown(reply), pending, 0, approval_card(pending, reply.approval))
        if not pending.questions:
            return None
        return _asks(shown(reply), pending, 0, ask_card(pending, session, 0))

    def words(self, reply: Reply) -> str:
        return shown(reply)


def _asks(text: str | None, pending: PendingAsk, card: int, arguments: dict[str, Any]) -> ChatCompletionMessage:
    """An assistant message that ends in the ask-user call of one card of a stop."""
    call = ChatCompletionMessageFunctionToolCall(
        id=call_id(pending, card),
        type="function",
        function=Function(name=ASK_TOOL, arguments=json.dumps(arguments)),
    )
    return ChatCompletionMessage(role="assistant", content=text, tool_calls=[call])


__all__ = [
    "ASK_TOOL",
    "CLIENT",
    "NOT_SHOWN",
    "SKIPPED",
    "LibreChatHitl",
    "approval_card",
    "ask_card",
    "call_id",
    "pause",
    "read",
    "shown",
]
