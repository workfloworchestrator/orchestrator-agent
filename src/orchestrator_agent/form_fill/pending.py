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

"""What a pending stop asked, kept so the human's response can be mapped back.

A stop of the skill goes out through a human-in-the-loop transport and its answer arrives later, on
another request. What has to be remembered in between is the same whichever transport shows the stop,
and it is what the skill asked, as it asked it: its ``AskField``s — the field each question fills, its
choices and the value behind each — or that it was the approval of the start. It is stored on the session
(``FormFillSession.pending``); how a transport shows the questions and correlates the response is derived
from this and from the stop's ``id``, so no transport keeps a model of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from orchestrator_agent.state import AskField, FormFillSession


class PendingAsk(BaseModel):
    """What the last stop asked, so the response can be mapped back (stored on the session)."""

    id: str
    kind: Literal["ask", "approval"]
    questions: list[AskField] = Field(default_factory=list)  # a page's questions, as the skill asked them


def pending_ask(request_id: str, ask: Sequence[AskField]) -> PendingAsk:
    """What to remember of a page's questions: the questions themselves."""
    return PendingAsk(id=request_id, kind="ask", questions=list(ask))


def pending_approval(request_id: str) -> PendingAsk:
    """What to remember of the start to approve: only that it is the approval, and its id."""
    return PendingAsk(id=request_id, kind="approval")


def pending_of(session: FormFillSession | None) -> PendingAsk | None:
    """The stop the session is paused at; None when there is none (or what is stored is not one)."""
    if session is None or not session.pending:
        return None
    try:
        return PendingAsk.model_validate(session.pending)
    except ValidationError:
        return None


def answered_values(pending: PendingAsk, answers: Sequence[Sequence[str]]) -> dict[str, Any] | None:
    """The human's answers, one list per question asked, as field values; None if they do not fit the ask.

    A choice is mapped to the value behind it; what was typed travels as typed. A multi-select question's
    picks stay a list, a single-answer question takes its first answer. An unanswered question stays out:
    nothing is sent for the field, so the form's default applies (and a required one comes back rejected
    by core).
    """
    if pending.kind != "ask" or len(answers) != len(pending.questions):
        return None
    values: dict[str, Any] = {}
    for field, answer in zip(pending.questions, answers, strict=True):
        items: list[Any] = [a for a in answer if a.strip()]
        if field.values:
            behind = dict(zip(field.choices, field.values, strict=True))
            items = [behind.get(item, item) for item in items]
        if items:
            values[field.name] = items if field.multiple else items[0]
    return values


__all__ = ["PendingAsk", "answered_values", "pending_approval", "pending_ask", "pending_of"]
