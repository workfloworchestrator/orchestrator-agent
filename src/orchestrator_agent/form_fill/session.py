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

"""The form-fill state machine: its phases, the events a turn brings, and the transition table between them.

A form is filled over turns that arrive as separate requests, so the machine's position is data on the
session (``FormFillSession.status``) and a turn is a function of that position and one event. The table
says what each turn does; the skill carries it out. The walk over core's pages is an action, not a phase:
it ends in a page's questions (``gathering``), the start to approve (``confirming``), or a close (``done``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from orchestrator.core.schemas.mcp_tools import WorkflowFormPage

from orchestrator_agent.form_fill.model import FieldSpec, fields_of
from orchestrator_agent.form_fill.pending import PendingAsk
from orchestrator_agent.state import Decision, FormFillSession, FormInput


class FormPhase(StrEnum):
    """Where a form stands; the strings are the session's ``status`` as persisted and as replies report it."""

    OPENING = "opening"  # handed off by the model, nothing walked yet
    GATHERING = "gathering"  # a page's questions are out
    CONFIRMING = "confirming"  # the start is out for approval
    DONE = "done"  # closed: started, cancelled or failed; the session is dropped


# --- events: what one turn brings, as the transport made of the human's response ----------------------------


@dataclass(frozen=True)
class Opened:
    """The model handed the form off (``start_workflow_form``)."""


@dataclass(frozen=True)
class Answered:
    """The human answered fields."""

    values: Mapping[str, Any]


@dataclass(frozen=True)
class Approved:
    """The human approved the start."""


@dataclass(frozen=True)
class Cancelled:
    """The human rejected the stop."""


@dataclass(frozen=True)
class Unanswered:
    """A message that is no response: the pending stop went unanswered."""


@dataclass(frozen=True)
class Shown:
    """The pending stop is shown (again): it had not been shown yet, or the response carried nothing to read."""


Event = Opened | Answered | Approved | Cancelled | Unanswered | Shown


def event_of(given: FormInput | None, unseen: bool) -> Event:
    """The event a turn's input is: a cancel first, then answers, then an approval; nothing readable shows the stop."""
    if given is None:
        return Shown() if unseen else Unanswered()
    if given.decision is Decision.CANCEL:
        return Cancelled()
    if given.values:
        return Answered(given.values)
    if given.decision is Decision.START:
        return Approved()
    return Shown()


# --- transitions ---------------------------------------------------------------------------------------------


class Action(StrEnum):
    """What a turn does; the walk also produces the same stop again, since it is a function of the session."""

    WALK = "walk"  # record any answers, then walk the form from page 0
    START = "start"  # send ``create_workflow`` with the validated pages
    CANCEL = "cancel"  # close the form as cancelled
    ABANDON = "abandon"  # the stop went unanswered: drop the session, the message is the model's
    NOTHING = "nothing"  # not this form's turn


_WALK_OR_CANCEL = {Answered: Action.WALK, Approved: Action.WALK, Cancelled: Action.CANCEL, Shown: Action.WALK}

TRANSITIONS: Mapping[tuple[FormPhase, type[Event]], Action] = {
    (FormPhase.OPENING, Opened): Action.WALK,
    **{(FormPhase.OPENING, event): Action.ABANDON for event in (Answered, Approved, Cancelled, Unanswered, Shown)},
    (FormPhase.GATHERING, Opened): Action.NOTHING,
    **{(FormPhase.GATHERING, event): action for event, action in _WALK_OR_CANCEL.items()},
    (FormPhase.GATHERING, Unanswered): Action.ABANDON,
    (FormPhase.CONFIRMING, Opened): Action.NOTHING,
    **{(FormPhase.CONFIRMING, event): action for event, action in _WALK_OR_CANCEL.items()},
    (FormPhase.CONFIRMING, Approved): Action.START,
    (FormPhase.CONFIRMING, Unanswered): Action.ABANDON,
    **{(FormPhase.DONE, event): Action.NOTHING for event in (Opened, Answered, Approved, Cancelled, Unanswered, Shown)},
}


def transition(phase: FormPhase, event: Event) -> Action:
    """What the turn does in this phase on this event."""
    return TRANSITIONS[(phase, type(event))]


# --- the session's writes: every change to a form's state, by name; nothing else writes a session -----------


def opened(session: FormFillSession) -> None:
    """The handoff is being walked: the form is gathering."""
    session.status = "gathering"


def answer(session: FormFillSession, values: Mapping[str, Any]) -> None:
    """Keep what the human answered (or an interpreter made of their words), by field."""
    session.values.update(values)


def keep_resolved(session: FormFillSession, name: str, value: Any, label: str) -> None:
    """A widget field's words resolved to an option: the value replaces the words, the label is what the person saw."""
    session.values[name] = value
    session.resolved[name] = {"value": value, "label": label}


def drop_answer(session: FormFillSession, name: str) -> None:
    """Words that resolved to nothing are dropped: core never sees them."""
    session.values.pop(name, None)
    session.resolved.pop(name, None)


def begin_walk(session: FormFillSession) -> None:
    """A walk starts from page 0: this walk's pages only, so a regenerated form leaves no stale field behind."""
    session.pages = []


def saw_page(session: FormFillSession, page: WorkflowFormPage) -> None:
    """A page as core described it (and the widgets enriched it), in walk order."""
    session.pages.append(page)


def interpret_words(session: FormFillSession, name: str, words: str) -> None:
    """The person's words for a field were read once; the same words again are not read anew."""
    session.interpreted[name] = words


def stop_at(
    session: FormFillSession, pages: list[dict[str, Any]], phase: Literal["gathering", "confirming"], waiting: list[str]
) -> None:
    """The walk stopped: ``pages`` are the validated ones (what ``create_workflow`` would be called with)."""
    session.page_inputs = pages
    session.status = phase
    session.waiting = waiting


def mark_asked(session: FormFillSession, names: Sequence[str]) -> None:
    """Fields asked once (a page of optional fields nobody has touched is asked only once)."""
    session.asked.extend(name for name in names if name not in session.asked)


def consent(session: FormFillSession, key: str, seen: str, name: str) -> None:
    """Consent for an accept field, given for what the page showed (``seen``); the answer itself is consumed."""
    session.consents[key] = seen
    session.values.pop(name, None)


def paused(session: FormFillSession, pending: PendingAsk, *, unseen: bool) -> None:
    """The stop a transport sent as a pause, to map the response back; ``unseen`` when the human has not seen it yet."""
    session.pending = pending.model_dump(mode="json", exclude_none=True)  # what a transport did not set is left out
    session.unseen = unseen


def close(session: FormFillSession) -> None:
    """The form is over: started, cancelled or failed; the session is dropped by the skill."""
    session.status = "done"


# --- projections: the session read as one thing ---------------------------------------------------------------


def fields(session: FormFillSession) -> dict[str, FieldSpec]:
    """The fields of the walked pages by name (a later page wins a name)."""
    return {field.name: field for page in session.pages for field in fields_of(page)}


def given(session: FormFillSession) -> dict[str, Any]:
    """The validated pages' values as one mapping: what would be submitted (a later page wins a name)."""
    return {name: value for page in session.page_inputs for name, value in page.items()}


def fingerprint(pages: list[dict[str, Any]]) -> str:
    """What a consent is given for — the pages before it and the rest of its own — as one comparable string."""
    return hashlib.sha256(json.dumps(pages, sort_keys=True, default=str).encode()).hexdigest()


__all__ = [
    "TRANSITIONS",
    "Action",
    "Answered",
    "Approved",
    "Cancelled",
    "Event",
    "FormPhase",
    "Opened",
    "Shown",
    "Unanswered",
    "answer",
    "begin_walk",
    "close",
    "consent",
    "drop_answer",
    "event_of",
    "fields",
    "fingerprint",
    "given",
    "interpret_words",
    "keep_resolved",
    "mark_asked",
    "opened",
    "paused",
    "saw_page",
    "stop_at",
    "transition",
]
