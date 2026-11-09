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

"""The form-fill state machine: its phases, what a turn's response is, and every write to a session.

A form is filled over turns that arrive as separate requests, so the machine's position is data on the
session (``FormFillSession.status``). While ``gathering`` the model fills the current page through its
tools (``tools``), and every tool call is a write here. At ``confirming`` the start is out for the
person's approval, and their response is one event (``event_of``) that the transition says what to do
with. Nothing else writes a session.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from orchestrator.core.schemas.mcp_tools import WorkflowFormPage
from pydantic_forms.core import FormFieldError

from orchestrator_agent.form_fill.model import FieldSpec, fields_of
from orchestrator_agent.form_fill.pending import PendingStop
from orchestrator_agent.state import Decision, FormFillSession, FormInput


class FormPhase(StrEnum):
    """Where a form stands; the strings are the session's ``status`` as persisted."""

    GATHERING = "gathering"  # the model fills the current page with the person
    CONFIRMING = "confirming"  # every page accepted: the start is out for approval
    DONE = "done"  # closed: started, cancelled, prepared or failed; the session is dropped


# --- events: what one turn brings, as the transport made of the person's response ---------------------------


@dataclass(frozen=True)
class Approved:
    """The person approved the start."""


@dataclass(frozen=True)
class Rejected:
    """The person did not approve the start; with a ``reason`` they said what to change."""

    reason: str | None = None


@dataclass(frozen=True)
class Shown:
    """The pending stop is shown (again): it had not been shown yet, or the response carried nothing to read."""


@dataclass(frozen=True)
class Message:
    """A message that is no response: the person wrote instead of answering (or no stop is pending)."""


Event = Approved | Rejected | Shown | Message


def event_of(given: FormInput | None, unseen: bool) -> Event:
    """The event a turn's input is: a message, a decision, or nothing readable (which shows the stop again)."""
    if given is None:
        return Shown() if unseen else Message()
    if given.decision is Decision.CANCEL:
        return Rejected((given.reason or "").strip() or None)
    if given.decision is Decision.START:
        return Approved()
    return Shown()


class Action(StrEnum):
    """What a turn does with the person's response before the model, if anything."""

    START = "start"  # send ``create_workflow`` with the validated pages
    CANCEL = "cancel"  # close the form as cancelled
    SUMMARY = "summary"  # show the start to approve again
    REOPEN = "reopen"  # back to gathering: the model acts on what the person said
    NOTHING = "nothing"  # not the form's turn: the model's


def transition(phase: FormPhase, event: Event) -> Action:
    """What the turn does in this phase on this event."""
    if phase is not FormPhase.CONFIRMING:
        return Action.NOTHING  # gathering: every message is the model's; done: the session is gone
    match event:
        case Approved():
            return Action.START
        case Rejected(reason=None):
            return Action.CANCEL
        case Rejected():
            return Action.REOPEN
        case Shown():
            return Action.SUMMARY
        case Message():
            return Action.REOPEN
    raise ValueError(f"no transition for {event!r}")  # pragma: no cover


# --- the session's writes: every change to a form's state, by name --------------------------------------------


def _fresh_page(session: FormFillSession) -> None:
    """The current page just changed: nothing submitted for it yet, no verdict, the call budget full."""
    session.draft = {}
    session.rejected = []
    session.calls = 0


def opened(session: FormFillSession, page: WorkflowFormPage | None) -> None:
    """The form's first page, as core described it: the form is gathering (without one, it is complete at once)."""
    session.pages = [page] if page is not None else []
    session.page_inputs = []
    session.status = "gathering" if page is not None else "confirming"
    _fresh_page(session)


def accepted(session: FormFillSession, values: dict[str, Any], next_page: WorkflowFormPage | None) -> None:
    """Core accepted the current page with ``values``: the next page is current, or the form is complete."""
    session.page_inputs = [*session.page_inputs, values]
    session.pages = session.pages[: len(session.page_inputs)]
    _fresh_page(session)
    if next_page is None:
        session.status = "confirming"
        session.reason = None
        return
    session.pages.append(next_page)
    session.status = "gathering"


def rejected(session: FormFillSession, values: dict[str, Any], errors: Sequence[FormFieldError]) -> None:
    """Core rejected the current page as submitted: what was sent and what core said, for the next attempt."""
    session.draft = dict(values)
    session.rejected = list(errors)


def restarted(session: FormFillSession, index: int, page: WorkflowFormPage | None) -> None:
    """Page ``index`` is current again, as core now describes it; what came after it is dropped.

    Without a page the form is complete as it stands at ``index`` (its later pages depended on what was
    dropped): the start is to be approved again.
    """
    session.page_inputs = session.page_inputs[:index]
    session.pages = session.pages[:index]
    session.pending = None
    session.reason = None
    _fresh_page(session)
    if page is None:
        session.status = "confirming"
        return
    session.pages.append(page)
    session.status = "gathering"


def found(session: FormFillSession, field: str, options: Mapping[Any, str]) -> None:
    """A search returned these options for ``field``: a value among them is accepted, its label is known."""
    known = session.found.setdefault(field, {})
    known.update({str(value): label for value, label in options.items()})


def counted(session: FormFillSession) -> None:
    """One more tool call on the current page."""
    session.calls += 1


def reopened(session: FormFillSession, reason: str | None) -> None:
    """The person did not approve the start and said why (or wrote instead): the form is the model's again."""
    session.status = "gathering"
    session.pending = None
    session.reason = reason


def paused(session: FormFillSession, pending: PendingStop, *, unseen: bool) -> None:
    """The stop a transport sent as a pause, to map the response back; ``unseen`` when the person has not seen it yet."""
    session.pending = pending.model_dump(mode="json", exclude_none=True)
    session.unseen = unseen


def close(session: FormFillSession) -> None:
    """The form is over: started, cancelled, prepared or failed; the session is dropped by the skill."""
    session.status = "done"


# --- projections: the session read as one thing ---------------------------------------------------------------


def current(session: FormFillSession) -> WorkflowFormPage | None:
    """The page being filled: the one after the accepted pages; None when the form is complete."""
    index = len(session.page_inputs)
    return session.pages[index] if index < len(session.pages) else None


def fields(session: FormFillSession) -> dict[str, FieldSpec]:
    """The fields of the pages seen, by name (a later page wins a name)."""
    return {field.name: field for page in session.pages for field in fields_of(page)}


def given(session: FormFillSession) -> dict[str, Any]:
    """The accepted pages' values as one mapping: what would be submitted (a later page wins a name)."""
    return {name: value for page in session.page_inputs for name, value in page.items()}


def known(session: FormFillSession) -> dict[str, Any]:
    """Every value in play: the accepted pages' and the current page's draft (what a dependent source may read)."""
    return {**given(session), **session.draft}


__all__ = [
    "Action",
    "Approved",
    "Event",
    "FormPhase",
    "Message",
    "Rejected",
    "Shown",
    "accepted",
    "close",
    "counted",
    "current",
    "event_of",
    "fields",
    "found",
    "given",
    "known",
    "opened",
    "paused",
    "rejected",
    "reopened",
    "restarted",
    "transition",
]
