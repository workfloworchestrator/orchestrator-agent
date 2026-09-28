# Copyright 2019-2025 SURF, GÉANT.
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

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from orchestrator.core.search.filters import FilterTree
from orchestrator.core.search.query.queries import Query
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic_forms.exceptions import ErrorDict

# Core's vocabulary the form-fill skill relies on.
ACCEPT_VALUE = "ACCEPTED"  # the value pydantic-forms' ``Accept`` field takes
# core's name for a subscription reference: the field of its ``ModifySubscriptionPage``, the first page of
# every modify / terminate workflow.
SUBSCRIPTION_ID = "subscription_id"

# A reply that is data: a JSON object keyed by field name, parsed and checked by pydantic.
REPLY: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])


def values_in(text: str) -> dict[str, Any] | None:
    """The JSON object the reply is, as sent; None when the reply is not one (prose is never mined for values)."""
    try:
        return REPLY.validate_json(text)
    except ValidationError:
        return None


class Decision(StrEnum):
    """What a message decides about an open form, when it decides anything.

    Never read from words by this code: it is kagent's structured approval, or what the interpreter
    (a model) made of the caller's message.
    """

    START = "start"  # start the workflow as summarised
    CANCEL = "cancel"  # abandon the form


# --- what the form-fill skill hands back per turn: text for any caller, plus structure for callers that can
# use it (kagent's human-in-the-loop extension shows questions with choices, and Approve/Reject for a call) ---


@dataclass(frozen=True)
class AskField:
    """One question for the caller: the form field it fills (None = free text passed through), wording, options."""

    name: str | None
    question: str
    choices: Sequence[str] = ()  # what a person is shown to pick from (labels)
    values: Sequence[str] = ()  # the form value behind each choice, same order; empty when choices are free
    multiple: bool = False


@dataclass(frozen=True)
class Approval:
    """The write the skill wants confirmed before it runs: shown to the human as a call to approve."""

    hint: str
    tool_name: str
    args: dict[str, Any]


class FormReply(BaseModel):
    """Every answer of the form-fill skill, as data: one JSON object the caller reads (its own replies are one too).

    Nothing in it is phrased by this code; it carries core's own data. ``gathering``: the page core did not
    accept — its ``schema`` (the page model's), what core ``rejected`` in its own words (pydantic-forms' error
    dicts, or a ``reason`` when core gave no field errors), the ``values`` known so far. ``confirming``: the
    ``values`` to be submitted and the ``defaults`` that apply. ``started``: the ``process_id``. ``failed``:
    the ``reason``, as the error came.
    """

    model_config = ConfigDict(populate_by_name=True)

    workflow_key: str
    status: Literal["gathering", "confirming", "started", "cancelled", "failed"]
    page: int | None = None
    title: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    rejected: list[ErrorDict] = Field(default_factory=list)
    reason: str | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    process_id: str | None = None

    def as_text(self) -> str:
        return self.model_dump_json(by_alias=True, exclude_defaults=True)


@dataclass(frozen=True)
class Reply:
    """One turn's answer: the ``FormReply`` as text, and the stop it represents (questions or an approval), if any."""

    text: str
    ask: Sequence[AskField] | None = None
    approval: Approval | None = None


class FormFillSession(BaseModel):
    """The state of one workflow form being filled over A2A turns (persisted with the thread).

    ``values`` is everything the caller said (a JSON object) or an interpreter made of a person's words,
    keyed by field name and applied to a page when its field appears; ``page_inputs`` is the last walk's
    validated pages — exactly what ``create_workflow`` is called with; ``pages`` is their schemas as core
    sent them, from which the page models are rebuilt each turn (a dynamic model cannot be persisted).
    """

    workflow_key: str
    status: Literal["opening", "gathering", "confirming", "done"] = "gathering"  # opening = handed off, not walked yet
    request: str = ""  # the caller's opening request (the message the model handed off)
    values: dict[str, Any] = Field(default_factory=dict)
    pages: list[dict[str, Any]] = Field(default_factory=list)  # the schemas of the last walk's pages, in order
    page_inputs: list[dict[str, Any]] = Field(default_factory=list)
    accepted: dict[str, int] = Field(default_factory=dict)  # accept fields: name -> the page its consent was given for
    hitl_request: dict[str, Any] | None = None  # the pending ask/approval sent through kagent's HITL extension
    task_id: str | None = None  # HITL: the A2A task this form lives in (a session never crosses tasks)
    asked: list[str] = Field(default_factory=list)  # fields of optional-only pages already asked once
    interpreted: dict[str, str] = Field(default_factory=dict)  # field -> the person's words already interpreted


class SearchState(BaseModel):
    """Agent state for search operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_input: str = ""
    run_id: UUID | None = None
    query_id: UUID | None = None
    query: Query | None = None
    pending_filters: FilterTree | None = None
    message_history: list[dict[str, Any]] = Field(default_factory=list)
    form_fill: FormFillSession | None = None
    form_reply: Reply | None = Field(default=None, exclude=True)  # the form-fill reply of this turn; never persisted
    form_decision: Decision | None = Field(default=None, exclude=True)  # this turn's structured decision (kagent)
    form_values: dict[str, Any] | None = Field(default=None, exclude=True)  # this turn's answers as data (kagent)
