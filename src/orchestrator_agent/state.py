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
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict

# Core's vocabulary the form-fill skill relies on.
ACCEPT_VALUE = "ACCEPTED"  # the value pydantic-forms' ``Accept`` field takes
# core's name for a subscription reference: the field of its ``ModifySubscriptionPage``, the first page of
# every modify / terminate workflow.
SUBSCRIPTION_ID = "subscription_id"


class Decision(StrEnum):
    """What the human decided about the start of the workflow: their answer to the approval, never words."""

    START = "start"  # start the workflow as summarised
    CANCEL = "cancel"  # abandon the form


# --- what goes into the form-fill skill per turn, and what it hands back --------------------------------
# A form is filled through a human-in-the-loop transport only: every stop is a pause that shows the human
# questions with their options, or the start to approve, and only their structured response continues it.


@dataclass(frozen=True)
class FormInput:
    """The human's response to the pending stop: their answers as data, or their decision on the start."""

    values: dict[str, Any] | None = None
    decision: Decision | None = None


@dataclass(frozen=True)
class AskField:
    """One question for the human: the form field it fills, its wording, the options to pick from."""

    name: str
    question: str
    choices: Sequence[str] = ()  # what the human is shown to pick from (labels)
    values: Sequence[Any] = ()  # the form value behind each choice, same order; empty when the answer is free
    multiple: bool = False
    # The parts ``question`` is worded from, for a transport that words a question its own way.
    required: bool = True
    title: str = ""
    problem: str = ""  # core's message when it rejected the answer
    hint: str = ""  # one line shown with the question: how many options a long list has, what typed words matched


@dataclass(frozen=True)
class Approval:
    """The write the skill wants confirmed before it runs: shown to the human as a call to approve."""

    hint: str
    tool_name: str
    args: dict[str, Any]


class FormError(TypedDict):
    """One validation error, as pydantic-forms reports it (the shape of its ``ErrorDict``).

    Declared here because pydantic only takes a ``typing_extensions.TypedDict`` on Python < 3.12, and
    pydantic-forms' own is a ``typing.TypedDict``.
    """

    loc: tuple[int | str, ...]
    msg: str
    type: str
    ctx: NotRequired[dict[str, Any]]


class SummaryTable(TypedDict, total=False):
    """One table of a workflow's own summary page, as pydantic-forms carries it (the shape of its ``SummaryData``).

    A row per label and a column per item compared (before and after, for a modify), headed when the form
    heads them.
    """

    headers: list[str]
    labels: list[str]
    columns: list[list[Any]]


class FormReply(BaseModel):
    """What a turn of the form-fill skill came to, as data: the text of its reply, one JSON object.

    Nothing in it is phrased by this code; it carries core's own data. ``gathering``: the page core did not
    accept and what core ``rejected`` in its own words (pydantic-forms' error dicts), the ``values`` known
    so far. ``confirming``: the ``values`` to be submitted and the ``defaults`` that apply, and the
    workflow's own ``summary`` of the start when its form ends in one (core's summary form). ``started``:
    the ``process_id`` (or, when core's answer was not one, that answer as the ``reason``). ``failed``: the
    ``reason``, as the error came. ``labels`` says how the form shows a value that is one of its options
    (an id is not something a person can confirm).
    """

    workflow_key: str
    status: Literal["gathering", "confirming", "started", "cancelled", "failed"]
    page: int | None = None
    title: str | None = None
    rejected: list[FormError] = Field(default_factory=list)
    reason: str | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, Any] = Field(default_factory=dict)  # field -> the label of its value, where the form has one
    defaults: dict[str, Any] = Field(default_factory=dict)
    summary: list[SummaryTable] = Field(default_factory=list)  # the tables of the form's own summary page
    process_id: str | None = None

    def as_text(self) -> str:
        return self.model_dump_json(exclude_defaults=True)


@dataclass(frozen=True)
class Reply:
    """One turn's answer: the ``FormReply`` as text, and the stop it is — questions or an approval — if any."""

    text: str
    ask: Sequence[AskField] | None = None
    approval: Approval | None = None


class FormFillSession(BaseModel):
    """The state of one workflow form being filled over A2A turns (persisted with the thread).

    ``values`` is everything the human answered (or an interpreter made of their words), keyed by field
    name and applied to a page when its field appears; ``page_inputs`` is the last walk's validated pages —
    exactly what ``create_workflow`` is called with; ``pages`` is their schemas as core sent them, from
    which the page models are rebuilt each turn (a dynamic model cannot be persisted).
    """

    workflow_key: str
    status: Literal["opening", "gathering", "confirming", "done"] = "gathering"  # opening = handed off, not walked yet
    values: dict[str, Any] = Field(default_factory=dict)
    pages: list[dict[str, Any]] = Field(default_factory=list)  # the schemas of the last walk's pages, in order
    page_inputs: list[dict[str, Any]] = Field(default_factory=list)
    # accept fields: "<page>:<name>" -> the fingerprint of what the consent was given for (see the skill)
    consents: dict[str, str] = Field(default_factory=dict)
    asked: list[str] = Field(default_factory=list)  # fields of optional-only pages already asked once
    interpreted: dict[str, str] = Field(default_factory=dict)  # field -> the person's words already interpreted
    # widget field -> what the person's words resolved to: {"value": <option value>, "label": <its label>}
    resolved: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # cascade widget field -> the steps chosen before its value ({"node": <node>}): never sent to core
    steps: dict[str, dict[str, Any]] = Field(default_factory=dict)
    pending: dict[str, Any] | None = None  # the stop the transport sent as a pause, to map the response back
    # The pending stop answered a response rather than a message: a parent runtime that pauses once per call
    # has not shown it yet, and its next message asks for it again.
    unseen: bool = False


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
    form_input: FormInput | None = Field(default=None, exclude=True)  # the human's response carried by this turn
    hitl: bool = Field(default=False, exclude=True)  # this turn's caller can show a pause (a form needs it)
