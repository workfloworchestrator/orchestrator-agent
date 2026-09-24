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
from typing import Any, Literal
from uuid import UUID

from orchestrator.core.search.filters import FilterTree
from orchestrator.core.search.query.queries import Query
from pydantic import BaseModel, ConfigDict, Field

FieldKind = Literal["accept", "choice", "multi", "boolean", "integer", "number", "json", "text"]


class FormField(BaseModel):
    """What the form-fill skill remembers about a field it has seen: enough to parse and label answers."""

    name: str
    title: str
    kind: FieldKind
    required: bool = False
    display_only: bool = False
    options: dict[str, str] | None = None  # value -> label for choice/multi fields
    as_list: bool = False  # a single-select choice_list: the form wants ``[value]``
    shape: str | None = None  # for json fields: what the caller must send (keys, allowed values, counts)
    format: str | None = None  # the schema's ``format`` (uuid, long, ...) for hints in the ask
    description: str | None = None  # the schema's own description, given to an interpreter as context
    has_default: bool = False
    default: Any = None  # applies when the caller sends nothing for an optional field


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


@dataclass(frozen=True)
class Reply:
    """One turn's answer: the contract text, and the stop it represents (questions or an approval), if any."""

    text: str
    ask: Sequence[AskField] | None = None
    approval: Approval | None = None


class FormFillSession(BaseModel):
    """The state of one workflow form being filled over A2A turns (persisted with the thread).

    ``values`` is everything the caller said (a JSON object) or an interpreter made of a person's words,
    keyed by field name and applied to a page when its field appears; ``page_inputs`` is the last walk's
    validated pages — exactly what ``create_workflow`` is called with.
    """

    workflow_key: str
    status: Literal["opening", "gathering", "confirming", "done"] = "gathering"  # opening = handed off, not walked yet
    request: str = ""  # the caller's opening request (the message the model handed off)
    values: dict[str, Any] = Field(default_factory=dict)
    fields: dict[str, FormField] = Field(default_factory=dict)  # the fields of the last walk's pages
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
