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

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from orchestrator.core.schemas.mcp_tools import WorkflowFormPage
from orchestrator.core.search.filters import FilterTree
from orchestrator.core.search.query.queries import Query
from pydantic import BaseModel, ConfigDict, Field, SerializationInfo, field_serializer
from pydantic_forms.core import FormFieldError
from typing_extensions import TypedDict

# core's name for a subscription reference: the field of its ``ModifySubscriptionPage``, the first page of
# every modify / terminate workflow.
SUBSCRIPTION_ID = "subscription_id"


class Decision(StrEnum):
    """What the person decided about the start of the workflow: their answer to the approval, never words."""

    START = "start"  # start the workflow as summarised
    CANCEL = "cancel"  # do not start it (with a reason, the form reopens for the model to act on it)


# --- what goes into the form-fill skill per turn, and what it hands back --------------------------------
# A form is filled by the model in conversation with the person; the one structured stop is the start to
# approve, shown through a human-in-the-loop transport, and only the person's structured response decides it.


@dataclass(frozen=True)
class FormInput:
    """The person's response to the pending approval: their decision, and what they said when rejecting.

    An empty input (no decision) means the person is with the form but the response could not be read:
    the approval is shown again.
    """

    decision: Decision | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Approval:
    """The write the skill wants confirmed before it runs: shown to the person as a call to approve."""

    hint: str
    tool_name: str
    args: dict[str, Any]


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

    Nothing in it is phrased by this code; it carries core's own data. ``gathering``: a page core
    ``rejected`` at the start, in its own words, for the person to correct. ``confirming``: the ``values`` to
    be submitted and the ``defaults`` that apply, and the workflow's own ``summary`` of the start when its
    form ends in one (core's summary form). ``started``: the ``process_id`` (or, when core's answer was not
    one, that answer as the ``reason``). ``prepared``: every page validated, nothing started, because this
    caller cannot approve a start. ``failed``: the ``reason``, as the error came. ``labels`` says how the form
    shows a value that is one of its field's options (an id is not something a person can confirm).
    """

    workflow_key: str
    status: Literal["gathering", "confirming", "started", "cancelled", "failed", "prepared"]
    page: int | None = None
    title: str | None = None
    rejected: list[FormFieldError] = Field(default_factory=list)
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
    """One turn's answer by the skill: the ``FormReply`` as text, and the approval it stops at, if any."""

    text: str
    approval: Approval | None = None


class FormFillSession(BaseModel):
    """The state of one workflow form being filled over turns (persisted with the thread).

    Code keeps the machine: ``page_inputs`` are the pages core accepted, exactly what ``create_workflow`` is
    called with; ``pages`` are the pages seen in order as core described them (with the options their
    sources gave), the accepted ones first and then the current one while ``gathering``. The model fills the
    current page: ``draft`` is what it last submitted for it and ``rejected`` core's verdict on that;
    ``found`` are the options its searches returned, by field, which is where a long list's value must come
    from and where its label is read; ``calls`` counts its tool calls on the current page.
    """

    workflow_key: str
    status: Literal["gathering", "confirming", "done"] = "gathering"
    pages: list[WorkflowFormPage] = Field(default_factory=list)
    page_inputs: list[dict[str, Any]] = Field(default_factory=list)
    draft: dict[str, Any] = Field(default_factory=dict)
    rejected: list[FormFieldError] = Field(default_factory=list)
    found: dict[str, dict[str, str]] = Field(default_factory=dict)  # field -> {value: label}
    calls: int = 0
    pending: dict[str, Any] | None = None  # the approval the transport sent as a pause, to map the response back
    # The pending stop answered a response rather than a message: a parent runtime that pauses once per call
    # has not shown it yet, and its next message asks for it again.
    unseen: bool = False
    reason: str | None = None  # why the person rejected the start, when they said: the model acts on it

    @field_serializer("pages")
    def _pages_by_alias(self, pages: list[WorkflowFormPage], info: SerializationInfo) -> list[dict[str, Any]]:
        """Each page by its public keys (``schema``, not ``schema_``), so a persisted session validates back as it was."""
        return [page.model_dump(mode=info.mode, by_alias=True) for page in pages]


class SearchState(BaseModel):
    """Agent state for search operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_input: str = ""
    run_id: UUID | None = None
    query_id: UUID | None = None
    query: Query | None = None
    pending_filters: FilterTree | None = None
    message_history: list[dict[str, Any]] = Field(default_factory=list)
    owner: str | None = None  # the subject of the person the thread belongs to, on a route that binds threads (AG-UI)
    form_fill: FormFillSession | None = None
    form_reply: Reply | None = Field(default=None, exclude=True)  # the form-fill reply of this turn; never persisted
    form_input: FormInput | None = Field(default=None, exclude=True)  # the person's response carried by this turn
    hitl: bool = Field(default=False, exclude=True)  # this turn's caller can show a pause (a start needs one)
