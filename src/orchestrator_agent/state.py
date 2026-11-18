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

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from orchestrator.core.schemas.mcp_tools import WorkflowFormPage
from orchestrator.core.search.filters import FilterTree
from orchestrator.core.search.query.queries import Query
from pydantic import BaseModel, ConfigDict, Field, SerializationInfo, field_serializer
from pydantic_forms.core import FormFieldError
from typing_extensions import TypedDict

# core's name for a subscription reference: the field of its ``ModifySubscriptionPage``, the first page of
# every modify / terminate workflow.
SUBSCRIPTION_ID = "subscription_id"


class SummaryTable(TypedDict, total=False):
    """One table of a workflow's own summary page, as pydantic-forms carries it (the shape of its ``SummaryData``).

    A row per label and a column per item compared (before and after, for a modify), headed when the form
    heads them.
    """

    headers: list[str]
    labels: list[str]
    columns: list[list[Any]]


class FormPhase(StrEnum):
    """Where a form stands; the strings are the session's ``status`` as persisted."""

    GATHERING = "gathering"  # the model fills the current page with the person
    CONFIRMING = "confirming"  # every page accepted: the start is out for approval
    DONE = "done"  # closed: started, cancelled, prepared or failed; the session is dropped


class ApprovalStop(BaseModel):
    """The start of a workflow as the person approves it: what a client shows on the approval.

    ``values`` are what the accepted pages hold, each as the form shows it (an id is not something a person
    can confirm); ``defaults`` the optional fields nobody gave a value; ``summary`` the tables of the
    workflow's own summary page, when its form ends in one.
    """

    workflow_key: str
    values: dict[str, Any] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    summary: list[SummaryTable] = Field(default_factory=list)


class FormFillSession(BaseModel):
    """The state of one workflow form being filled over turns (persisted with the thread).

    Code keeps the machine: ``page_inputs`` are the pages core accepted, exactly what ``create_workflow`` is
    called with, and ``page_labels`` how core shows their searched values; ``pages`` are the pages seen in
    order as core described them, the accepted ones first and then the current one while ``gathering``. The
    model fills the current page: ``draft`` is what it last submitted for it and ``rejected`` core's verdict
    on that.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)  # the activity message a chat shows the form as
    workflow_key: str
    status: FormPhase = FormPhase.GATHERING
    pages: list[WorkflowFormPage] = Field(default_factory=list)
    page_inputs: list[dict[str, Any]] = Field(default_factory=list)
    draft: dict[str, Any] = Field(default_factory=dict)
    rejected: list[FormFieldError] = Field(default_factory=list)
    page_labels: list[dict[str, Any]] = Field(default_factory=list)  # per accepted page: field -> label

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
    hitl: bool = Field(default=False, exclude=True)  # this turn's caller can show an approval (a start needs one)
