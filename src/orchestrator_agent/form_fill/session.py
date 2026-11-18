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

"""A form's state, and every write to it.

A form is filled over turns that arrive as separate requests, so where it stands is data on the session
(``FormFillSession.status``): ``gathering`` while the model fills the current page through its tools
(``tools``), ``confirming`` once core accepted every page and the start waits for the person's approval,
``done`` when it is closed. Every tool call is a write here; nothing else writes a session.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from orchestrator.core.schemas.mcp_tools import WorkflowFormPage
from pydantic_forms.core import FormFieldError

from orchestrator_agent.form_fill.model import FieldSpec, fields_of
from orchestrator_agent.state import FormFillSession, FormPhase

# --- the session's writes: every change to a form's state, by name --------------------------------------------


def _fresh_page(session: FormFillSession) -> None:
    """The current page just changed: nothing submitted for it yet, no verdict."""
    session.draft = {}
    session.rejected = []


def opened(session: FormFillSession, page: WorkflowFormPage | None) -> None:
    """The form's first page, as core described it: the form is gathering (without one, it is complete at once)."""
    session.pages = [page] if page is not None else []
    session.page_inputs = []
    session.page_labels = []
    session.status = FormPhase.GATHERING if page is not None else FormPhase.CONFIRMING
    _fresh_page(session)


def accepted(
    session: FormFillSession, values: dict[str, Any], labels: dict[str, Any], next_page: WorkflowFormPage | None
) -> None:
    """Core accepted the current page with ``values``, shown as ``labels``: the next page is current, or the form is complete."""
    session.page_inputs = [*session.page_inputs, values]
    session.page_labels = [*session.page_labels, labels]
    session.pages = session.pages[: len(session.page_inputs)]
    _fresh_page(session)
    if next_page is None:
        session.status = FormPhase.CONFIRMING
        return
    session.pages.append(next_page)
    session.status = FormPhase.GATHERING


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
    session.page_labels = session.page_labels[:index]
    session.pages = session.pages[:index]
    _fresh_page(session)
    if page is None:
        session.status = FormPhase.CONFIRMING
        return
    session.pages.append(page)
    session.status = FormPhase.GATHERING


def close(session: FormFillSession) -> None:
    """The form is over: started, cancelled, prepared or failed; the session is dropped by the skill."""
    session.status = FormPhase.DONE


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


__all__ = [
    "FormPhase",
    "accepted",
    "close",
    "current",
    "fields",
    "given",
    "opened",
    "rejected",
    "restarted",
]
