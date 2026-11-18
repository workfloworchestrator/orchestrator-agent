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

"""The workflow form-fill skill: the state machine behind the page tools, the approval and the start.

A form is filled by the model in conversation with the person: it reads their words into the current
page and submits it (``submit``), searches a long list (``search``), brings an earlier page back
(``restart``) or ends the form (``cancel``). Everything that must not be the model's is here, in code:
which page is current and which pages core accepted (``session``), what a page's schema fills by itself
(``pages.obvious``), what the person is shown to approve (``approval``), and the start itself (``start``),
which runs only once they approved. Core's own validation is the only validation: it also refuses a
searched field's value that its options tool did not offer.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import structlog
from orchestrator.core.schemas.mcp_tools import GetWorkflowFormOptionsRequest, WorkflowFormOptions, WorkflowFormPage
from orchestrator.core.schemas.process import ProcessIdSchema
from orchestrator.core.schemas.workflow import WorkflowSchema
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import ApprovalRequired, ModelRetry
from pydantic_forms.core import FormFieldError

from orchestrator_agent.form_fill import session as fs
from orchestrator_agent.form_fill.model import fields_of, options_of
from orchestrator_agent.form_fill.pages import CallTool, approval, describe, fillable, obvious, page_of, searched
from orchestrator_agent.state import SUBSCRIPTION_ID, FormFillSession, FormPhase, SearchState
from orchestrator_agent.tool_names import (
    CREATE_WORKFLOW_TOOL,
    GET_WORKFLOW_FORM_OPTIONS_TOOL,
    JSON_DATA_PARAM,
    LIST_WORKFLOWS_TOOL,
    WORKFLOW_KEY_PARAM,
)

logger = structlog.get_logger(__name__)

# Core's tool results are validated with core's own response models (the agent depends on orchestrator-core),
# and its tool arguments are built from core's request models — nothing here spells out a field of either.
_WORKFLOW_LIST: TypeAdapter[list[WorkflowSchema]] = TypeAdapter(list[WorkflowSchema])
_WORKFLOWS_TTL = 300.0  # seconds the workflow catalogue is cached per skill instance

MAX_FOUND = 50  # options a search answers with; a list that fits comes whole, for the model to pick from
COMPLETE_NOTE = "Every page is accepted. Call start_workflow now: it asks the person to approve the start."
CANNOT_APPROVE = "this client cannot approve a start, so nothing was started"


@dataclass
class FormFillSkill:
    _catalogue: dict[str, str] | None = dc_field(default=None, init=False, repr=False)
    _catalogue_at: float = dc_field(default=0.0, init=False, repr=False)

    # --- core's catalogue ---------------------------------------------------

    async def workflows(self, call_tool: CallTool) -> dict[str, str]:
        """Core's workflows, tasks included, as ``{key: description}``; cached briefly (static per deployment)."""
        now = time.monotonic()
        if self._catalogue is not None and now - self._catalogue_at < _WORKFLOWS_TTL:
            return self._catalogue
        listed = _WORKFLOW_LIST.validate_python(await call_tool(LIST_WORKFLOWS_TOOL, {}))
        self._catalogue = {wf.name: wf.description or wf.name for wf in listed}
        self._catalogue_at = now
        return self._catalogue

    # --- the page tools ----------------------------------------------------------------------------

    async def open(
        self, state: SearchState, call_tool: CallTool, workflow_key: str, subscription_id: str | None = None
    ) -> dict[str, Any]:
        """Open the form the model handed off: its first page from core; what the model is told about it.

        A subscription the model passed on is submitted by code where the first page asks for one (every
        modify and terminate workflow's), so core's verdict on it is the first thing the model sees.
        """
        session = FormFillSession(workflow_key=workflow_key)
        try:
            first = await page_of(call_tool, workflow_key, [])
        except ModelRetry as exc:  # core refuses the workflow itself
            return {"status": "refused", "reason": str(exc)}
        state.form_fill = session
        if first.complete:
            fs.opened(session, None)
            return {"status": "complete", "workflow": workflow_key, "note": COMPLETE_NOTE}
        page = first
        fs.opened(session, page)
        logger.info("Form-fill opened by the model", workflow=workflow_key, subscription_id=subscription_id)
        if subscription_id and any(f.name == SUBSCRIPTION_ID and fillable(f) for f in fields_of(page)):
            return {
                "workflow": workflow_key,
                **await self.submit(session, call_tool, {SUBSCRIPTION_ID: subscription_id}),
            }
        return {"status": "opened", "workflow": workflow_key, "page": describe(page)}

    async def submit(self, session: FormFillSession, call_tool: CallTool, values: Mapping[str, Any]) -> dict[str, Any]:
        """Submit the current page with ``values``: core's verdict on it, and the page that follows.

        What the model sent for fields the page does not have is dropped (and said). What the page fills
        by itself goes in on top.
        """
        page = fs.current(session)
        if page is None:
            return {"error": "The form is complete: nothing to submit. Use restart_from_page to change a page."}
        specs = {field.name: field for field in fields_of(page) if fillable(field)}
        given = {name: value for name, value in values.items() if name in specs}
        ignored = [name for name in values if name not in specs]
        submitted = {**given, **obvious(page)}
        try:
            verdict = await page_of(call_tool, session.workflow_key, [*session.page_inputs, submitted])
        except ModelRetry as exc:  # no verdict on the page: core refuses, or is in trouble
            return {
                "status": "refused",
                "reason": str(exc),
                "note": "Tell the person; try again only if it was a passing failure.",
            }
        if verdict.status == "rejected":
            fs.rejected(session, submitted, verdict.errors)
            return _rejected(page, verdict.errors, ignored)
        if verdict.complete:
            fs.accepted(session, submitted, verdict.labels, None)
            return {"status": "complete", "note": COMPLETE_NOTE}
        fs.accepted(session, submitted, verdict.labels, verdict)
        result: dict[str, Any] = {"status": "accepted", "next_page": describe(verdict)}
        return {**result, "ignored": ignored} if ignored else result

    async def search(
        self, session: FormFillSession, call_tool: CallTool, field: str, words: str, limit: int = MAX_FOUND
    ) -> dict[str, Any]:
        """The options of a field of the current page that the words fit (the first ones, for none).

        A long list is searched by core's options tool; a short one is given whole. A field without
        options says what it takes instead.
        """
        page = fs.current(session)
        if page is None:
            return {"error": "The form is complete: no page to search."}
        properties = (page.schema_ or {}).get("properties") or {}
        spec = next((f for f in fields_of(page) if f.name == field and fillable(f)), None)
        prop = properties.get(field)
        if spec is None or not isinstance(prop, Mapping):
            fields = sorted(f.name for f in fields_of(page) if fillable(f))
            return {"error": f"No field {field!r} on this page. Its fields: {fields}."}
        if searched(prop):
            request = GetWorkflowFormOptionsRequest(
                workflow_key=session.workflow_key,
                page_inputs=session.page_inputs,
                field=field,
                search=words.strip() or None,
                page_values=session.draft,
                limit=limit,
            )
            try:
                answer = await call_tool(GET_WORKFLOW_FORM_OPTIONS_TOOL, request.model_dump())
            except ModelRetry as exc:  # core says why it has none: a field it depends on, a source it cannot read
                return {"error": str(exc)}
            found = WorkflowFormOptions.model_validate(answer)
            if not found.options:
                return {
                    "field": field,
                    "options": [],
                    "note": "No option contains those words: try fewer or other words.",
                }
            return {"field": field, "options": _options(found.options), "truncated": found.total > len(found.options)}
        if options := options_of(spec):  # a short list: all of it is in the page's schema already
            return {"field": field, "options": _options(options[:limit]), "total": len(options)}
        return {"field": field, "note": f"{spec.title} has no option list: it takes a {spec.kind}."}

    async def restart(self, session: FormFillSession, call_tool: CallTool, page: Any) -> dict[str, Any]:
        """Page ``page`` (an accepted one) is current again, as core now describes it; what came after it is dropped."""
        try:
            index = int(page)
        except (TypeError, ValueError):
            return {"error": "page must be the number of an accepted page."}
        if not 0 <= index < len(session.page_inputs):
            return {"error": f"No accepted page {index}: the accepted pages are 0 to {len(session.page_inputs) - 1}."}
        kept = session.page_inputs[:index]
        try:
            verdict = await page_of(call_tool, session.workflow_key, kept)
        except ModelRetry as exc:
            return {"status": "refused", "reason": str(exc)}
        if verdict.complete:  # the form no longer has that page: it is complete as it stands
            fs.restarted(session, index, None)
            return {"status": "complete", "note": COMPLETE_NOTE}
        restarted = verdict
        fs.restarted(session, index, restarted)
        if verdict.status == "rejected":
            fs.rejected(session, {}, verdict.errors)
        return {
            "status": "restarted",
            "page": describe(restarted),
            "note": "Submit it again with the corrected values; the pages after it are filled again.",
        }

    def cancel(self, state: SearchState) -> dict[str, Any]:
        """End the form: nothing was started, the session is dropped."""
        if state.form_fill is not None:
            fs.close(state.form_fill)
        state.form_fill = None
        return {"status": "cancelled", "note": "Nothing was started. Tell the person in one line."}

    # --- the start, which the person approves --------------------------------------------------------

    async def start(self, state: SearchState, call_tool: CallTool, *, approved: bool) -> dict[str, Any]:
        """Start the workflow with exactly the accepted pages, once the person approved; what the model is told.

        Unapproved, the call stops the run for the person's decision (``ApprovalRequired``, with what a
        client shows). A caller that cannot show an approval gets the form closed as prepared. When core
        refuses the start on a page, that page is current again with core's verdict, for the model to fix
        with the person. After any other failure whether core started the process is unknown, and after
        an answer the form is done whatever the answer looks like: in both cases it closes.
        """
        session = state.form_fill
        if session is None or session.status != FormPhase.CONFIRMING:
            return {"error": "The form is not complete yet: fill and submit its pages first."}
        if not state.hitl:
            prepared = approval(session).model_dump(mode="json")
            fs.close(session)
            state.form_fill = None
            return {"status": "prepared", "reason": CANNOT_APPROVE, **prepared}
        if not approved:
            raise ApprovalRequired(metadata=approval(session).model_dump(mode="json"))
        args = {WORKFLOW_KEY_PARAM: session.workflow_key, JSON_DATA_PARAM: session.page_inputs}
        try:
            result = await call_tool(CREATE_WORKFLOW_TOOL, args)
        except Exception as exc:
            if isinstance(exc, ModelRetry):  # core refused: if that is its verdict on a page, that page is open again
                verdict = await page_of(call_tool, session.workflow_key, session.page_inputs)
                if verdict.status == "rejected":
                    return self._refused(session, verdict)
            logger.exception("Form-fill: create_workflow failed", workflow=session.workflow_key)
            fs.close(session)
            state.form_fill = None
            return {"status": "failed", "reason": str(exc) or type(exc).__name__}
        fs.close(session)
        state.form_fill = None
        try:
            return {"status": "started", "process_id": str(ProcessIdSchema.model_validate(result).id)}
        except ValidationError:  # core answered, so it started; what it answered goes back as it came
            logger.error("Form-fill: create_workflow answered without a process id", workflow=session.workflow_key)
            return {"status": "started", "answer": str(result)}

    def _refused(self, session: FormFillSession, verdict: WorkflowFormPage) -> dict[str, Any]:
        """Core refused the start on a page: that page is current again, with core's verdict on it."""
        index = min(verdict.page, len(session.page_inputs) - 1) if session.page_inputs else 0
        values = session.page_inputs[index] if index < len(session.page_inputs) else {}
        page = session.pages[index] if index < len(session.pages) else verdict  # as seen, with its options
        fs.restarted(session, index, page)
        fs.rejected(session, values, verdict.errors)
        return _rejected(page, verdict.errors, [])


def _rejected(page: WorkflowFormPage, errors: Sequence[FormFieldError], ignored: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "rejected",
        "page": page.page,
        "errors": [{"field": ".".join(str(part) for part in error.loc), "message": error.msg} for error in errors],
        "note": "Fix what is named from what the person said, or ask them.",
    }
    return {**result, "ignored": list(ignored)} if ignored else result


def _options(options: Sequence[Any]) -> list[dict[str, Any]]:
    return [{"value": option.value, "label": option.label} for option in options]


__all__ = ["CANNOT_APPROVE", "COMPLETE_NOTE", "MAX_FOUND", "CallTool", "FormFillSkill"]
