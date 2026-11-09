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
which page is current and which pages core accepted (``session``), that a searched field only takes a
value a search returned, what a page's schema fills by itself (``pages.obvious``), the summary of the
start the person approves (``summary``), the start itself (``start``), and what their decision does
(``handle``). Core's own validation is the only validation.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import structlog
from orchestrator.core.schemas.mcp_tools import WorkflowFormPage
from orchestrator.core.schemas.process import ProcessIdSchema
from orchestrator.core.schemas.workflow import WorkflowSchema
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import ModelRetry
from pydantic_forms.core import FormFieldError

from orchestrator_agent.form_fill import session as fs
from orchestrator_agent.form_fill.model import FieldSpec, fields_of, options_of
from orchestrator_agent.form_fill.pages import (
    CallTool,
    accepted_values,
    defaults,
    describe,
    fillable,
    labels,
    obvious,
    page_of,
    searched,
    seen,
    summary_tables,
)
from orchestrator_agent.form_fill.session import Action, event_of, transition
from orchestrator_agent.form_fill.widgets import (
    Fetch,
    Option,
    WidgetContext,
    narrow_options,
    source_options,
    widget_target,
)
from orchestrator_agent.state import SUBSCRIPTION_ID, Approval, FormFillSession, FormReply, Reply, SearchState
from orchestrator_agent.tool_names import CREATE_WORKFLOW_TOOL, JSON_DATA_PARAM, LIST_WORKFLOWS_TOOL, WORKFLOW_KEY_PARAM

logger = structlog.get_logger(__name__)

# Core's tool results are validated with core's own response models (the agent depends on orchestrator-core),
# and its tool arguments are built from core's request models — nothing here spells out a field of either.
_WORKFLOW_LIST: TypeAdapter[list[WorkflowSchema]] = TypeAdapter(list[WorkflowSchema])
_WORKFLOWS_TTL = 300.0  # seconds the workflow catalogue is cached per skill instance

MAX_FOUND = 25  # options a search answers with
COMPLETE_NOTE = (
    "Every page is accepted. The person is now shown the values and asked to approve the start: write nothing."
)
CANNOT_APPROVE = "this client cannot approve a start, so nothing was started"


def as_reply(data: FormReply, *, approval: Approval | None = None) -> Reply:
    return Reply(data.as_text(), approval=approval)


def reply_of(session: FormFillSession, status: Any, **data: Any) -> Reply:
    return as_reply(FormReply(workflow_key=session.workflow_key, status=status, **data))


async def _no_fetch(url: str) -> Any:
    raise RuntimeError("no API of core is configured to fetch a field's options from")


def _values_of(pages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {name: value for page in pages for name, value in page.items()}


@dataclass
class FormFillSkill:
    fetch: Fetch | None = None  # core's API, where a field's source is fetched from
    _catalogue: dict[str, str] | None = dc_field(default=None, init=False, repr=False)
    _catalogue_at: float = dc_field(default=0.0, init=False, repr=False)

    def _context(self, call_tool: CallTool, values: Mapping[str, Any]) -> WidgetContext:
        """What a sourced field may use to find its options: core's tools and API, and the values known."""
        return WidgetContext(call_tool=call_tool, fetch=self.fetch or _no_fetch, values=dict(values))

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
        modify and terminate workflow's), so core's verdict on it is the first thing the model sees. A
        new form replaces any open one.
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
        page = await seen(first, self._context(call_tool, {}))
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

        What the model sent for fields the page does not have is dropped (and said). A searched field's
        value must be one a search returned, or one written into the page: anything else is refused here,
        without asking core, which takes any string for such a field. What the page fills by itself goes
        in on top.
        """
        page = fs.current(session)
        if page is None:
            return {"error": "The form is complete: nothing to submit. Use restart_from_page to change a page."}
        specs = {field.name: field for field in fields_of(page) if fillable(field)}
        given = {name: value for name, value in values.items() if name in specs}
        ignored = [name for name in values if name not in specs]
        submitted = {**given, **obvious(page)}
        if problems := _strangers(session, specs, given):
            fs.rejected(session, submitted, problems)
            return _rejected(page, problems, ignored)
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
            fs.accepted(session, submitted, None)
            return {"status": "complete", "note": COMPLETE_NOTE}
        next_page = await seen(verdict, self._context(call_tool, {**fs.given(session), **submitted}))
        fs.accepted(session, submitted, next_page)
        result: dict[str, Any] = {"status": "accepted", "next_page": describe(next_page)}
        return {**result, "ignored": ignored} if ignored else result

    async def search(
        self, session: FormFillSession, call_tool: CallTool, field: str, words: str, limit: int = MAX_FOUND
    ) -> dict[str, Any]:
        """The options of a field of the current page that the words fit (the first ones, for none).

        A long list is fetched from the field's source, searched on the server when the source can, else
        narrowed here; what is returned is remembered as the values the field takes. The page's own
        options are narrowed here. A field without options says what it takes instead.
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
            return await self._search_source(session, call_tool, spec, prop, words, limit)
        if options := options_of(spec):
            listed = [Option(value=o.value, label=o.label) for o in options]
            matches = narrow_options(listed, words, limit) if words.strip() else listed[:limit]
            return {"field": field, "options": _options(matches), "total": len(listed)}
        return {"field": field, "note": f"{spec.title} has no option list: it takes a {spec.kind}."}

    async def _search_source(
        self,
        session: FormFillSession,
        call_tool: CallTool,
        spec: FieldSpec,
        prop: Mapping[str, Any],
        words: str,
        limit: int,
    ) -> dict[str, Any]:
        ctx = self._context(call_tool, fs.known(session))
        try:
            options = await source_options(widget_target(prop), ctx, search=words.strip() or None)
        except Exception as exc:  # the source is an add-on; its failure is said, not raised
            logger.warning("Form-fill: a field's options could not be fetched", field=spec.name, error=str(exc))
            return {"error": f"The options of {spec.title} could not be fetched: {exc}"}
        if options is None:
            return {
                "error": f"The options of {spec.title} depend on another field of this page: submit the page with that first."
            }
        listed = list(options)[:limit]
        fs.found(session, spec.name, {option.value: option.label for option in listed})
        return {"field": spec.name, "options": _options(listed), "truncated": len(options) > limit}

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
        restarted = await seen(verdict, self._context(call_tool, _values_of(kept)))
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

    # --- the summary, and the start ----------------------------------------------------------------

    @staticmethod
    def _create_args(session: FormFillSession) -> dict[str, Any]:
        """The ``create_workflow`` call: what the person approves is exactly what is sent."""
        return {WORKFLOW_KEY_PARAM: session.workflow_key, JSON_DATA_PARAM: session.page_inputs}

    def _outcome(self, session: FormFillSession, status: Any, **data: Any) -> FormReply:
        """Every value to be submitted, with its label, and the defaults that apply: what the person approves."""
        given = fs.given(session)
        return FormReply(
            workflow_key=session.workflow_key,
            status=status,
            values=given,
            labels=labels(session, given),
            defaults=defaults(session),
            summary=summary_tables(session),
            **data,
        )

    def summary(self, session: FormFillSession) -> Reply:
        """The start to approve: every value as it will be sent, and the create call for the person to approve.

        A form that ends in the workflow's own summary page (core's summary form) has its tables on the
        reply as well: what the workflow's author wants confirmed, as core made it of these very values.
        """
        data = self._outcome(session, "confirming")
        # The call is shown as it will be made; ``labels`` only says what its ids stand for.
        args = {**self._create_args(session), **({"labels": data.labels} if data.labels else {})}
        hint = f"Start workflow `{session.workflow_key}` with these values?"
        return as_reply(data, approval=Approval(hint=hint, tool_name=CREATE_WORKFLOW_TOOL, args=args))

    def prepared(self, session: FormFillSession) -> Reply:
        """Every page validated, nothing started: this caller cannot approve a start. The form closes."""
        data = self._outcome(session, "prepared", reason=CANNOT_APPROVE)
        fs.close(session)
        return as_reply(data)

    async def start(self, session: FormFillSession, call_tool: CallTool) -> Reply:
        """Start the workflow with exactly the validated pages. A start is never sent twice for one approval.

        The form reopens only on core's own verdict on the pages: a refusal is checked against the form
        tool, and the page it rejects is current again, with the verdict, for the model to fix with the
        person. After any other failure whether core started the process is unknown, and after an answer
        the form is done whatever the answer looks like — so in both cases it closes.
        """
        try:
            result = await call_tool(CREATE_WORKFLOW_TOOL, self._create_args(session))
        except Exception as exc:
            if isinstance(exc, ModelRetry):  # core refused: if that is its verdict on a page, that page is open again
                verdict = await page_of(call_tool, session.workflow_key, session.page_inputs)
                if verdict.status == "rejected":
                    return self._refused(session, verdict)
            logger.exception("Form-fill: create_workflow failed", workflow=session.workflow_key)
            fs.close(session)
            return reply_of(session, "failed", reason=str(exc) or type(exc).__name__)
        fs.close(session)
        try:
            process_id = str(ProcessIdSchema.model_validate(result).id)
        except ValidationError:  # core answered, so it started; what it answered goes back as it came
            logger.error("Form-fill: create_workflow answered without a process id", workflow=session.workflow_key)
            return reply_of(session, "started", reason=str(result))
        return reply_of(session, "started", process_id=process_id)

    def _refused(self, session: FormFillSession, verdict: WorkflowFormPage) -> Reply:
        """Core refused the start on a page: that page is current again with core's verdict; the person reads it."""
        index = min(verdict.page, len(session.page_inputs) - 1) if session.page_inputs else 0
        rejected = session.page_inputs[index] if index < len(session.page_inputs) else {}
        page = session.pages[index] if index < len(session.pages) else verdict  # as seen, with its options
        fs.restarted(session, index, page)
        fs.rejected(session, rejected, verdict.errors)
        data = FormReply(
            workflow_key=session.workflow_key,
            status="gathering",
            page=index,
            title=page.title,
            rejected=list(verdict.errors),
            values=fs.given(session),
        )
        return as_reply(data)

    # --- the person's decision ---------------------------------------------------------------------

    async def handle(self, state: SearchState, call_tool: CallTool) -> Reply | None:
        """What the person's response to the pending approval does; None when the turn is the model's.

        The response is one event of the machine (``session.event_of``) and the transition says what the
        turn does: the start, a cancel, the approval shown again — each a reply in place of the model — or
        the form reopened for the model to act on what the person said.
        """
        session = state.form_fill
        if session is None or not session.pending:
            return None  # no approval is out: there is nothing to decide, and nothing is started
        event = event_of(state.form_input, session.unseen)
        match transition(fs.FormPhase(session.status), event):
            case Action.NOTHING:
                return None
            case Action.START:
                reply = await self.start(session, call_tool)
            case Action.CANCEL:
                fs.close(session)
                reply = reply_of(session, "cancelled")
            case Action.SUMMARY:
                reply = self.summary(session)
            case Action.REOPEN:
                fs.reopened(session, getattr(event, "reason", None))
                return None
        state.form_fill = None if session.status == "done" else session
        return reply


def _strangers(
    session: FormFillSession, specs: Mapping[str, FieldSpec], given: Mapping[str, Any]
) -> list[FormFieldError]:
    """A searched field's values that no search returned and the page does not list: refused here, not sent."""
    problems: list[FormFieldError] = []
    for name, value in given.items():
        allowed = accepted_values(session, specs[name])
        if allowed is None:
            continue
        items = value if isinstance(value, list) else [value]
        if strangers := [item for item in items if str(item) not in allowed]:
            message = (
                f"{strangers!r} is not among the options of {specs[name].title}: find the value with "
                "search_options and give one it returned."
            )
            problems.append(FormFieldError(loc=[name], msg=message, type="value_error"))
    return problems


def _rejected(page: WorkflowFormPage, errors: Sequence[FormFieldError], ignored: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "rejected",
        "page": page.page,
        "errors": [{"field": ".".join(str(part) for part in error.loc), "message": error.msg} for error in errors],
        "note": "Fix what is named from what the person said, or ask them.",
    }
    return {**result, "ignored": list(ignored)} if ignored else result


def _options(options: Sequence[Option]) -> list[dict[str, Any]]:
    return [{"value": option.value, "label": option.label} for option in options]


__all__ = ["CANNOT_APPROVE", "COMPLETE_NOTE", "MAX_FOUND", "CallTool", "FormFillSkill", "as_reply", "reply_of"]
