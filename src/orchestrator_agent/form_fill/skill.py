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

"""The workflow form-fill skill: the state machine that fills core's form with a human in the loop.

A form is filled through a human-in-the-loop transport only (kagent's extension over A2A, or LibreChat's
ask-user tool over chat completions): every stop of the skill is a pause that shows the human the page's
questions with their options, or the start to approve, and only their structured response continues the
form. No chat text is read for a value or a decision, by this code or by a model. *Which* workflow a request
asks for is the model's call: it hands off with the ``start_workflow_form`` tool (``handoff``), and ``open``
walks the first pages in the same turn.

The machine is ``session``: its phases (``opening``, ``gathering``, ``confirming``, ``done``), the event a
turn's input is, and the transition table that says what the turn does. This class carries the actions out:
the walk over core's pages (``walk``), the summary with the start to approve, and the start itself. Core's
own validation is the only validation; what a page's schema alone decides is said by the walk before
anything is submitted.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import structlog
from orchestrator.core.schemas.process import ProcessIdSchema
from orchestrator.core.schemas.workflow import WorkflowSchema
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import ModelRetry

from orchestrator_agent.form_fill import session as fs
from orchestrator_agent.form_fill.interpret import Chooser, Interpreter
from orchestrator_agent.form_fill.model import SUMMARY, askable
from orchestrator_agent.form_fill.session import Action, Answered, Event, FormPhase, Opened, event_of, transition
from orchestrator_agent.form_fill.walk import (
    CallTool,
    Completed,
    Walk,
    as_reply,
    as_walked,
    labels,
    page_of,
    page_stop,
    reply_of,
    walk,
)
from orchestrator_agent.form_fill.widgets import FieldWidget, GraphQL
from orchestrator_agent.state import SUBSCRIPTION_ID, Approval, FormFillSession, FormReply, Reply, SearchState
from orchestrator_agent.tool_names import CREATE_WORKFLOW_TOOL, JSON_DATA_PARAM, LIST_WORKFLOWS_TOOL, WORKFLOW_KEY_PARAM

logger = structlog.get_logger(__name__)

# Core's tool results are validated with core's own response models (the agent depends on orchestrator-core),
# and its tool arguments are built from core's request models — nothing here spells out a field of either.
_WORKFLOW_LIST: TypeAdapter[list[WorkflowSchema]] = TypeAdapter(list[WorkflowSchema])
_WORKFLOWS_TTL = 300.0  # seconds the workflow catalogue is cached per skill instance


@dataclass
class FormFillSkill:
    interpret: Interpreter | None = None  # a person's answers core rejected -> the values the fields expect
    widgets: Sequence[FieldWidget] = ()  # the formats whose options core leaves to a frontend (``form_fill.widgets``)
    graphql: GraphQL | None = None  # core's GraphQL API, for the widgets that read it
    choose: Chooser | None = None  # words a person typed for a widget field -> the options they fit
    _catalogue: dict[str, str] | None = dc_field(default=None, init=False, repr=False)
    _catalogue_at: float = dc_field(default=0.0, init=False, repr=False)

    async def handle(self, state: SearchState, call_tool: CallTool) -> Reply | None:
        """Continue the open form with the human's response (``state.form_input``); None when the turn is not the form's.

        Without a human-in-the-loop transport no stop can be shown, so the form is dropped. Otherwise the
        response is one event of the machine (``session.event_of``) and the transition table says what the
        turn does.
        """
        session = state.form_fill
        if session is None:
            return None  # routing is the model's, via the handoff tool
        if not state.hitl:
            state.form_fill = None
            return None
        return await self._dispatch(state, session, event_of(state.form_input, session.unseen), call_tool)

    async def open(self, state: SearchState, call_tool: CallTool) -> Reply | None:
        """Walk the form the model handed off (``start_workflow_form``); None if there is none.

        The subscription the model passed on fills the form's first page, and core's verdict on it is the
        stop like any other. So is core's refusal of the workflow itself (a key the handoff tool checked
        but core no longer knows): the form closes with it.
        """
        session = state.form_fill
        if session is None:
            return None
        return await self._dispatch(state, session, Opened(), call_tool)

    # --- one turn ------------------------------------------------------------------------------------

    async def _dispatch(
        self, state: SearchState, session: FormFillSession, event: Event, call_tool: CallTool
    ) -> Reply | None:
        """One turn: the table's action for the session's phase and the event, carried out."""
        match transition(FormPhase(session.status), event):
            case Action.NOTHING:
                return None
            case Action.ABANDON:
                state.form_fill = None
                return None
            case Action.CANCEL:
                fs.close(session)
                reply = reply_of(session, "cancelled")
            case Action.START:
                reply = await self._start(session, call_tool)
            case _:
                match event:
                    case Opened():
                        logger.info(
                            "Form-fill opened by the model",
                            workflow=session.workflow_key,
                            subscription_id=session.values.get(SUBSCRIPTION_ID),  # passed with the handoff
                        )
                        fs.opened(session)
                    case Answered(values=values):
                        fs.answer(session, values)
                outcome = await walk(self._walk(session, call_tool))
                reply = self._summary(session) if isinstance(outcome, Completed) else outcome
        return self._finish(state, session, reply)

    def _walk(self, session: FormFillSession, call_tool: CallTool) -> Walk:
        """A walk over this session with the skill's collaborators."""
        return Walk(session, call_tool, self.widgets, self.graphql, self.choose, self.interpret)

    @staticmethod
    def _finish(state: SearchState, session: FormFillSession, reply: Reply) -> Reply:
        state.form_fill = None if session.status == "done" else session
        return reply

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

    # --- the summary, and the start ----------------------------------------------------------------

    @staticmethod
    def _create_args(session: FormFillSession) -> dict[str, Any]:
        """The ``create_workflow`` call: what the human approves is exactly what is sent."""
        return {WORKFLOW_KEY_PARAM: session.workflow_key, JSON_DATA_PARAM: session.page_inputs}

    def _summary(self, session: FormFillSession) -> Reply:
        """Every value to be submitted and the defaults that apply, with the create call for the human to approve.

        A form that ends in the workflow's own summary page (core's summary form) has its tables on the
        reply as well: what the workflow's author wants confirmed, as core made it of these very values.
        """
        given = fs.given(session)
        shown = labels(session, given)
        fields = fs.fields(session)
        defaults = {
            name: field.default
            for name, field in fields.items()
            if askable(field) and name not in given and not field.required
        }
        tables: list[Any] = [f.data for f in fields.values() if f.format == SUMMARY and isinstance(f.data, dict)]
        data = FormReply(
            workflow_key=session.workflow_key,
            status="confirming",
            values=given,
            labels=shown,
            defaults=defaults,
            summary=[table for table in tables if table.get("labels")],
        )
        # The call is shown as it will be made; ``labels`` only says what its ids stand for.
        args = {**self._create_args(session), **({"labels": shown} if shown else {})}
        hint = f"Start workflow `{session.workflow_key}` with these values?"
        return as_reply(data, approval=Approval(hint=hint, tool_name=CREATE_WORKFLOW_TOOL, args=args))

    async def _start(self, session: FormFillSession, call_tool: CallTool) -> Reply:
        """Start the workflow with exactly the validated pages. A start is never sent twice for one confirmation.

        The form reopens only on core's own verdict on the pages: a refusal is checked against the form
        tool, and a page it rejects is asked again, so nothing was started. After any other failure whether
        core started the process is unknown, and after an answer the form is done whatever the answer looks
        like — so in both cases it closes.
        """
        try:
            result = await call_tool(CREATE_WORKFLOW_TOOL, self._create_args(session))
        except Exception as exc:
            if isinstance(exc, ModelRetry):  # core refused: if that is its verdict on a page, the page is asked again
                ctx = self._walk(session, call_tool)
                verdict = await page_of(ctx, session.page_inputs)
                if verdict.status == "rejected":
                    verdict = as_walked(session, verdict)
                    return page_stop(ctx, verdict, session.page_inputs[: verdict.page + 1], verdict.errors)
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


__all__ = ["CallTool", "FormFillSkill"]
