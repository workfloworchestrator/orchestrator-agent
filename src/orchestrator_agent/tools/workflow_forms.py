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

"""Workflow form-fill — direct-dispatch handler.

Routing is fully deterministic: the form submission's ``action`` field
(set by the surface adapter via ``state.adapter_metadata["form_submission"]``)
combined with the current ``state.form_fill`` session picks one of four
implementations. Only the *start* path needs an LLM, and only to extract
the workflow_key from the user's free-text — a single structured-output
call. There is no LLM for routing and none for composing a post-tool reply.

This replaces a pydantic-ai Agent with prompted tool selection, which hit
OpenAI's "empty assistant content" 400 when we instructed the LLM to stay
silent after emitting an artifact. The artifact is the message.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturn

from orchestrator_agent.artifacts import (
    ConfirmRequestArtifact,
    RenderFormArtifact,
)
from orchestrator_agent.memory import ToolStep
from orchestrator_agent.settings import agent_settings
from orchestrator_agent.state import FormFillSession, SearchState
from orchestrator_agent.tools.form_prefill import prefill_form_fields
from orchestrator_agent.tools.wfo_mcp_client import create_workflow, get_workflow_form

logger = structlog.get_logger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────


def _form_id(state: SearchState, page_index: int) -> str:
    base = str(state.run_id) if state.run_id else "form"
    return f"{base}-page-{page_index}"


def _consume_form_submission(state: SearchState) -> None:
    """Drop ``form_submission`` from ``adapter_metadata`` once consumed."""
    if state.adapter_metadata is None:
        return
    state.adapter_metadata = {k: v for k, v in state.adapter_metadata.items() if k != "form_submission"} or None


def _build_form_artifact(
    *,
    schema: dict[str, Any],
    form_id: str,
    workflow_key: str,
    page_index: int,
    prefill: dict[str, Any] | None,
) -> RenderFormArtifact:
    title = schema.get("title") or workflow_key
    if title == "unknown":
        title = workflow_key
    return RenderFormArtifact(
        description=f"Form page {page_index} for {workflow_key}",
        form_id=form_id,
        title=title,
        form_schema=schema,
        prefill=prefill or None,
    )


def _summarise(page_inputs: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for page in page_inputs:
        for k, v in page.items():
            if not k.startswith("__"):
                merged[k] = v
    return merged


# ── workflow_key extraction (the only LLM call in this skill) ─────────────


class _WorkflowKeyOutput(BaseModel):
    """Extracted workflow identifier from the user's free-text request."""

    workflow_key: str = Field(
        description=(
            "The snake_case identifier of the workflow the user wants to start "
            "(e.g. ``task_validate_products``, ``create_core_link``, "
            "``modify_subscription``). Take it verbatim if present, otherwise "
            "derive the most likely match from the natural-language phrasing."
        )
    )


async def _extract_workflow_key(user_input: str) -> str:
    """One structured-output LLM call. Returns the workflow_key string.

    No tool calls, no follow-up text — the agent's output type is
    :class:`_WorkflowKeyOutput`, so the LLM responds via a synthetic
    output tool and we read ``result.output.workflow_key``.
    """
    instructions = (
        "Extract the snake_case workflow_key the user wants to start. "
        "Examples of valid keys: 'task_validate_products', 'create_core_link', "
        "'modify_subscription'. If the user already used a snake_case form, "
        "return it verbatim; otherwise derive the most likely match from the "
        "user's natural-language phrasing."
    )
    agent: Agent[None, _WorkflowKeyOutput] = Agent(
        model=agent_settings.create_model(),
        output_type=_WorkflowKeyOutput,
        instructions=instructions,
    )
    result = await agent.run(user_input or "")
    return result.output.workflow_key


# ── tool implementations (private; called by the dispatch handler) ────────


async def _start_form(state: SearchState, workflow_key: str) -> ToolReturn:
    page = await get_workflow_form(workflow_key, page_inputs=[])
    schema = page.get("schema")
    complete = bool(page.get("complete"))
    page_index = int(page.get("page", 0))

    state.form_fill = FormFillSession(
        workflow_key=workflow_key,
        page_inputs=[],
        current_page_schema=schema,
        current_form_id=_form_id(state, page_index),
        state="summary" if complete else "gathering",
    )

    state.memory.record_tool_step(
        ToolStep(
            step_type="start_workflow_form",
            description=f"Started form for {workflow_key}",
            context={"workflow_key": workflow_key, "page": page_index, "complete": complete},
        )
    )

    if complete or schema is None:
        return ToolReturn(
            return_value=f"Workflow {workflow_key} has no input pages; awaiting confirmation.",
            metadata=ConfirmRequestArtifact(
                description=f"Confirm creation of {workflow_key}",
                request_id=f"{state.run_id}-confirm",
                title=f"Create {workflow_key}",
                summary={"workflow_key": workflow_key, "page_inputs": []},
            ),
        )

    prefill = await prefill_form_fields(state.user_input, schema, previous_pages=[])
    return ToolReturn(
        return_value=f"Form page {page_index} for {workflow_key} ready{' (pre-filled)' if prefill else ''}.",
        metadata=_build_form_artifact(
            schema=schema,
            form_id=state.form_fill.current_form_id or _form_id(state, page_index),
            workflow_key=workflow_key,
            page_index=page_index,
            prefill=prefill,
        ),
    )


async def _submit_page(state: SearchState) -> ToolReturn:
    if state.form_fill is None:
        raise RuntimeError("No active form-fill session.")

    submission = (state.adapter_metadata or {}).get("form_submission") or {}
    values = dict(submission.get("values") or {})

    session = state.form_fill
    session.page_inputs.append(values)
    _consume_form_submission(state)

    page = await get_workflow_form(session.workflow_key, page_inputs=session.page_inputs)
    schema = page.get("schema")
    complete = bool(page.get("complete"))
    page_index = int(page.get("page", len(session.page_inputs)))

    state.memory.record_tool_step(
        ToolStep(
            step_type="submit_workflow_page",
            description=f"Submitted page {page_index - 1} for {session.workflow_key}",
            context={"workflow_key": session.workflow_key, "next_page": page_index, "complete": complete},
        )
    )

    if complete or schema is None:
        session.state = "summary"
        session.current_page_schema = None
        session.current_form_id = None
        return ToolReturn(
            return_value=f"All pages submitted for {session.workflow_key}; awaiting confirmation.",
            metadata=ConfirmRequestArtifact(
                description=f"Confirm creation of {session.workflow_key}",
                request_id=f"{state.run_id}-confirm",
                title=f"Create {session.workflow_key}",
                summary=_summarise(session.page_inputs),
            ),
        )

    session.current_page_schema = schema
    session.current_form_id = _form_id(state, page_index)
    prefill = await prefill_form_fields(state.user_input, schema, previous_pages=session.page_inputs)
    return ToolReturn(
        return_value=f"Form page {page_index} for {session.workflow_key} ready{' (pre-filled)' if prefill else ''}.",
        metadata=_build_form_artifact(
            schema=schema,
            form_id=session.current_form_id,
            workflow_key=session.workflow_key,
            page_index=page_index,
            prefill=prefill,
        ),
    )


async def _confirm(state: SearchState) -> ToolReturn:
    if state.form_fill is None or state.form_fill.state != "summary":
        raise RuntimeError("No form-fill session is awaiting confirmation.")

    session = state.form_fill
    result = await create_workflow(session.workflow_key, session.page_inputs)
    process_id = result.get("id") or result.get("process_id")

    state.memory.record_tool_step(
        ToolStep(
            step_type="create_workflow",
            description=f"Started workflow {session.workflow_key}",
            context={"workflow_key": session.workflow_key, "process_id": str(process_id)},
        )
    )

    state.form_fill = None
    _consume_form_submission(state)

    return ToolReturn(
        return_value=f"Workflow {session.workflow_key} started, process_id={process_id}",
        metadata=None,
    )


async def _cancel(state: SearchState) -> ToolReturn:
    workflow_key = state.form_fill.workflow_key if state.form_fill else None
    state.form_fill = None
    _consume_form_submission(state)

    state.memory.record_tool_step(
        ToolStep(
            step_type="cancel_workflow_form",
            description=f"Cancelled form for {workflow_key}" if workflow_key else "Cancelled form (no active session)",
            context={"workflow_key": workflow_key},
        )
    )

    return ToolReturn(
        return_value=f"Cancelled {workflow_key}." if workflow_key else "No active form to cancel.",
        metadata=None,
    )


# ── the dispatch handler ──────────────────────────────────────────────────


async def workflow_form_fill_handler(state: SearchState, reasoning: str) -> ToolReturn:
    """Direct-dispatch entry point for the WORKFLOW_FORM_FILL skill.

    Routing tree:

    1. ``form_submission.action == "cancel"`` → :func:`_cancel`
    2. ``form_submission.action == "confirm"`` → :func:`_confirm`
    3. active session AND ``action == "submit"`` → :func:`_submit_page`
    4. no session → :func:`_start_form` (LLM extracts workflow_key from
       ``state.user_input`` via one structured-output call)
    5. fallback (active session, no clear action) → :func:`_submit_page`

    ``reasoning`` is the planner's task reasoning; we don't read it here
    but it's part of the dispatch-handler contract so form-fill and
    delegation skills share one shape.
    """
    del reasoning  # unused — kept for handler-signature parity

    submission = (state.adapter_metadata or {}).get("form_submission") or {}
    action = submission.get("action")

    if action == "cancel":
        return await _cancel(state)

    if action == "confirm":
        return await _confirm(state)

    if state.form_fill is not None and action == "submit":
        return await _submit_page(state)

    if state.form_fill is None:
        workflow_key = await _extract_workflow_key(state.user_input or "")
        return await _start_form(state, workflow_key)

    # Active session, no clear action. The surface adapter normally sets one,
    # but the planner can re-fire WORKFLOW_FORM_FILL on a follow-up turn —
    # treat as submit so the loop progresses.
    return await _submit_page(state)
