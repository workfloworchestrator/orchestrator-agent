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

"""Workflow form-fill toolset.

Drives orchestrator-core's multi-page workflow form via MCP. State for the
form lives on ``SearchState.form_fill`` so the session resumes across A2A
turns. Surface adapters attach button-click submissions to
``SearchState.adapter_metadata["form_submission"]`` (an opaque envelope —
the adapter does not need to know about form-fill specifically).
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic_ai import RunContext
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.messages import ToolReturn
from pydantic_ai.toolsets import FunctionToolset

from orchestrator_agent.artifacts import (
    ConfirmRequestArtifact,
    FormField,
    FormFieldType,
    RenderFormArtifact,
)
from orchestrator_agent.memory import ToolStep
from orchestrator_agent.state import FormFillSession, SearchState
from orchestrator_agent.tools.form_prefill import prefill_form_fields
from orchestrator_agent.tools.wfo_mcp_client import create_workflow, get_workflow_form

logger = structlog.get_logger(__name__)

workflow_forms_toolset: FunctionToolset[StateDeps[SearchState]] = FunctionToolset(max_retries=2)


def _form_id(state: SearchState, page_index: int) -> str:
    base = str(state.run_id) if state.run_id else "form"
    return f"{base}-page-{page_index}"


def _consume_form_submission(state: SearchState) -> None:
    """Drop ``form_submission`` from ``adapter_metadata`` once consumed.

    Prevents subsequent tool calls in the same turn from reprocessing the
    submission as if it were fresh.
    """
    if state.adapter_metadata is None:
        return
    state.adapter_metadata = {k: v for k, v in state.adapter_metadata.items() if k != "form_submission"} or None


def _infer_field_type(prop: dict[str, Any]) -> FormFieldType:
    if "enum" in prop:
        return "select"
    fmt = prop.get("format")
    if fmt == "date":
        return "date"
    if fmt == "uuid":
        return "uuid"
    jt = prop.get("type")
    if jt == "string":
        return "text"
    if jt == "integer":
        return "int"
    if jt == "number":
        return "float"
    if jt == "boolean":
        return "bool"
    if jt == "array":
        items = prop.get("items") or {}
        if "enum" in items:
            return "multiselect"
    return "text"


def _extract_options(prop: dict[str, Any]) -> list[dict[str, str]] | None:
    if "enum" in prop:
        return [{"value": str(v), "label": str(v)} for v in prop["enum"]]
    items = prop.get("items") or {}
    if isinstance(items, dict) and "enum" in items:
        return [{"value": str(v), "label": str(v)} for v in items["enum"]]
    return None


def json_schema_to_render_form(
    schema: dict[str, Any],
    form_id: str,
    workflow_key: str,
    page_index: int,
    prefill: dict[str, Any] | None = None,
) -> RenderFormArtifact:
    """Map a (pydantic-forms produced) JSON Schema page to a RenderFormArtifact.

    ``prefill`` overrides the schema-level default per field. Used to seed
    forms with LLM-extracted values from the user's natural-language input
    (see :mod:`orchestrator_agent.tools.form_prefill`).

    Complex JSON-Schema shapes (``$ref`` / ``allOf`` / ``oneOf``) are not
    resolved here — they fall through to a plain text field. Add resolution
    when a real workflow needs it.
    """
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    prefill = prefill or {}
    fields: list[FormField] = []
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        value = prefill[name] if name in prefill else prop.get("default")
        fields.append(
            FormField(
                name=name,
                label=prop.get("title") or name,
                type=_infer_field_type(prop),
                required=name in required,
                value=value,
                options=_extract_options(prop),
                help_text=prop.get("description"),
            )
        )
    title = schema.get("title") or workflow_key
    if title == "unknown":
        title = workflow_key
    return RenderFormArtifact(
        description=f"Form page {page_index} for {workflow_key}",
        form_id=form_id,
        title=title,
        fields=fields,
    )


def _summarise(page_inputs: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for page in page_inputs:
        for k, v in page.items():
            if not k.startswith("__"):
                merged[k] = v
    return merged


@workflow_forms_toolset.tool
async def start_workflow_form(
    ctx: RunContext[StateDeps[SearchState]],
    workflow_key: str,
) -> ToolReturn:
    """Start a new workflow form-fill session.

    Call this when the user expresses intent to create/start a workflow but no
    ``form_fill`` session is active yet. Fetches page 0 and emits the form for
    rendering. Use ``submit_workflow_page`` for subsequent pages.
    """
    page = await get_workflow_form(workflow_key, page_inputs=[])

    schema = page.get("schema")
    complete = bool(page.get("complete"))
    page_index = int(page.get("page", 0))

    state = ctx.deps.state
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
        metadata=json_schema_to_render_form(
            schema=schema,
            form_id=state.form_fill.current_form_id or _form_id(state, page_index),
            workflow_key=workflow_key,
            page_index=page_index,
            prefill=prefill,
        ),
    )


@workflow_forms_toolset.tool
async def submit_workflow_page(
    ctx: RunContext[StateDeps[SearchState]],
    values: dict[str, Any] | None = None,
) -> ToolReturn:
    """Submit values for the current form page and advance.

    Reads submitted values from the ``values`` argument when provided,
    otherwise from ``state.adapter_metadata["form_submission"]["values"]``
    (set by the surface adapter). Returns the next page as a
    RenderFormArtifact, or a ConfirmRequestArtifact when the form is complete.
    """
    state = ctx.deps.state
    if state.form_fill is None:
        raise RuntimeError("No active form-fill session. Call start_workflow_form first.")

    if values is None:
        submission = (state.adapter_metadata or {}).get("form_submission") or {}
        values = dict(submission.get("values") or {})

    session = state.form_fill
    session.page_inputs.append(values or {})
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
        metadata=json_schema_to_render_form(
            schema=schema,
            form_id=session.current_form_id,
            workflow_key=session.workflow_key,
            page_index=page_index,
            prefill=prefill,
        ),
    )


@workflow_forms_toolset.tool
async def confirm_and_create_workflow(
    ctx: RunContext[StateDeps[SearchState]],
) -> ToolReturn:
    """Submit accumulated pages to orchestrator-core to start the workflow."""
    state = ctx.deps.state
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


@workflow_forms_toolset.tool
async def cancel_workflow_form(
    ctx: RunContext[StateDeps[SearchState]],
) -> ToolReturn:
    """Cancel the active form-fill session and clear state."""
    state = ctx.deps.state
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
