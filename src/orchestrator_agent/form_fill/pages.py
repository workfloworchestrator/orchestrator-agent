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

"""A form page as core describes it, read for the model that fills it and for the stops that show it.

Core's form tool answers a page as one JSON Schema (``model``). Here that page becomes the parameters of
the ``submit_page`` tool (``tool_schema``: the fields a model may fill, the frontend's hints dropped, a long
list described as something to search), the one-line description a tool result gives of it
(``describe``), the values code fills without asking anyone (``obvious``), and the labels the approval
shows for the values (``labels``). A short option list is in the schema; a field core left its
``optionsSource`` on has a long one, found through core's options tool.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from orchestrator.core.schemas.mcp_tools import GetWorkflowFormRequest, WorkflowFormPage
from pydantic_forms.utils.pointer import parse_pointer
from pydantic_forms.utils.schema import DEPENDS_ON, OPTIONS_SOURCE

from orchestrator_agent.form_fill.model import (
    ACCEPT,
    ACCEPTED,
    HINTS,
    SUMMARY,
    FieldSpec,
    askable,
    fields_of,
    label_of,
    options_of,
)
from orchestrator_agent.state import FormFillSession, SummaryTable
from orchestrator_agent.tool_names import GET_WORKFLOW_FORM_TOOL


class CallTool(Protocol):
    """Call one of core's MCP tools by name (bound to an open MCP session by the adapter)."""

    async def __call__(self, name: str, args: dict[str, Any]) -> Any: ...


async def page_of(call_tool: CallTool, workflow_key: str, pages: Sequence[Mapping[str, Any]]) -> WorkflowFormPage:
    """Core's verdict on ``pages``: the next page, the form complete, or the page it rejected with its errors."""
    request = GetWorkflowFormRequest(
        workflow_key=workflow_key, page_inputs=[dict(page) for page in pages], verdict="result"
    )
    return WorkflowFormPage.model_validate(await call_tool(GET_WORKFLOW_FORM_TOOL, request.model_dump()))


# --- what a field is to the model --------------------------------------------------------------------------


def fillable(field: FieldSpec) -> bool:
    """A field the model may fill: not one that is only shown, and not consent, which is code's (``obvious``)."""
    return askable(field) and field.format != ACCEPT


def source_of(prop: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The options source core left on a property (or a list's items, or a nullable field's branch); None without one."""
    branches = [prop, prop.get("items"), *(prop.get("anyOf") or [])]
    return next((b[OPTIONS_SOURCE] for b in branches if isinstance(b, Mapping) and OPTIONS_SOURCE in b), None)


def searched(prop: Mapping[str, Any]) -> bool:
    """Whether a property is a long list: its options are found with ``search_options``, not read."""
    return source_of(prop) is not None


def obvious(page: WorkflowFormPage) -> dict[str, Any]:
    """The values code fills without asking anyone.

    Consent (an ``accept`` field) is given at the approval of the start, by the person, before anything is
    written; the field is filled so the form can reach that approval. A required choice of the form's own
    with exactly one option is that option (as the one-item list a single-select list wants).
    """
    values: dict[str, Any] = {}
    for field in fields_of(page):
        if not askable(field):
            continue
        if field.format == ACCEPT:
            values[field.name] = ACCEPTED
        elif field.required and len(options := options_of(field) or []) == 1:
            if field.kind != "list" or field.constraints.get("max_length") == 1:
                values[field.name] = [options[0].value] if field.kind == "list" else options[0].value
    return values


# --- the page as the parameters of ``submit_page`` -----------------------------------------------------------


def tool_schema(page: WorkflowFormPage) -> dict[str, Any]:
    """The page as the parameters of ``submit_page``: its fillable fields as core wrote them, less the hints.

    A short option list stays in the schema (``enum`` with ``options`` as the labels), as core or the
    options-source reader wrote it. A long list has no options here: it is a value to find with
    ``search_options``, and its description says so.
    """
    schema = page.schema_ or {}
    specs = {field.name: field for field in fields_of(page)}
    properties: dict[str, Any] = {}
    for name, prop in (schema.get("properties") or {}).items():
        spec = specs.get(name)
        if spec is None or not fillable(spec) or not isinstance(prop, Mapping):
            continue
        properties[name] = _property(spec, prop)
    required = [name for name in schema.get("required") or [] if name in properties]
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _property(spec: FieldSpec, prop: Mapping[str, Any]) -> dict[str, Any]:
    clean = _strip(prop)
    if not searched(prop):
        return clean
    if deps := ", ".join(_dependencies(prop)):
        note = f"Its options depend on {deps}: submit the page with that first, then find a value with search_options."
    else:
        note = "Find the value with search_options and give one it returned."
    return {**clean, "description": " ".join(filter(None, (spec.description, note)))}


def _dependencies(prop: Mapping[str, Any]) -> list[str]:
    """The fields a sourced property's options depend on (``dependsOn`` pointers, by their first segment)."""
    names: list[str] = []
    for pointer in (source_of(prop) or {}).get(DEPENDS_ON, {}).values():
        parsed = parse_pointer(str(pointer))
        if parsed is not None and parsed.segments:
            names.append(parsed.segments[0])
    return names


def _strip(prop: Mapping[str, Any]) -> dict[str, Any]:
    """The property without the frontend's hints, the widget's mark and the source: what a model reads."""
    out: dict[str, Any] = {}
    for key, value in prop.items():
        if key in HINTS or key == OPTIONS_SOURCE:
            continue
        if key == "items" and isinstance(value, Mapping):
            value = _strip(value)
        elif key == "anyOf" and isinstance(value, list):
            value = [_strip(item) if isinstance(item, Mapping) else item for item in value]
        elif key == "properties" and isinstance(value, Mapping):
            value = {name: _strip(item) if isinstance(item, Mapping) else item for name, item in value.items()}
        out[key] = value
    return out


# --- the page as a tool result tells it ------------------------------------------------------------------------


def describe(page: WorkflowFormPage) -> dict[str, Any]:
    """A page as a tool result tells the model about it: its number and title, and each fillable field in one line."""
    properties = (page.schema_ or {}).get("properties") or {}
    fields = {
        field.name: _line(field, properties.get(field.name) or {}) for field in fields_of(page) if fillable(field)
    }
    return {"page": page.page, "title": page.title, "fields": fields}


def _line(field: FieldSpec, prop: Mapping[str, Any]) -> str:
    parts = [field.title, "required" if field.required else "optional"]
    if searched(prop):
        parts.append("find the value with search_options")
    elif options := options_of(field):
        parts.append("one of " + ", ".join(_option(o.value, o.label) for o in options))
    else:
        parts.append(field.kind)
    if field.kind == "list":
        parts.append("a list")
    return "; ".join(parts)


def _option(value: Any, label: str) -> str:
    return str(value) if label == str(value) else f"{value} ({label})"


# --- what the approval shows ------------------------------------------------------------------------------------


def labels(session: FormFillSession, values: Mapping[str, Any]) -> dict[str, Any]:
    """How the form shows each of ``values`` that is one of its field's options (an id says nothing to a person).

    A field's own options label it; a searched value is labelled as core labelled it when it accepted the page.
    """
    from orchestrator_agent.form_fill import session as fs

    fields = fs.fields(session)
    shown = {name: label_of(fields[name], value) for name, value in values.items() if name in fields}
    result = {name: label for name, label in shown.items() if label is not None}
    for page_values, page_labels in zip(session.page_inputs, session.page_labels, strict=False):
        result.update({name: label for name, label in page_labels.items() if page_values.get(name) == values.get(name)})
    return result


def defaults(session: FormFillSession) -> dict[str, Any]:
    """The optional fields nobody gave a value: what the form fills in by itself."""
    from orchestrator_agent.form_fill import session as fs

    given = fs.given(session)
    return {
        name: field.default
        for name, field in fs.fields(session).items()
        if askable(field) and name not in given and not field.required
    }


def summary_tables(session: FormFillSession) -> list[SummaryTable]:
    """The tables of the workflow's own summary page, when its form ends in one (core's summary form)."""
    from orchestrator_agent.form_fill import session as fs

    tables = [
        cast(SummaryTable, f.data)
        for f in fs.fields(session).values()
        if f.format == SUMMARY and isinstance(f.data, dict)
    ]
    return [table for table in tables if table.get("labels")]


__all__ = [
    "CallTool",
    "defaults",
    "describe",
    "fillable",
    "labels",
    "obvious",
    "page_of",
    "searched",
    "summary_tables",
    "tool_schema",
]
