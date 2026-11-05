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

"""A form page as core describes it, read into the fields the skill works from.

Core's ``get_workflow_form`` describes a page as one JSON Schema: every reference inlined, ``readOnly`` on a
field that is only shown, a choice's values in ``enum`` with their labels in ``options``, consent as
``format: "accept"``. ``fields_of`` reads that schema into one ``FieldSpec`` per field, which every stop of
the skill works from. ``form_schema`` is the walked pages' askable fields as one schema again, the page as
core defined it: what the interpreter hands its model as the output type. No pydantic model is rebuilt from
the schema anywhere; core's validation is the only validation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import chain
from typing import Any

from orchestrator.core.schemas.mcp_tools import WorkflowFormPage
from pydantic import BaseModel, ConfigDict, Field

# pydantic-forms' ``format`` markers the skill acts on.
ACCEPT = "accept"  # consent, given once per page by picking its one option
SUMMARY = "summary"  # a table of the workflow's own recap: shown, never filled
ACCEPTED = "ACCEPTED"  # the one value of an accept field that core's validation passes
HINTS = ("uniforms", "extraProperties")  # pydantic-forms' widget hints: a widget reads them, a model never does
WIDGET_MARK = "x-widget"  # what the agent's own widget wrote on a field (``widgets.enrich``): its id, size, step

_TYPES = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean"}
_TYPES |= {"array": "list", "object": "object"}


class Option(BaseModel):
    """One value a field is limited to, with the label the form shows for it."""

    model_config = ConfigDict(frozen=True)

    value: Any
    label: str


class FieldSpec(BaseModel):
    """One field of a page as the skill reads it off the page schema."""

    model_config = ConfigDict(frozen=True)

    name: str
    title: str
    description: str | None = None
    kind: str  # string, integer, number, boolean, list, object, or any
    format: str | None = None
    required: bool = False
    default: Any = None
    nullable: bool = False
    options: list[Option] | None = None  # the values it is limited to, in the form's order
    item: FieldSpec | None = None  # of a list: what one item is
    constraints: dict[str, Any] = Field(default_factory=dict)  # of a list: ``min_length`` / ``max_length``
    fields: list[FieldSpec] | None = None  # of a nested object: its own fields
    read_only: bool = False  # shown, never submitted
    data: Any = None  # of a summary: its table
    widget: dict[str, Any] | None = None  # what the agent's widget wrote on it, if one did (``widgets.enrich``)


# --- the page schema -> its fields ---------------------------------------------------------------------------


def fields_of(page: WorkflowFormPage) -> list[FieldSpec]:
    """The fields of a page as core described it; [] for a page without a schema (the form complete)."""
    if page.schema_ is None:
        return []
    return list(_fields_cached(json.dumps(page.schema_, default=str)))  # the form's order is the key's


def fields_in(schema: Mapping[str, Any]) -> list[FieldSpec]:
    """The fields of one object schema, in the form's order."""
    required = set(schema.get("required") or [])
    return [
        _spec(name, prop, required=name in required)
        for name, prop in (schema.get("properties") or {}).items()
        if isinstance(prop, Mapping)
    ]


@lru_cache(maxsize=256)
def _fields_cached(schema_json: str) -> tuple[FieldSpec, ...]:
    return tuple(fields_in(json.loads(schema_json)))


def _spec(name: str, prop: Mapping[str, Any], *, required: bool) -> FieldSpec:
    shape, nullable = _shape(prop)
    kind = _TYPES.get(str(shape.get("type")), "any")
    items = shape.get("items")
    hints = shape.get("uniforms") or shape.get("extraProperties") or {}
    sizes = (("min_length", shape.get("minItems")), ("max_length", shape.get("maxItems")))
    return FieldSpec(
        name=name,
        title=str(shape.get("title") or name.replace("_", " ").title()),
        description=shape.get("description"),
        kind=kind,
        format=shape.get("format"),
        required=required,
        default=shape.get("default"),
        nullable=nullable,
        options=_options(shape),
        item=_spec("", items, required=False) if kind == "list" and isinstance(items, Mapping) else None,
        constraints={key: value for key, value in sizes if value is not None},
        fields=fields_in(shape) if kind == "object" and shape.get("properties") else None,
        read_only=bool(shape.get("readOnly")),
        data=hints.get("data") if shape.get("format") == SUMMARY else None,
        widget=shape[WIDGET_MARK] if isinstance(shape.get(WIDGET_MARK), dict) else None,
    )


def shape_of(prop: Mapping[str, Any]) -> dict[str, Any]:
    """The field's shape with a nullable field's ``anyOf`` unwrapped: what a widget is matched on."""
    return _shape(prop)[0]


def _shape(prop: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """The field's shape with a nullable field's ``anyOf`` unwrapped, and whether it allows null.

    Core inlines every reference, so a field's shape is at top level except for the ``anyOf`` of a nullable
    field, whose null variant is a flag rather than a shape. The field's own keys win over what they wrap.
    """
    shape = dict(prop)
    variants = shape.pop("anyOf", None) or []
    shapes = [variant for variant in variants if variant.get("type") != "null"]
    if len(shapes) == 1:
        shape = {**shapes[0], **shape}
    elif shapes:
        shape["anyOf"] = shapes  # several real shapes: the field is read as free, and core judges the value
    return shape, len(shapes) < len(variants)


def _options(shape: Mapping[str, Any]) -> list[Option] | None:
    """The values a field is limited to: consent's one value, its ``const``, or its ``enum`` with the labels.

    An accept field lists ``ACCEPTED`` and ``INCOMPLETE`` for the checkbox that renders it; only ``ACCEPTED``
    passes core's validation, so that is its one option.
    """
    if shape.get("format") == ACCEPT:
        return [Option(value=ACCEPTED, label=ACCEPTED)]
    if "const" in shape:
        return [Option(value=shape["const"], label=str(shape["const"]))]
    if not isinstance(shape.get("enum"), list):
        return None
    labels = shape.get("options") or {}
    return [Option(value=value, label=str(labels.get(str(value), value))) for value in shape["enum"]]


# --- what a field says ----------------------------------------------------------------------------------------


def askable(field: FieldSpec) -> bool:
    """A field a caller may send a value for: not one that is only shown."""
    return not field.read_only


def options_of(field: FieldSpec) -> list[Option] | None:
    """The options a field (or, for a list, its item) is limited to; None when the value is free."""
    if field.options is None and field.item is not None:
        return field.item.options
    return field.options


def label_of(field: FieldSpec, value: Any) -> Any:
    """How the form shows ``value`` (each item of a list), or None when it shows the value itself."""
    shown = {str(o.value): o.label for o in options_of(field) or [] if o.label != str(o.value)}
    if isinstance(value, list):
        return [shown.get(str(item), item) for item in value] if any(str(item) in shown for item in value) else None
    return shown.get(str(value))


# --- the walked pages as one schema ---------------------------------------------------------------------------


def form_schema(pages: Sequence[WorkflowFormPage]) -> dict[str, Any]:
    """The askable fields of the walked pages as one object schema (a later page wins a name).

    What a reading of the form works from: the properties as core sent them, less the fields that are only
    shown and pydantic-forms' widget hints. Nothing is required, since a reading may leave any field out.
    """
    properties = dict(chain.from_iterable(_askable_properties(page) for page in pages))
    return {"type": "object", "properties": properties, "additionalProperties": False}


def _askable_properties(page: WorkflowFormPage) -> list[tuple[str, dict[str, Any]]]:
    properties = (page.schema_ or {}).get("properties") or {}
    return [
        (field.name, {key: value for key, value in properties[field.name].items() if key not in (*HINTS, WIDGET_MARK)})
        for field in fields_of(page)
        if askable(field)
    ]


__all__ = [
    "ACCEPT",
    "ACCEPTED",
    "HINTS",
    "SUMMARY",
    "WIDGET_MARK",
    "FieldSpec",
    "Option",
    "askable",
    "fields_in",
    "fields_of",
    "form_schema",
    "label_of",
    "options_of",
    "shape_of",
]
