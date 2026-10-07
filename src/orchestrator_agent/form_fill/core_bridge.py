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

"""One pydantic model per form page, built from what core's form tool returns.

The page model is the one artifact everything else works from: the interpreter's output type is its
partial variant, an agent caller gets its JSON schema, and the summary shows values with the labels it
carries.

Half of this module exists only until core's form tool speaks agent (the plan doc's follow-up 5).
``get_workflow_form`` returns the raw pydantic-forms JSON schema, written for a browser form: ``$ref`` /
``allOf`` / nullable ``anyOf`` indirection, widget hints in ``uniforms`` and ``extraProperties``,
display-only ``format`` markers, enum labels in an ``options`` side table, nested ``$defs`` for structured
fields; and a rejected page comes back as pydantic-forms' error body, relayed by fastmcp as its Python
repr inside the tool error text. ``page_model``'s schema reading and ``form_errors`` undo that on the
agent side; when core returns a field spec and errors as data, that half goes and the model is built from
the spec. What a built model says about its fields (``choices``, ``labels``, ...) stays.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from annotated_types import MaxLen, MinLen
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, create_model
from pydantic.fields import FieldInfo

from orchestrator_agent.state import ACCEPT_VALUE, FormError, SummaryTable

# pydantic-forms ``format`` markers the skill interprets (the ``json_schema_extra`` of its field types).
# Those field classes are not imported: ``pydantic_forms.validators`` pulls in a contact-person field that
# needs ``email-validator``, which this agent does not ship.
FORMAT_ACCEPT = "accept"
FORMAT_SUMMARY = "summary"  # a table of the workflow's own recap (``MigrationSummary``): shown, never filled
DISPLAY_ONLY_FORMATS = frozenset({"label", "divider", FORMAT_SUMMARY, "markdown", "callout", "hidden", "subscription"})

# What a page model's fields carry in ``json_schema_extra``: the form's ``format`` marker, and the label
# behind each allowed value when the two differ.
FORMAT = "format"
LABELS = "labels"
WIDGET = "x-widget"  # what a form-fill widget wrote on a field (``widgets.enrich``)


# --- core's page schema -> the page model ----------------------------------------------------------------


def page_model(schema: Mapping[str, Any]) -> type[BaseModel]:
    """The page as a pydantic model: one field per value the caller may send; display-only fields left out.

    Pure and cached per schema, so a page persisted on the session gives the same model every turn. The
    fields keep the form's order.
    """
    return _page_model(json.dumps(schema, default=str))


def form_model(schemas: Sequence[Mapping[str, Any]]) -> type[BaseModel]:
    """The walked pages as one model (a later page wins a name), for readings and lookups that span the form."""
    return _form_model(tuple(json.dumps(schema, default=str) for schema in schemas))


@lru_cache(maxsize=256)
def _page_model(schema_json: str) -> type[BaseModel]:
    schema = json.loads(schema_json)
    defs = schema.get("$defs") or {}
    title = str(schema.get("title") or "Page")
    return _object_model(title, schema, defs)


@lru_cache(maxsize=256)
def _form_model(pages_json: tuple[str, ...]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for page_json in pages_json:
        for name, info in _page_model(page_json).model_fields.items():
            fields[name] = (info.annotation, info)
    return create_model("Form", __config__=ConfigDict(protected_namespaces=()), **fields)


def _object_model(title: str, obj: Mapping[str, Any], defs: Mapping[str, Any]) -> type[BaseModel]:
    required = set(obj.get("required") or [])
    fields: dict[str, Any] = {}
    for name, prop in (obj.get("properties") or {}).items():
        if not isinstance(prop, Mapping):
            continue
        resolved = resolve_property(prop, defs)
        if is_read_only(resolved) or resolved.get(FORMAT) in DISPLAY_ONLY_FORMATS:
            continue
        fields[name] = _field(name, resolved, defs, required=name in required)
    name = re.sub(r"\W", "", title) or "Page"
    return create_model(name, __config__=ConfigDict(title=title, protected_namespaces=()), **fields)


def _field(name: str, prop: Mapping[str, Any], defs: Mapping[str, Any], *, required: bool) -> tuple[Any, FieldInfo]:
    """One model field: the type the value must have, its default, and what the stops need to know about it."""
    annotation = _annotation(prop, defs)
    extra: dict[str, Any] = {**_no_options(prop, defs)}
    if fmt := prop.get(FORMAT):
        extra[FORMAT] = fmt
    if labels := _labels(prop, defs):
        extra[LABELS] = labels
    if mark := prop.get(WIDGET):
        extra[WIDGET] = mark
    constraints: dict[str, Any] = {}
    if prop.get("type") == "array":
        constraints = {"min_length": prop.get("minItems"), "max_length": prop.get("maxItems")}
    default = prop.get("default")
    if not required and default is None:
        annotation = annotation | None
    info = Field(
        ... if required else default,
        title=str(prop.get("title") or name),
        description=prop.get("description"),
        json_schema_extra=extra or None,
        **{k: v for k, v in constraints.items() if v is not None},
    )
    return annotation, info


def _annotation(prop: Mapping[str, Any], defs: Mapping[str, Any]) -> Any:
    """The Python type of the value a field expects: its allowed values, a scalar, a list, or a nested model."""
    if allowed := _allowed(prop):
        return _literal(allowed)
    match prop.get("type"):  # a choice core offers no option for keeps its base type: see ``_no_options``
        case "boolean":
            return bool
        case "integer":
            return int
        case "number":
            return float
        case "string":
            return str
        case "array":
            return list[_annotation(resolve_property(prop.get("items") or {}, defs), defs)]  # type: ignore[misc]
        case "object":
            if prop.get("properties"):
                return _object_model(str(prop.get("title") or "Item"), prop, defs)
            return dict[str, Any]
    return Any


def _allowed(prop: Mapping[str, Any]) -> tuple[Any, ...]:
    """The values a field is limited to — consent, its one ``const``, its ``enum`` — or () when it is not limited."""
    if prop.get(FORMAT) == FORMAT_ACCEPT:
        return (ACCEPT_VALUE,)
    if "const" in prop:
        return (prop["const"],)
    return tuple(prop["enum"]) if isinstance(prop.get("enum"), list) else ()


def _labels(prop: Mapping[str, Any], defs: Mapping[str, Any]) -> dict[str, str] | None:
    """``value -> label`` of an enum field (or of a list of them), when any label differs from its value."""
    if prop.get("type") == "array":
        return _labels(resolve_property(prop.get("items") or {}, defs), defs)
    if not isinstance(prop.get("enum"), list):
        return None
    options = prop.get("options") or {}
    labels = {str(value): str(options.get(value, value)) for value in prop["enum"]}
    return labels if any(value != text for value, text in labels.items()) else None


def _no_options(prop: Mapping[str, Any], defs: Mapping[str, Any]) -> dict[str, Any]:
    """The empty ``enum`` of a choice core has no option for (or of a list of them), kept in the field's schema.

    ``Literal`` cannot be empty, so such a field is typed by its base type; the caller still reads, from the
    schema, that no value is allowed, and core rejects whatever is sent.
    """
    if prop.get("enum") == []:
        return {"enum": []}
    if prop.get("type") == "array":
        items = resolve_property(prop.get("items") or {}, defs)
        if items.get("enum") == []:
            return {"items": {key: items[key] for key in ("type", "enum") if key in items}}
    return {}


def _literal(values: tuple[Any, ...]) -> Any:
    """``Literal`` of runtime values (a field's allowed ones); a static checker cannot type a dynamic Literal."""
    return Literal.__getitem__(values)


def resolve_property(prop: Mapping[str, Any], defs: Mapping[str, Any]) -> dict[str, Any]:
    """Follow ``$ref`` / single-``allOf`` / ``anyOf``-with-null so the field's real shape is at top level."""
    merged: dict[str, Any] = dict(prop)
    if ref := merged.pop("$ref", None):
        key = str(ref).rsplit("/", 1)[-1]  # the definition's name is its title unless it says otherwise
        merged = {"title": key, **resolve_property(defs.get(key, {}), defs), **merged}
    if len(all_of := merged.pop("allOf", None) or []) == 1:
        merged = {**resolve_property(all_of[0], defs), **merged}
    variants = [v for v in merged.pop("anyOf", None) or [] if v.get("type") != "null"]
    if len(variants) == 1:
        merged = {**resolve_property(variants[0], defs), **merged}
    return merged


_SUMMARY_TABLE: TypeAdapter[SummaryTable] = TypeAdapter(SummaryTable)


def summaries(schema: Mapping[str, Any]) -> list[SummaryTable]:
    """The tables of the workflow's own summary on a page: what its author wants confirmed before the start.

    A summary field is display-only, so it is no field of the page model; its table is not a value but
    part of the field's schema, carried with the widget hints (core's summary form, ``MigrationSummary``).
    """
    defs = schema.get("$defs") or {}
    tables: list[SummaryTable] = []
    for prop in (schema.get("properties") or {}).values():
        resolved = resolve_property(prop, defs) if isinstance(prop, Mapping) else {}
        if resolved.get(FORMAT) != FORMAT_SUMMARY:
            continue
        hints = resolved.get("extraProperties") or resolved.get("uniforms") or {}
        try:
            table = _SUMMARY_TABLE.validate_python(hints.get("data"))
        except ValidationError:
            continue
        if table.get("labels"):
            tables.append(table)
    return tables


def is_read_only(prop: Mapping[str, Any]) -> bool:
    """A field the form renders but never lets the user change (``ReadOnlyField``).

    A bare ``const`` is not that: it is a field with one allowed value, which core still requires.
    """
    widgets = (prop.get("uniforms") or {}, prop.get("extraProperties") or {})
    return bool(prop.get("readOnly")) or any(w.get("disabled") for w in widgets)


# --- what a page model says about its fields ---------------------------------------------------------------


def value_type(info: FieldInfo) -> Any:
    """The field's type without the ``None`` an optional field allows."""
    annotation = info.annotation
    if get_origin(annotation) in (Union, UnionType):
        variants = [a for a in get_args(annotation) if a is not NoneType]
        if len(variants) == 1:
            return variants[0]
    return annotation


def is_list(info: FieldInfo) -> bool:
    return get_origin(value_type(info)) is list


def item_type(info: FieldInfo) -> Any:
    """What a list field holds, or the field's own type when it is not a list."""
    inner = value_type(info)
    return get_args(inner)[0] if get_origin(inner) is list else inner


def choices(info: FieldInfo) -> tuple[Any, ...] | None:
    """The values an enum field (or a list of them) allows, as core gave them, in order; None when the field is free."""
    item = item_type(info)
    return get_args(item) if get_origin(item) is Literal else None


def labels(info: FieldInfo) -> dict[str, str]:
    """``value -> label`` of the field's allowed values, where a label differs from the value."""
    extra = info.json_schema_extra
    found = extra.get(LABELS) if isinstance(extra, dict) else None
    return {str(value): str(text) for value, text in found.items()} if isinstance(found, dict) else {}


def label_of(info: FieldInfo, value: Any) -> Any:
    """How the form shows ``value`` (each item of a list), or None when it shows the value itself."""
    shown = labels(info)
    if isinstance(value, list):
        return [shown.get(str(item), item) for item in value] if any(str(item) in shown for item in value) else None
    return shown.get(str(value))


def is_accept(info: FieldInfo) -> bool:
    """pydantic-forms' ``Accept`` field: consent, given once per page, never a value the caller states in advance."""
    extra = info.json_schema_extra
    return isinstance(extra, dict) and extra.get(FORMAT) == FORMAT_ACCEPT


def widget_mark(info: FieldInfo) -> dict[str, Any] | None:
    """What a form-fill widget wrote on the field: its id, and ``total`` for a long list or ``later`` while it waits."""
    extra = info.json_schema_extra
    mark = extra.get(WIDGET) if isinstance(extra, dict) else None
    return mark if isinstance(mark, dict) else None


def item_bounds(info: FieldInfo) -> tuple[int | None, int | None]:
    """(min, max) items of a list field, from its constraints."""
    lo = next((m.min_length for m in info.metadata if isinstance(m, MinLen)), None)
    hi = next((m.max_length for m in info.metadata if isinstance(m, MaxLen)), None)
    return lo, hi


def is_single_pick(info: FieldInfo) -> bool:
    """A field that takes one of its options: an enum, or a list of one (a single-select ``choice_list``)."""
    return choices(info) is not None and (not is_list(info) or item_bounds(info)[1] == 1)


# --- core's rejection out of the tool error text ------------------------------------------------


class _FormErrorBody(BaseModel):
    """What pydantic-forms' FastAPI handler returns for an invalid page (its ``form_error_handler``)."""

    detail: str = ""
    validation_errors: list[FormError] = Field(default_factory=list)


def form_errors(error: str) -> list[FormError]:
    """Core's validation errors out of the tool error text, as pydantic-forms reports them; [] when it is no such body.

    fastmcp relays core's 400 body as its Python repr inside the text; that body is pydantic-forms' own
    error shape, so it is validated as such rather than picked apart.
    """
    start, end = error.find("{"), error.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        return _FormErrorBody.model_validate(ast.literal_eval(error[start : end + 1])).validation_errors
    except (ValueError, SyntaxError, ValidationError):
        return []


__all__ = [
    "DISPLAY_ONLY_FORMATS",
    "FORMAT",
    "FORMAT_ACCEPT",
    "LABELS",
    "WIDGET",
    "choices",
    "form_errors",
    "form_model",
    "is_accept",
    "is_list",
    "is_read_only",
    "is_single_pick",
    "item_bounds",
    "item_type",
    "label_of",
    "labels",
    "page_model",
    "resolve_property",
    "summaries",
    "value_type",
    "widget_mark",
]
