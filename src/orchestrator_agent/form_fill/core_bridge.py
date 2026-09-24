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

"""What the agent does in core's place until core's form tool speaks agent — DELETE WITH core follow-up 5.

Core's ``get_workflow_form`` returns the raw pydantic-forms JSON schema, written for a browser form: ``$ref`` /
``allOf`` / nullable ``anyOf`` indirection, widget hints in ``uniforms`` and ``extraProperties``, display-only
``format`` markers, enum labels in an ``options`` side table, nested ``$defs`` for structured fields. And a
rejected page comes back as pydantic-forms' error body, relayed by fastmcp as its Python repr inside the tool
error text. Everything in this module exists only to undo that on the agent side:

- ``page_fields`` and its helpers: the schema into the ``FormField``s the skill asks about.
- ``rejected_fields`` / ``error_detail``: core's per-field messages out of the error text.

When core returns a flat field list (name, kind, required, display-only, default, options as value/label
pairs, shape of structured fields) and errors as data — the plan doc's follow-up 5 — this whole module goes,
``FormField`` becomes core's model, and nothing here needs a replacement.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from pydantic_forms.exceptions import ErrorDict

from orchestrator_agent.state import FieldKind, FormField

# pydantic-forms ``format`` markers the skill interprets (the ``json_schema_extra`` of its field types). The
# types themselves are not imported: ``pydantic_forms.validators`` pulls in a contact-person field that needs
# ``email-validator``, which this agent does not ship.
FORMAT_ACCEPT = "accept"
FORMAT_UUID = "uuid"
FORMAT_LONG = "long"
DISPLAY_ONLY_FORMATS = frozenset({"label", "divider", "summary", "markdown", "callout", "hidden", "subscription"})


def option_list(options: Mapping[str, str]) -> str:
    """Allowed values as the caller sees them: ``value (label)``, or just the value when it is its own label."""
    return ", ".join(f"`{v}` ({t})" if t != v else f"`{v}`" for v, t in options.items())


# --- schema -> fields ---------------------------------------------------------------------------


def page_fields(schema: Mapping[str, Any]) -> list[FormField]:
    """Every property of a page schema as a ``FormField`` (display-only ones flagged, not dropped)."""
    defs = schema.get("$defs") or {}
    required = set(schema.get("required") or [])
    fields: list[FormField] = []
    for name, prop in (schema.get("properties") or {}).items():
        if not isinstance(prop, Mapping):
            continue
        resolved = resolve_property(prop, defs)
        kind, options, as_list = _kind(resolved, defs)
        fields.append(
            FormField(
                name=name,
                title=str(resolved.get("title") or name),
                kind=kind,
                required=name in required,
                display_only=is_read_only(resolved) or resolved.get("format") in DISPLAY_ONLY_FORMATS,
                options=options,
                as_list=as_list,
                shape=_shape(resolved, defs) if kind == "json" else None,
                format=resolved.get("format"),
                description=resolved.get("description"),
                has_default="default" in resolved,
                default=resolved.get("default"),
            )
        )
    return fields


def resolve_property(prop: Mapping[str, Any], defs: Mapping[str, Any]) -> dict[str, Any]:
    """Follow ``$ref`` / single-``allOf`` / ``anyOf``-with-null so the field's real shape is at top level."""
    merged: dict[str, Any] = dict(prop)
    if ref := merged.pop("$ref", None):
        merged = {**resolve_property(defs.get(str(ref).rsplit("/", 1)[-1], {}), defs), **merged}
    if len(all_of := merged.pop("allOf", None) or []) == 1:
        merged = {**resolve_property(all_of[0], defs), **merged}
    variants = [v for v in merged.pop("anyOf", None) or [] if v.get("type") != "null"]
    if len(variants) == 1:
        merged = {**resolve_property(variants[0], defs), **merged}
    return merged


def is_read_only(prop: Mapping[str, Any]) -> bool:
    """A field the form renders but never lets the user change (``ReadOnlyField`` / ``const``)."""
    widgets = (prop.get("uniforms") or {}, prop.get("extraProperties") or {})
    return "const" in prop or bool(prop.get("readOnly")) or any(w.get("disabled") for w in widgets)


def _shape(prop: Mapping[str, Any], defs: Mapping[str, Any]) -> str:
    """What a structured field expects, spelled out for the caller: item count and keys with allowed values."""
    if prop.get("type") == "array":
        items = resolve_property(prop.get("items") or {}, defs)
        lo, hi = prop.get("minItems"), prop.get("maxItems")
        count = (
            f"exactly {lo}"
            if lo is not None and lo == hi
            else " to ".join(str(n) for n in (lo, hi) if n is not None) or "any number of"
        )
        return f"a JSON list of {count} objects, each with {_keys(items, defs)}"
    return f"a JSON object with {_keys(prop, defs)}"


def _keys(obj: Mapping[str, Any], defs: Mapping[str, Any]) -> str:
    required = set(obj.get("required") or [])
    parts = []
    for name, sub in (obj.get("properties") or {}).items():
        resolved = resolve_property(sub, defs)
        kind, options, _ = _kind(resolved, defs)
        detail = "one of " + option_list(options) if options else kind
        default = f", default {resolved['default']!r}" if "default" in resolved else ""
        parts.append(f"`{name}` ({'required' if name in required else 'optional'}: {detail}{default})")
    return "keys " + ", ".join(parts) if parts else "no fixed keys"


def _kind(prop: Mapping[str, Any], defs: Mapping[str, Any]) -> tuple[FieldKind, dict[str, str] | None, bool]:
    if prop.get("format") == FORMAT_ACCEPT:
        return "accept", None, False
    if isinstance(prop.get("enum"), list):
        return "choice", _options(prop), False
    match prop.get("type"):
        case "boolean":
            return "boolean", None, False
        case "array":
            items = resolve_property(prop.get("items") or {}, defs)
            if isinstance(items.get("enum"), list):
                single = prop.get("maxItems") == 1
                return ("choice" if single else "multi"), _options(items), single
            return "json", None, False
        case "object":
            return "json", None, False
        case "integer":
            return "integer", None, False
        case "number":
            return "number", None, False
    return "text", None, False


def _options(prop: Mapping[str, Any]) -> dict[str, str]:
    labels = prop.get("options") or {}
    return {str(value): str(labels.get(value, value)) for value in prop["enum"]}


# --- core's rejection out of the tool error text ------------------------------------------------


class _FormErrorBody(BaseModel):
    """What pydantic-forms' FastAPI handler returns for an invalid page (its ``form_error_handler``)."""

    detail: str = ""
    validation_errors: list[ErrorDict] = Field(default_factory=list)


def _error_body(error: str) -> _FormErrorBody | None:
    """Core's validation error out of the tool error text.

    fastmcp relays core's 400 body as its Python repr inside the text; that body is pydantic-forms' own
    error shape, so it is validated as such rather than picked apart.
    """
    start, end = error.find("{"), error.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return _FormErrorBody.model_validate(ast.literal_eval(error[start : end + 1]))
    except (ValueError, SyntaxError, ValidationError):
        return None


def rejected_fields(error: str) -> dict[str, str]:
    """Core's message per field it rejected, in order (a nested ``loc`` counts for its top-level field)."""
    body = _error_body(error)
    problems: dict[str, str] = {}
    for e in body.validation_errors if body is not None else []:
        if e["loc"]:
            problems.setdefault(str(e["loc"][0]), str(e["msg"]))
    return problems


def error_detail(error: str) -> str:
    """The field messages out of core's validation error, as text for the caller."""
    body = _error_body(error)
    if body is None:
        return error.strip()[:300]
    messages = [f"{'.'.join(str(p) for p in e['loc']) or 'form'}: {e['msg']}" for e in body.validation_errors]
    return "; ".join(messages) or body.detail or error.strip()[:300]


__all__ = [
    "DISPLAY_ONLY_FORMATS",
    "FORMAT_ACCEPT",
    "FORMAT_LONG",
    "FORMAT_UUID",
    "error_detail",
    "is_read_only",
    "option_list",
    "page_fields",
    "rejected_fields",
    "resolve_property",
]
