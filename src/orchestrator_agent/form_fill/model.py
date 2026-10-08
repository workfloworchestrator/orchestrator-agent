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

"""A form page as a pydantic model, built from core's field spec: what the interpreter reads and answers with.

Core's ``get_workflow_form`` describes a page as data (``FormField``s), and the skill works from that data.
The one place that needs a pydantic model is the interpreter: the model run that turns a person's words
into values has the page as its output type and is shown its JSON schema. ``page_model`` builds it — a
choice as a ``Literal`` of its values with the labels kept in the schema, a list of what its item is, a
nested object as a model of its own, and nothing for a field that is only shown.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from orchestrator.core.schemas.mcp_tools import FormField, FormFieldOption
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

_KINDS: dict[str, Any] = {"string": str, "integer": int, "number": float, "boolean": bool, "object": dict[str, Any]}


# --- what a field spec says ------------------------------------------------------------------------------


def askable(field: FormField) -> bool:
    """A field a caller may send a value for: not one that is only shown, or shown and fixed."""
    return not (field.display_only or field.read_only)


def options_of(field: FormField) -> list[FormFieldOption] | None:
    """The options a field (or, for a list, its item) is limited to; None when the value is free."""
    if field.options is None and field.item is not None:
        return field.item.options
    return field.options


def label_of(field: FormField, value: Any) -> Any:
    """How the form shows ``value`` (each item of a list), or None when it shows the value itself."""
    shown = {str(o.value): o.label for o in options_of(field) or [] if o.label != str(o.value)}
    if isinstance(value, list):
        return [shown.get(str(item), item) for item in value] if any(str(item) in shown for item in value) else None
    return shown.get(str(value))


# --- the page as a model -----------------------------------------------------------------------------------


def page_model(fields: Sequence[FormField]) -> type[BaseModel]:
    """The page as a model of the fields a caller may send; one page gives one model (cached)."""
    return _model("Page", tuple(field.model_dump_json() for field in fields))


def form_model(pages: Sequence[Sequence[FormField]]) -> type[BaseModel]:
    """The walked pages as one model (a later page wins a name): what a reading of the form works from."""
    named = {field.name: field for page in pages for field in page}
    return page_model(list(named.values()))


@lru_cache(maxsize=256)
def _model(name: str, fields_json: tuple[str, ...]) -> type[BaseModel]:
    fields = [FormField.model_validate_json(field) for field in fields_json]
    attributes: dict[str, Any] = {field.name: _attribute(field) for field in fields if askable(field)}
    return create_model(name, __config__=ConfigDict(protected_namespaces=()), **attributes)


def _attribute(field: FormField) -> tuple[Any, FieldInfo]:
    annotation = _annotation(field)
    if field.nullable or (not field.required and field.default is None):
        annotation = annotation | None
    labels = {str(o.value): o.label for o in options_of(field) or [] if o.label != str(o.value)}
    extra: dict[str, Any] = {key: value for key, value in (("format", field.format), ("labels", labels)) if value}
    info = Field(
        ... if field.required else field.default,
        title=field.title,
        description=field.description,
        json_schema_extra=extra or None,
    )
    return annotation, info


def _annotation(field: FormField) -> Any:
    """The type of the value a field takes: its allowed values, a scalar, a list of its item, or a nested model."""
    if field.options:  # a choice with no option today keeps its base type: core rejects whatever is sent
        return Literal.__getitem__(tuple(option.value for option in field.options))
    if field.kind == "list":
        item = field.item
        inner = (_annotation(item) | None if item.nullable else _annotation(item)) if item else Any
        return list[inner]  # type: ignore[valid-type]
    if field.kind == "object" and field.fields:
        return _model(field.title.replace(" ", "") or "Item", tuple(f.model_dump_json() for f in field.fields))
    return _KINDS.get(field.kind, Any)


# --- what a built model says about its fields (the interpreter reads its own output type) ----------------


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


def choices(info: FieldInfo) -> tuple[Any, ...] | None:
    """The values an option field (or a list of them) allows, in order; None when the field is free."""
    inner = value_type(info)
    item = get_args(inner)[0] if get_origin(inner) is list else inner
    return get_args(item) if get_origin(item) is Literal else None


__all__ = ["askable", "choices", "form_model", "is_list", "label_of", "options_of", "page_model", "value_type"]
