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

"""A page schema with the widgets' options written in, the way core would have sent a static list.

A field a widget knows is rewritten before the page model is built: a short list becomes core's own choice
shape (``enum`` + ``options``), so the stops, the labels and the approval need nothing new; a long list is
marked and asked as text. Either way the field and its options are kept aside (``EnrichedPage.choices``), and
what a person types for it is resolved to one of its options before core sees it (``widgets.resolve``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog
from pydantic_forms.utils.schema import OPTIONS_SOURCE

from orchestrator_agent.form_fill.model import WIDGET_MARK, shape_of
from orchestrator_agent.form_fill.widgets.base import (
    MAX_INLINE,
    FieldWidget,
    Option,
    WidgetContext,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class LongList:
    """A widget field with its options: its widget, its (items') property, every option.

    Named for the long list a question cannot show; a short list shown as chips is kept the same way, so what
    is typed for either is resolved to one of its options.
    """

    widget: FieldWidget
    field: dict[str, Any]
    options: tuple[Option, ...]
    title: str = ""
    multiple: bool = False


@dataclass(frozen=True)
class EnrichedPage:
    """The rewritten schema, and every widget field that got options (inlined or long), by name."""

    schema: dict[str, Any]
    choices: dict[str, LongList] = field(default_factory=dict)


@dataclass(frozen=True)
class _Property:
    """One property of the page after ``enrich``: as it goes into the schema, and its options if a widget gave some."""

    name: str
    schema: Any
    choice: LongList | None = None


def widget_target(prop: Mapping[str, Any]) -> Mapping[str, Any]:
    """What a widget is matched on: a list's items, any other property itself.

    A source on the list itself describes the options of its items (a list of a sourced type carries it on the
    items instead), so the items are matched with the list's source when they have none of their own.
    """
    items = prop.get("items")
    if prop.get("type") != "array" or not isinstance(items, Mapping):
        return prop
    if OPTIONS_SOURCE in prop and OPTIONS_SOURCE not in items:
        return {**items, OPTIONS_SOURCE: prop[OPTIONS_SOURCE]}
    return items


def match_widget(widgets: Sequence[FieldWidget], prop: Mapping[str, Any]) -> FieldWidget | None:
    """The widget of a resolved property, if any: the first whose ``matches`` accepts it.

    Core's own options always win (a property with ``enum`` or ``const``), what nobody fills is no widget's
    (``readOnly``: shown, never submitted), and a list is matched on its items.
    """
    target = widget_target(prop)
    if any(key in target for key in ("enum", "const")) or prop.get("readOnly") or target.get("readOnly"):
        return None
    return next((widget for widget in widgets if widget.matches(target)), None)


def _choice(target: Mapping[str, Any], options: Sequence[Option]) -> dict[str, Any]:
    return {**target, "enum": [o.value for o in options], "options": {str(o.value): o.label for o in options}}


def _with_target(prop: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {**prop, "items": target} if prop.get("type") == "array" else target


class _Fetches:
    """The options each widget has on one page, fetched once per widget and source: two fields of one type share."""

    def __init__(self, ctx: WidgetContext) -> None:
        self.ctx = ctx
        self.seen: dict[tuple[str, str], Sequence[Option] | None] = {}

    async def __call__(self, widget: FieldWidget, target: Mapping[str, Any]) -> Sequence[Option] | None:
        decides = {key: target.get(key) for key in ("format", OPTIONS_SOURCE)}
        key = (widget.id, json.dumps(decides, sort_keys=True, default=str))
        if key not in self.seen:
            self.seen[key] = await widget.options(target, self.ctx)
        return self.seen[key]


async def enrich(schema: Mapping[str, Any], widgets: Sequence[FieldWidget], ctx: WidgetContext) -> EnrichedPage:
    """``schema`` with every property a widget matches rewritten; every other property as it came."""
    fetch = _Fetches(ctx)
    found = [await _property(name, prop, widgets, fetch) for name, prop in (schema.get("properties") or {}).items()]
    widget_choices = {p.name: p.choice for p in found if p.choice is not None}
    if not found:
        return EnrichedPage(dict(schema))
    return EnrichedPage({**schema, "properties": {p.name: p.schema for p in found}}, widget_choices)


async def _property(name: str, prop: Any, widgets: Sequence[FieldWidget], fetch: _Fetches) -> _Property:
    """One property after ``enrich``: unchanged unless a widget matches it and could say what its options are."""
    resolved = shape_of(prop) if isinstance(prop, Mapping) else {}
    widget = match_widget(widgets, resolved) if resolved else None
    if widget is None:
        return _Property(name, prop)
    target = shape_of(widget_target(resolved))
    try:
        return await _with_options(name, resolved, target, widget, fetch)
    except Exception as exc:  # a widget is an add-on: without it the field is asked as before
        logger.warning("Form-fill widget failed", widget=widget.id, field=name, error=str(exc))
        return _Property(name, prop)


async def _with_options(
    name: str, resolved: dict[str, Any], target: dict[str, Any], widget: FieldWidget, fetch: _Fetches
) -> _Property:
    """The property with its widget's options written in, or marked as waiting when they depend on a value not known yet."""
    options = await fetch(widget, target)
    if options is None:
        return _Property(name, {**resolved, WIDGET_MARK: {"id": widget.id, "later": True}})
    title = str(resolved.get("title") or name)
    choice = LongList(widget, target, tuple(options), title=title, multiple=resolved.get("type") == "array")
    return _shown(name, resolved, target, widget, choice)


def _shown(name: str, prop: dict[str, Any], target: dict[str, Any], widget: FieldWidget, choice: LongList) -> _Property:
    """The property as asked: its options as chips when they fit a question, else marked as a long list to type."""
    if choice.options and len(choice.options) <= MAX_INLINE:
        inlined = {**_with_target(prop, _choice(target, choice.options)), WIDGET_MARK: {"id": widget.id}}
        return _Property(name, inlined, choice)
    total = {"id": widget.id, "total": len(choice.options)}
    return _Property(name, {**_with_target(prop, target), WIDGET_MARK: total}, choice)


__all__ = ["EnrichedPage", "LongList", "enrich", "match_widget", "widget_target"]
