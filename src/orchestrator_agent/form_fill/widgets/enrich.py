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

from orchestrator_agent.form_fill.core_bridge import resolve_property
from orchestrator_agent.form_fill.widgets.base import (
    MAX_INLINE,
    WIDGET_MARK,
    FieldWidget,
    Option,
    WidgetContext,
)
from orchestrator_agent.form_fill.widgets.registry import match_widget, widget_target

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


def _choice(target: Mapping[str, Any], options: Sequence[Option]) -> dict[str, Any]:
    return {**target, "enum": [o.value for o in options], "options": {str(o.value): o.label for o in options}}


def _with_target(prop: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {**prop, "items": target} if prop.get("type") == "array" else target


class _Fetches:
    """The options each widget has on one page, fetched once per widget and hints."""

    def __init__(self, ctx: WidgetContext) -> None:
        self.ctx = ctx
        self.seen: dict[tuple[str, str], Sequence[Option] | None] = {}

    async def __call__(self, widget: FieldWidget, target: Mapping[str, Any]) -> Sequence[Option] | None:
        hints = {key: target.get(key) for key in ("format", "extraProperties", "uniforms")}
        key = (widget.id, json.dumps(hints, sort_keys=True, default=str))
        if key not in self.seen:
            self.seen[key] = await widget.options(target, self.ctx)
        return self.seen[key]


async def enrich(schema: Mapping[str, Any], widgets: Sequence[FieldWidget], ctx: WidgetContext) -> EnrichedPage:
    """``schema`` with every property a widget matches rewritten; every other property as it came."""
    defs = schema.get("$defs") or {}
    fetch = _Fetches(ctx)
    found = [
        await _property(name, prop, defs, widgets, fetch) for name, prop in (schema.get("properties") or {}).items()
    ]
    widget_choices = {p.name: p.choice for p in found if p.choice is not None}
    if not found:
        return EnrichedPage(dict(schema))
    return EnrichedPage({**schema, "properties": {p.name: p.schema for p in found}}, widget_choices)


async def _property(
    name: str, prop: Any, defs: Mapping[str, Any], widgets: Sequence[FieldWidget], fetch: _Fetches
) -> _Property:
    """One property after ``enrich``: unchanged unless a widget matches it and could say what its options are."""
    resolved = resolve_property(prop, defs) if isinstance(prop, Mapping) else {}
    widget = match_widget(widgets, resolved) if resolved else None
    if widget is None:
        return _Property(name, prop)
    target = resolve_property(widget_target(resolved), defs)
    try:
        options = await fetch(widget, target)
    except Exception as exc:  # a widget is an add-on: without it the field is asked as before
        logger.warning("Form-fill widget failed", widget=widget.id, field=name, error=str(exc))
        return _Property(name, prop)
    if options is None:
        return _Property(name, {**resolved, WIDGET_MARK: {"id": widget.id, "later": True}})
    title = str(resolved.get("title") or name)
    choice = LongList(widget, target, tuple(options), title=title, multiple=resolved.get("type") == "array")
    if len(options) <= MAX_INLINE:
        inlined = {**_with_target(resolved, _choice(target, options)), WIDGET_MARK: {"id": widget.id}}
        return _Property(name, inlined, choice)
    marked = {**_with_target(resolved, target), WIDGET_MARK: {"id": widget.id, "total": len(options)}}
    return _Property(name, marked, choice)


__all__ = ["EnrichedPage", "LongList", "enrich"]
