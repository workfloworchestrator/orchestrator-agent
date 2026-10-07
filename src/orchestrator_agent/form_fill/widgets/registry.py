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

"""Which widget a property is: the built-ins, as a deployment's extender rearranges them.

Mirrors pydantic-forms' ``ComponentMatcherExtender``: the extender receives the agent's widgets and returns
the list to use; the first widget that matches a property is its widget, so an extender puts its own
first or drops a built-in by ``id``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

from orchestrator_agent.form_fill.core_bridge import DISPLAY_ONLY_FORMATS, FORMAT, is_read_only
from orchestrator_agent.form_fill.widgets.base import FieldWidget, FieldWidgetExtender

SETTING = "FORM_WIDGET_EXTENDER"


def build_widgets(builtins: Sequence[FieldWidget], extender: FieldWidgetExtender | None = None) -> list[FieldWidget]:
    """The widgets in the order they are tried: the built-ins, or what the extender made of them."""
    if extender is None:
        return list(builtins)
    widgets = extender(list(builtins))
    if not isinstance(widgets, list) or not all(hasattr(widget, "matches") for widget in widgets):
        raise ValueError(f"{SETTING}: the extender must return a list of widgets, got {widgets!r}")
    return widgets


def load_extender(path: str | None) -> FieldWidgetExtender | None:
    """The extender named ``package.module:callable``; None when unset. A bad path fails loudly, at startup."""
    if not path:
        return None
    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"{SETTING}={path!r}: expected 'package.module:callable'")
    try:
        extender = getattr(import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"{SETTING}={path!r}: {exc}") from exc
    if not callable(extender):
        raise ValueError(f"{SETTING}={path!r}: {attribute} is not callable")
    return extender  # type: ignore[no-any-return]


def widget_target(prop: Mapping[str, Any]) -> Mapping[str, Any]:
    """What a widget is matched on: a list's items, any other property itself."""
    items = prop.get("items")
    return items if prop.get("type") == "array" and isinstance(items, Mapping) else prop


def match_widget(widgets: Sequence[FieldWidget], prop: Mapping[str, Any]) -> FieldWidget | None:
    """The widget of a resolved property, if any.

    Core's own options always win (a property with ``enum`` or ``const``), what nobody fills is no widget's
    (read-only or display-only), and a list is matched on its items.
    """
    target = widget_target(prop)
    if any(key in target for key in ("enum", "const")) or is_read_only(prop) or is_read_only(target):
        return None
    if target.get(FORMAT) in DISPLAY_ONLY_FORMATS:
        return None
    return next((widget for widget in widgets if widget.matches(target)), None)


__all__ = ["SETTING", "build_widgets", "load_extender", "match_widget", "widget_target"]
