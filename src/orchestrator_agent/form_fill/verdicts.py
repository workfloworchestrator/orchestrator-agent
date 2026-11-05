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

"""What a page's schema alone decides about an answer, before anything is submitted.

Core is the judge of a page; these are the verdicts its schema already spells out, said by the agent so that no
core round trip and no model call is spent on them: a field no answer can satisfy, a list answered with the
wrong number of items, and the shape of a field (a single pick, an answer made of picks).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_forms.core import FormFieldError

from orchestrator_agent.form_fill.model import FieldSpec, options_of


def single_pick(field: FieldSpec) -> bool:
    """A field that takes one of its options: a choice, or a list of at most one (a single-select ``choice_list``)."""
    return options_of(field) is not None and (field.kind != "list" or field.constraints.get("max_length") == 1)


def picked(field: FieldSpec, raw: Any) -> bool:
    """Whether an answer is made of the field's own option values: picked, not typed, so nothing to read."""
    options = options_of(field)
    if options is None or raw is None:
        return False
    values = {option.value for option in options}
    items = raw if isinstance(raw, list) else [raw]
    return bool(items) and all(item in values for item in items)


def count_of(field: FieldSpec) -> str:
    """How many items a list takes ("Pick 2", "Give 1 to 6", "Pick at least 1"); empty for a single pick or no limit."""
    low, high = field.constraints.get("min_length"), field.constraints.get("max_length")
    if field.kind != "list" or single_pick(field) or low is None and high is None:
        return ""
    verb = "Pick" if options_of(field) is not None else "Give"
    match (low, high):
        case (_, None):
            return f"{verb} at least {low}"
        case (None, _):
            return f"{verb} at most {high}"
        case _ if low == high:
            return f"{verb} {high}"
        case _:
            return f"{verb} {low} to {high}"


def _fits(field: FieldSpec, count: int) -> bool:
    """Whether ``count`` items satisfy the list's size limits."""
    low, high = field.constraints.get("min_length"), field.constraints.get("max_length")
    return (low is None or count >= low) and (high is None or count <= high)


def _dead_end(field: FieldSpec) -> str:
    """Why no answer can satisfy the field, if so: nothing to choose from, or a size no list has; empty otherwise.

    Core's own option lists only: a widget's empty list says nothing about what core takes for the field.
    """
    options = options_of(field) if field.widget is None else None
    low, high = field.constraints.get("min_length"), field.constraints.get("max_length")
    needs_one = field.required if field.kind != "list" else (low or 0) >= 1
    if options == [] and needs_one:
        return f"{field.title} has no options to choose from"
    if field.kind == "list" and low is not None and high is not None and low > high:
        return f"{field.title} needs at least {low} items but takes at most {high}"
    return ""


def unanswerable(fields: Sequence[FieldSpec]) -> str:
    """Why a page can never be answered, if so: known from its schema alone, so said instead of asked."""
    return next(filter(None, map(_dead_end, fields)), "")


def _count_error(field: FieldSpec, count: int) -> FormFieldError:
    """The verdict on a list of ``count`` items, in the words of its hint and pydantic's error type."""
    high = field.constraints.get("max_length")
    kind = "too_long" if high is not None and count > high else "too_short"
    return FormFieldError(loc=[field.name], msg=f"{count_of(field)}, not {count}.", type=kind)


def miscounted(fields: Sequence[FieldSpec], values: Mapping[str, Any]) -> list[FormFieldError]:
    """Each list answered with the wrong number of items, said as the hint said it: known here, so never submitted."""
    return [
        _count_error(field, len(values[field.name]))
        for field in fields
        if field.kind == "list"
        and isinstance(values.get(field.name), list)
        and not _fits(field, len(values[field.name]))
    ]


__all__ = ["count_of", "miscounted", "picked", "single_pick", "unanswerable"]
