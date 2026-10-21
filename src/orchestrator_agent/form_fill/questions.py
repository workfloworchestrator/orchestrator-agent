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

"""A stop as questions for the human: how a field is worded, with its options, its hint and core's verdict on it.

The questions are worded with the labels and help texts core writes into the schema, as the frontend shows
them; a widget field says how many options there are to search or what typed words matched; a list says how
many items it takes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from typing import Any

from pydantic_forms.core import FormFieldError

from orchestrator_agent.form_fill.model import FieldSpec, askable, options_of
from orchestrator_agent.form_fill.verdicts import count_of
from orchestrator_agent.form_fill.widgets import MAX_INLINE
from orchestrator_agent.form_fill.widgets.resolve import Resolution
from orchestrator_agent.form_fill.widgets.steps import CHANGE_STEP, waits
from orchestrator_agent.state import AskField, Reply

_MISSING = "missing"  # pydantic's error type for a required field without a value: the question says so already
_FIELD_REQUIRED = "Field required"  # its message, for a verdict that carries no type

# What a stop says about a widget field (``AskField.hint``): the size of a long list, what typed words matched.
LONG_LIST_HINT = "Type to search {total} options."
SEVERAL_FIT = "More than one option fits what was typed: pick one, or type more of the name."
TOO_MANY_FIT = "Many options fit what was typed: type more of the name."
NOTHING_FIT = "Nothing matched {words}: type another name."
NO_OPTIONS = "There are no options for {title} here."


def missing(error: FormFieldError) -> bool:
    """Whether core's error only says the field has no value yet: the question's ``*`` says that already."""
    return error.type == _MISSING or error.msg == _FIELD_REQUIRED


def questions(
    fields: Sequence[FieldSpec],
    values: Mapping[str, Any],
    errors: Sequence[FormFieldError],
    reason: str | None = None,
    *,
    asked: Collection[str] = (),
) -> list[AskField]:
    """The stop as questions for the human, in page order: the fields core rejected, with its message, and the open ones.

    A page core refused without naming one of its fields — its errors name none, or the refusal is no form
    verdict at all (``reason``) — is asked again in full, the first question with what core said. An optional
    field the person was ``asked`` before and left at its default is not asked again. A field waiting for
    another value (``later``) is left out while anything else is asked; when nothing else is, it is asked as
    typed text, like a field without its widget, so a stop never comes without a question. A field asked for
    a step that comes before its value (a node, before the port) ends the stop: what follows it on the page
    may depend on that choice, so it is asked with the value, on the next stop.
    """
    open_fields = [field for field in fields if askable(field)]
    waiting = {field.name for field in open_fields if waits(field)}
    found = _questions(open_fields, values, errors, reason, waiting, asked) or _questions(
        open_fields, values, errors, reason, set(), asked
    )
    return _up_to_a_step(open_fields, found)


def _up_to_a_step(fields: Sequence[FieldSpec], found: list[AskField]) -> list[AskField]:
    """The questions up to the first one asked for a step (in page order); all of them when none is."""
    order = {field.name: index for index, field in enumerate(fields)}
    marks = {field.name: field.widget or {} for field in fields}
    steps = [order[asked.name] for asked in found if marks[asked.name].get("step")]
    return [asked for asked in found if order[asked.name] <= min(steps)] if steps else found


def _questions(
    fields: Sequence[FieldSpec],
    values: Mapping[str, Any],
    errors: Sequence[FormFieldError],
    reason: str | None,
    waiting: set[str],
    asked: Collection[str],
) -> list[AskField]:
    """``questions``, leaving out the ``waiting`` fields and core's errors on them."""
    shown = {field.name: field for field in fields if field.name not in waiting}
    errors = [error for error in errors if not (error.loc and str(error.loc[0]) in waiting)]
    problems: dict[str, str] = {}
    for error in errors:
        if error.loc and (name := str(error.loc[0])) in shown:
            problems.setdefault(name, "" if missing(error) else error.msg)
    if (errors or reason) and not problems:
        said = "; ".join(error.msg for error in errors) or str(reason)
        problems = {name: said if index == 0 else "" for index, name in enumerate(shown)}
    open_ = {
        name
        for name, field in shown.items()
        if name not in problems and name not in values and (field.required or name not in asked)
    }
    return [  # in page order, as a frontend shows them
        question(field, problems.get(name, "")) for name, field in shown.items() if name in problems or name in open_
    ]


def question(field: FieldSpec, problem: str = "") -> AskField:
    """One field as a question: its label, ``*`` when it is required, its options as chips, core's message if it rejected the answer.

    Nothing spells out what the field expects: a person answers in words, the interpreter has the schema. The
    hint under the question is the field's help text from the schema, or for a widget field how many options
    there are to search, or what the typed words matched.
    """
    title = _title(field)
    text = title + (" *" if field.required else "") + (f" — {problem}" if problem else "")
    options = options_of(field)
    values: tuple[Any, ...] = tuple(option.value for option in options) if options is not None else ()
    if options is None and field.kind == "boolean":
        values = (True, False)
    shown = {str(option.value): option.label for option in options or []}
    mark = field.widget or {}
    hint = LONG_LIST_HINT.format(total=mark["total"]) if mark.get("total") else ""
    if "total" in mark and not mark["total"]:  # a widget field whose source has nothing to offer
        hint = NO_OPTIONS.format(title=title)
    # The field's help text, where nothing else is said — a step's question asks for the step, not the field.
    if not mark.get("total") and not mark.get("step") and field.description:
        hint = " ".join(filter(None, (hint, field.description)))
    if count := count_of(field):  # a list's size is known from the schema: said up front, not learnt from core
        hint = " ".join(filter(None, (hint, f"{count}.")))
    if stepped := mark.get("stepped"):  # its options follow from steps chosen first: offer to choose again
        change = CHANGE_STEP.format(step=" / ".join(stepped))
        values, hint = (*values, change), " ".join(filter(None, (hint, f'Pick "{change}" to go back.')))
    return AskField(
        name=field.name,
        question=text,
        choices=unique_choices(values, shown),  # the human sees labels, told apart where they coincide
        values=values,
        multiple=field.kind == "list",
        required=field.required,
        title=title,
        problem=problem,
        hint=hint,
        searchable=bool(mark.get("total")),
    )


def _title(field: FieldSpec) -> str:
    """The field's label, the schema's title; a cascade step keeps its own part ("Port — Node")."""
    schema_title = field.title or field.name
    stepped = " — " in schema_title and bool((field.widget or {}).get("step"))
    step = schema_title.split(" — ", 1)[1] if stepped else None
    label = schema_title.split(" — ", 1)[0]
    return f"{label} — {step.capitalize()}" if step else label


def unique_choices(options: Sequence[Any], shown: Mapping[str, str]) -> tuple[str, ...]:
    """The options as a person sees them: each by its label, with its value added where labels coincide.

    Labels are descriptions and descriptions are not unique, while a chip's text is all a transport gets
    back from a pick; two options must never read the same, or a pick could not be mapped to the value core
    takes (an id, never a label). The value is what tells them apart, and it is what the person confirms.
    """
    texts = [shown.get(str(value), str(value)) for value in options]
    counts = Counter(texts)
    return tuple(f"{text} ({value})" if counts[text] > 1 else text for text, value in zip(texts, options, strict=True))


def offered(reply: Reply, unresolved: Mapping[str, Resolution]) -> Reply:
    """The stop, with what resolving said about each field whose words it could not resolve."""
    if not reply.ask or not unresolved:
        return reply
    return replace(reply, ask=[_offer(field, unresolved.get(field.name)) for field in reply.ask])


def _offer(field: AskField, outcome: Resolution | None) -> AskField:
    """A field asked again: the options that fit as chips, or why nothing is offered."""
    match outcome:
        case None:
            return field
        case Resolution(candidates=candidates) if 0 < len(candidates) <= MAX_INLINE:
            labels, values = tuple(o.label for o in candidates), tuple(o.value for o in candidates)
            return replace(field, choices=labels, values=values, hint=SEVERAL_FIT)
        case Resolution(candidates=candidates) if candidates:
            return replace(field, hint=TOO_MANY_FIT)
        case _:
            return replace(field, hint=NOTHING_FIT.format(words=", ".join(f"'{w}'" for w in outcome.unmatched)))


__all__ = [
    "CHANGE_STEP",
    "LONG_LIST_HINT",
    "NOTHING_FIT",
    "NO_OPTIONS",
    "SEVERAL_FIT",
    "TOO_MANY_FIT",
    "missing",
    "offered",
    "question",
    "questions",
    "unique_choices",
]
