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

"""What a person typed for a field, interpreted into the value the field expects — one call per page.

A form's stops are questions a human answers. A picked option travels as its value and needs nothing
here. What a person *types* is words, and words are not form values: core rejects them, the skill then
asks the ``Interpreter`` once, for all rejected fields of the page at once, and core decides again. The
human sees the result before anything starts, at the approval. Nothing else is ever read: no chat
message, no decision.

The seam is one protocol call — the form's model plus the person's words, in; the values, out — so any
engine that answers a page of typed questions in one call can sit behind it. Here it is pydantic-ai with
the page model's partial variant as the output type (``ModelInterpreter``); the Jev branch puts its typed
decision engine behind the same protocol.

Words that fit more than one option of a field are not a value: the output type asks for every option the
words fit, and the field is taken only when that is exactly one (``reading_type``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import structlog
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_ai import Agent
from pydantic_ai.models import Model

from orchestrator_agent.form_fill.core_bridge import choices, is_list, value_type

logger = structlog.get_logger(__name__)

INSTRUCTIONS = (
    "Form fields expect values and a person answered in their own words. The fields are given as a JSON schema; "
    "where a field carries labels, they map each allowed value to how it is shown to people. Translate what was "
    "said into the values the fields expect, typed exactly as the schema says: a number written in words or with "
    "a unit is that number, a label or a description of an option is that option's value, details of items are "
    "the objects the schema describes. A field that takes one of several options is asked as a list: give every "
    "allowed value the words fit, so more than one when they do not tell the options apart; choosing between "
    "them is not yours. Leave a field null when nothing usable was said for it. Never add what the words do not "
    "support."
)


class Interpreter(Protocol):
    """The one reading a form needs from an engine: the person's words per rejected field -> the values they meant.

    ``form`` is the pydantic model of the fields in play (``core_bridge.form_model``); a field the words
    say nothing usable about is left out.
    """

    async def answers(self, form: type[BaseModel], words: Mapping[str, str]) -> Mapping[str, Any]: ...


class Chooser(Protocol):
    """Which of a field's options a person's words fit: every value they fit, so the caller sees when several do."""

    async def choose(self, title: str, options: Sequence[tuple[Any, str]], words: str) -> list[Any]: ...


CHOOSE_INSTRUCTIONS = (
    "A person answered a form field in their own words. The field's options are given as values with how they are "
    "shown to people. Give every option value the words fit — a name, a short name, a description or a part of "
    "one — so more than one when the words do not tell the options apart, and none when they fit none. Never give "
    "a value that is not one of the options."
)


def fields_model(form: type[BaseModel], names: Iterable[str]) -> type[BaseModel]:
    """The named fields of ``form`` as a model of their own (what one reading is about)."""
    wanted = set(names)
    picked: dict[str, Any] = {
        name: (info.annotation, info) for name, info in form.model_fields.items() if name in wanted
    }
    return create_model("Fields", __config__=ConfigDict(protected_namespaces=()), **picked)


def reading_type(form: type[BaseModel]) -> type[BaseModel]:
    """The output model of one reading: every field of ``form`` made optional.

    A field that takes one of several options is read as every option the words fit (``_one_of_several``): a
    model asked for the one value settles words that fit two by guessing, asked for all of them it names both,
    and ``_values`` then takes the field only when exactly one fits.
    """
    attributes: dict[str, Any] = {
        name: (
            _optional(list[value_type(info)] if _one_of_several(info) else info.annotation),  # type: ignore[misc]
            Field(
                default=None, title=info.title, description=info.description, json_schema_extra=info.json_schema_extra
            ),
        )
        for name, info in form.model_fields.items()
    }
    return create_model("Reading", __config__=ConfigDict(protected_namespaces=()), **attributes)


def _one_of_several(info: FieldInfo) -> bool:
    """A field whose value is one of several options (not a list of them, not a single-option field)."""
    return not is_list(info) and len(choices(info) or ()) > 1


def _optional(annotation: Any) -> Any:
    return annotation | None


def _schema(form: type[BaseModel]) -> str:
    return json.dumps(form.model_json_schema(), separators=(",", ":"))


@dataclass(frozen=True)
class ModelInterpreter:
    """pydantic-ai as the interpreter: one run per reading, with the page model's partial variant as the output type."""

    model: Model | str

    async def answers(self, form: type[BaseModel], words: Mapping[str, str]) -> dict[str, Any]:
        asked = fields_model(form, words)
        if not asked.model_fields:
            return {}
        agent: Agent[None, Any] = Agent(self.model, output_type=reading_type(asked), instructions=INSTRUCTIONS)
        prompt = "\n".join(
            [
                f"Fields (JSON schema): {_schema(asked)}",
                "Answers:",
                *(f"- {name}: {words[name]}" for name in asked.model_fields),
            ]
        )
        values = _values((await agent.run(prompt)).output, asked)
        logger.info("Form-fill answers interpreted", asked=list(asked.model_fields), values=values)
        return values

    async def choose(self, title: str, options: Sequence[tuple[Any, str]], words: str) -> list[Any]:
        values = tuple(value for value, _ in options)
        if not values:
            return []
        reading = create_model("Choice", values=(list[Literal.__getitem__(values)], Field(default_factory=list)))  # type: ignore[misc]
        agent: Agent[None, Any] = Agent(self.model, output_type=reading, instructions=CHOOSE_INSTRUCTIONS)
        listed = "\n".join(f"- {json.dumps(value)}: {label}" for value, label in options)
        prompt = f"Field: {title}\nOptions (value: shown as):\n{listed}\nAnswer: {words}"
        chosen = list((await agent.run(prompt)).output.values)
        logger.info("Form-fill option chosen", field=title, words=words, chosen=chosen)
        return chosen


def _values(output: BaseModel, form: type[BaseModel]) -> dict[str, Any]:
    """The fields the reading gave a value, as plain data (a nested model is its object); nulls are left out.

    A field read as the options the words fit is a value only when exactly one fits: words that fit none, or
    do not tell several apart, leave it open and the person is asked again.
    """
    values: dict[str, Any] = {}
    for name, value in output.model_dump(mode="json", exclude_unset=True).items():
        if value is None:
            continue
        if _one_of_several(form.model_fields[name]):
            if len(value) != 1:
                continue
            (value,) = value
        values[name] = value
    return values


__all__ = [
    "CHOOSE_INSTRUCTIONS",
    "INSTRUCTIONS",
    "Chooser",
    "Interpreter",
    "ModelInterpreter",
    "fields_model",
    "reading_type",
]
