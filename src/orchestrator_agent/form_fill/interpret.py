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

The seam is one protocol call — the fields' schema plus the person's words, in; the values, out — so any
engine that answers a page of typed questions in one call can sit behind it. Here it is pydantic-ai with
the page schema itself as the output type (``ModelInterpreter``, through ``StructuredDict``): the page as
core defined it, every field optional, no model rebuilt from it. What comes back is not validated here;
it goes to core as a page, and core's validation is the only validation. The Jev branch puts its typed
decision engine behind the same protocol.

Words that fit more than one option of a field are not a value: the output schema asks for every option the
words fit, and the field is taken only when that is exactly one (``reading_schema``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import structlog
from pydantic_ai import Agent, StructuredDict
from pydantic_ai.models import Model

from orchestrator_agent.form_fill.model import FieldSpec, fields_in

logger = structlog.get_logger(__name__)

INSTRUCTIONS = (
    "Form fields expect values and a person answered in their own words. The fields are given as a JSON schema; "
    "where a field carries options, they map each allowed value to how it is shown to people. Translate what was "
    "said into the values the fields expect, typed exactly as the schema says: a number written in words or with "
    "a unit is that number, a label or a description of an option is that option's value, details of items are "
    "the objects the schema describes. A field that takes one of several options is asked as a list: give every "
    "allowed value the words fit, so more than one when they do not tell the options apart; choosing between "
    "them is not yours. Leave a field null when nothing usable was said for it. Never add what the words do not "
    "support."
)


class Interpreter(Protocol):
    """The one reading a form needs from an engine: the person's words per rejected field -> the values they meant.

    ``form`` is the schema of the fields in play (``model.form_schema``); a field the words say nothing
    usable about is left out.
    """

    async def answers(self, form: Mapping[str, Any], words: Mapping[str, str]) -> Mapping[str, Any]: ...


def fields_schema(form: Mapping[str, Any], names: Iterable[str]) -> dict[str, Any]:
    """The named fields of ``form`` as a schema of their own (what one reading is about)."""
    wanted = set(names)
    properties = {name: prop for name, prop in _properties(form).items() if name in wanted}
    return {"type": "object", "properties": properties, "additionalProperties": False}


def reading_schema(form: Mapping[str, Any]) -> dict[str, Any]:
    """The output schema of one reading: every field of ``form`` optional.

    A field that takes one of several options is read as every option the words fit (``_one_of_several``): a
    model asked for the one value settles words that fit two by guessing, asked for all of them it names both,
    and ``_values`` then takes the field only when exactly one fits.
    """
    several = _several(form)
    properties = {name: _every_option(prop) if name in several else prop for name, prop in _properties(form).items()}
    return {"type": "object", "title": "Reading", "properties": properties, "additionalProperties": False}


def _properties(form: Mapping[str, Any]) -> Mapping[str, Any]:
    return form.get("properties") or {}


def _several(form: Mapping[str, Any]) -> set[str]:
    """The fields of ``form`` whose value is one of several options (not a list of them, not a single-option field)."""
    return {field.name for field in fields_in(form) if _one_of_several(field)}


def _one_of_several(field: FieldSpec) -> bool:
    return field.kind != "list" and len(field.options or ()) > 1


def _every_option(prop: Mapping[str, Any]) -> dict[str, Any]:
    """The field as a list of its options, keeping what names it to a person."""
    shown = {key: prop[key] for key in ("title", "description") if key in prop}
    return {**shown, "type": "array", "items": {key: value for key, value in prop.items() if key not in shown}}


def _schema(form: Mapping[str, Any]) -> str:
    return json.dumps(form, separators=(",", ":"))


@dataclass(frozen=True)
class ModelInterpreter:
    """pydantic-ai as the interpreter: one run per reading, with the page schema's partial variant as the output type."""

    model: Model | str

    async def answers(self, form: Mapping[str, Any], words: Mapping[str, str]) -> dict[str, Any]:
        asked = fields_schema(form, words)
        if not asked["properties"]:
            return {}
        reading = StructuredDict(reading_schema(asked), name="Reading", description="The values the words meant.")
        agent: Agent[None, Any] = Agent(self.model, output_type=reading, instructions=INSTRUCTIONS)
        prompt = "\n".join(
            [
                f"Fields (JSON schema): {_schema(asked)}",
                "Answers:",
                *(f"- {name}: {words[name]}" for name in asked["properties"]),
            ]
        )
        values = _values((await agent.run(prompt)).output, asked)
        logger.info("Form-fill answers interpreted", asked=list(asked["properties"]), values=values)
        return values


def _values(output: Mapping[str, Any], form: Mapping[str, Any]) -> dict[str, Any]:
    """The fields the reading gave a value, as plain data; nulls, and fields that were not asked, are left out.

    A field read as the options the words fit is a value only when exactly one fits: words that fit none, or
    do not tell several apart, leave it open and the person is asked again.
    """
    several = _several(form)

    def value_of(name: str, value: Any) -> Any:
        if name not in several:
            return value
        fits = value if isinstance(value, list) else [value]
        return fits[0] if len(fits) == 1 else None

    read = {name: value_of(name, value) for name, value in output.items() if name in _properties(form)}
    return {name: value for name, value in read.items() if value is not None}
