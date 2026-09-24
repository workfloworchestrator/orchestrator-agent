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

"""A person's answers that core rejected, interpreted into the values the fields expect — one call per page.

Over A2A the caller is an agent: it reads the stop, which lists every field with its allowed values and
shape, and sends exactly that. Nothing here runs for it. Through kagent's human-in-the-loop extension the
caller is a person and the answers are forwarded verbatim: a chip is a label the adapter maps back to its
value, but free text for a typed or structured field ("vlan twenty", "Jan Jansen, jan@example.org") is not
a form value. Core rejects it; the skill then asks an ``Interpreter`` once, for all rejected fields of the
page at once, and core decides again.

The seam is one protocol call — the rejected fields plus the person's words for each, in; the values, out —
so any engine that answers a page of typed questions in one call can sit behind it. Here it is pydantic-ai
with an output model typed per field (``ModelInterpreter``); the Jev branch puts its typed decision engine
behind the same protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import structlog
from pydantic import BaseModel, ConfigDict, create_model
from pydantic_ai import Agent
from pydantic_ai.models import Model

from orchestrator_agent.form_fill.contract import describe_field
from orchestrator_agent.state import FormField

logger = structlog.get_logger(__name__)

INSTRUCTIONS = (
    "Form fields expect values and a person answered each in their own words. Translate every answer into the "
    "value its field expects, typed exactly as the field's allowed values and shape say: a number written in "
    "words or with a unit is that number, a label or a description of an option is that option's value, details "
    "of items are the objects the shape describes. Leave a field null only when the person said nothing usable "
    "for it. Never add what an answer does not support."
)


class Interpreter(Protocol):
    """One call per page: the rejected fields and the person's words for each -> the values the fields expect.

    A field the words say nothing usable about is left out of the result.
    """

    async def __call__(self, fields: Sequence[FormField], answers: Mapping[str, str]) -> Mapping[str, Any]: ...


def value_type(field: FormField) -> Any:
    """The Python type of the value a field expects: its allowed values, a scalar, or the shape of its items."""
    options = tuple(field.options or ())
    match field.kind:
        case "choice":
            return _literal(options) if options else str
        case "multi":
            return list[_literal(options)] if options else list[str]  # type: ignore[misc]
        case "boolean":
            return bool
        case "integer":
            return int
        case "number":
            return float
        case "json":
            return list[dict[str, Any]] | dict[str, Any]
    return str


def output_type(fields: Sequence[FormField]) -> type[BaseModel]:
    """A pydantic output model with one optional attribute per field, typed as that field's value."""
    attributes: dict[str, Any] = {f.name: (value_type(f) | None, None) for f in fields}
    return create_model("Interpretation", __config__=ConfigDict(protected_namespaces=()), **attributes)


def _literal(values: tuple[str, ...]) -> Any:
    """``Literal`` of runtime values (the field's allowed ones); a static checker cannot type a dynamic Literal."""
    return Literal.__getitem__(values)


def _as_submitted(field: FormField, value: Any) -> Any:
    return [value] if field.as_list and not isinstance(value, list) else value


@dataclass(frozen=True)
class ModelInterpreter:
    """pydantic-ai as the interpreter: one run per page, with an output model typed per rejected field."""

    model: Model | str

    async def __call__(self, fields: Sequence[FormField], answers: Mapping[str, str]) -> dict[str, Any]:
        asked = [f for f in fields if f.name in answers]
        if not asked:
            return {}
        agent: Agent[None, Any] = Agent(self.model, output_type=output_type(asked), instructions=INSTRUCTIONS)
        prompt = "\n\n".join(f"Field: {describe_field(f)}\nAnswer: {answers[f.name]}" for f in asked)
        output = (await agent.run(prompt)).output
        values = {f.name: _as_submitted(f, value) for f in asked if (value := getattr(output, f.name)) is not None}
        logger.info("Form-fill answers interpreted", asked=[f.name for f in asked], values=values)
        return values


__all__ = ["INSTRUCTIONS", "Interpreter", "ModelInterpreter", "output_type", "value_type"]
