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

Over A2A the caller is an agent: it reads the stop, which carries the page model's JSON schema, and sends
exactly that. Nothing here runs for it. Through a chat client (a parent agent relaying the person's
messages) the answers are a person's words, and words are not form values. Core rejects them; the skill
then asks the ``Interpreter`` once, for all rejected fields of the page at once, and core decides again.
Any message that is not the data model — prose from an agent, a person's words — is read the same way
against the current stop, and a decision (start, cancel) is only ever read by the interpreter: no word is
ever matched by this code.

The seam is one protocol call — the form's model plus the person's words, in; the values, out — so any
engine that answers a page of typed questions in one call can sit behind it. Here it is pydantic-ai with
the page model's partial variant as the output type (``ModelInterpreter``); the Jev branch puts its typed
decision engine behind the same protocol.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import structlog
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_ai import Agent
from pydantic_ai.models import Model

from orchestrator_agent.state import Decision

logger = structlog.get_logger(__name__)

INSTRUCTIONS = (
    "Form fields expect values and a person answered in their own words. The fields are given as a JSON schema; "
    "where a field carries labels, they map each allowed value to how it is shown to people. Translate what was "
    "said into the values the fields expect, typed exactly as the schema says: a number written in words or with "
    "a unit is that number, a label or a description of an option is that option's value, details of items are "
    "the objects the schema describes. Leave a field null when nothing usable was said for it. Never add what "
    "the words do not support."
)
DECISION_INSTRUCTIONS = (
    " The message answers what the form just asked (given). It may instead decide about the form itself: `start` "
    "means the person agrees that the workflow be run as summarised, `cancel` that the person abandons the form "
    "and the workflow is not run; read it in the light of what was asked. Set the decision only when the message "
    "states that decision outright and on its own; a message that also gives or changes values decides nothing, "
    "and a remark or a question decides nothing."
)


class Interpretation(BaseModel):
    """What a message said: values for the form, and a decision about it when one was stated outright."""

    values: dict[str, Any] = {}
    decision: Decision | None = None


class Interpreter(Protocol):
    """The two readings a form needs from an engine, one call each; a field the words say nothing usable about is left out.

    ``form`` is the pydantic model of the fields in play (a page's, or every walked page's — see
    ``core_bridge.page_model`` / ``form_model``).

    ``answers``: the person's words per rejected field -> the values they meant.
    ``message``: the caller's whole message -> values and, only when stated outright, a decision; ``asked`` says
    what the form just asked (the workflow, and whether it was a page's values or the confirmation of the start),
    the context the message is read in.
    """

    async def answers(self, form: type[BaseModel], words: Mapping[str, str]) -> Mapping[str, Any]: ...

    async def message(
        self, form: type[BaseModel], text: str, decisions: Sequence[Decision], asked: str
    ) -> Interpretation: ...


def fields_model(form: type[BaseModel], names: Iterable[str]) -> type[BaseModel]:
    """The named fields of ``form`` as a model of their own (what one reading is about)."""
    wanted = set(names)
    picked: dict[str, Any] = {
        name: (info.annotation, info) for name, info in form.model_fields.items() if name in wanted
    }
    return create_model("Fields", __config__=ConfigDict(protected_namespaces=()), **picked)


def reading_type(form: type[BaseModel], decisions: Sequence[Decision] = ()) -> type[BaseModel]:
    """The output model of one reading: every field of ``form`` made optional, plus the decision when one may be made."""
    attributes: dict[str, Any] = {
        name: (
            _optional(info.annotation),
            Field(
                default=None, title=info.title, description=info.description, json_schema_extra=info.json_schema_extra
            ),
        )
        for name, info in form.model_fields.items()
    }
    if decisions:
        attributes["decision"] = (_optional(_literal(tuple(d.value for d in decisions))), None)
    return create_model("Reading", __config__=ConfigDict(protected_namespaces=()), **attributes)


def _literal(values: tuple[str, ...]) -> Any:
    """``Literal`` of runtime values; a static checker cannot type a dynamic Literal."""
    return Literal.__getitem__(values)


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
        values = _values((await agent.run(prompt)).output)
        logger.info("Form-fill answers interpreted", asked=list(asked.model_fields), values=values)
        return values

    async def message(
        self, form: type[BaseModel], text: str, decisions: Sequence[Decision], asked: str
    ) -> Interpretation:
        agent: Agent[None, Any] = Agent(
            self.model, output_type=reading_type(form, decisions), instructions=INSTRUCTIONS + DECISION_INSTRUCTIONS
        )
        prompt = "\n".join(
            [
                f"Asked: {asked}",
                f"Fields (JSON schema): {_schema(form)}",
                "Decisions the message may state: " + ", ".join(d.value for d in decisions),
                f"Message: {text}",
            ]
        )
        output = (await agent.run(prompt)).output
        decision = getattr(output, "decision", None)
        read = Interpretation(values=_values(output), decision=Decision(decision) if decision else None)
        logger.info("Form-fill message interpreted", values=read.values, decision=read.decision)
        return read


def _values(output: BaseModel) -> dict[str, Any]:
    """The fields the reading gave a value, as plain data (a nested model is its object); nulls are left out."""
    return {
        name: value
        for name, value in output.model_dump(mode="json", exclude_unset=True).items()
        if value is not None and name != "decision"
    }


__all__ = [
    "DECISION_INSTRUCTIONS",
    "INSTRUCTIONS",
    "Interpretation",
    "Interpreter",
    "ModelInterpreter",
    "fields_model",
    "reading_type",
]
