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

"""What a widget is: the agent's counterpart of a pydantic-forms component matcher.

The browser renders a field with the first component whose matcher accepts it, and a component such as the
customer select fetches its own options. Core's page schema carries only the ``format`` marker and hints for
those fields; a widget is what the agent knows about one such format — whether a property is one
(``matches``) and the options it has (``options``), fetched from core as the person asking.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from orchestrator_agent.form_fill.skill import CallTool

MAX_INLINE = 10  # at most this many options are asked as chips (LibreChat takes twelve, "Keep default" included)
MAX_FULL_READ = 200  # a long list this short is read in full by the interpreter, so abbreviations still resolve
MAX_CANDIDATES = 50  # a search returns at most this many options
WIDGET_MARK = "x-widget"  # the property key ``enrich`` marks a widget field with


@dataclass(frozen=True)
class Option:
    """One option of a widget field: the value sent to core, how a person sees it, other names they may type."""

    value: str | int
    label: str
    aliases: tuple[str, ...] = ()


class GraphQL(Protocol):
    """A query against core's GraphQL API as the person asking; the ``data`` of the answer."""

    async def __call__(self, query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]: ...


@dataclass(frozen=True)
class WidgetContext:
    """What a widget may use to find its options: core's two APIs, and the form values known so far."""

    call_tool: CallTool
    graphql: GraphQL
    values: Mapping[str, Any] = field(default_factory=dict)


class FieldWidget(Protocol):
    """A format the agent knows: whether a property is one, and its options.

    ``options(search=None)`` is every option; with ``search``, at most ``MAX_CANDIDATES`` the words could
    mean. None means the options depend on a value not known yet: the field waits for it.
    """

    id: str

    def matches(self, field: Mapping[str, Any]) -> bool: ...

    async def options(
        self, field: Mapping[str, Any], ctx: WidgetContext, search: str | None = None
    ) -> Sequence[Option] | None: ...


FieldWidgetExtender = Callable[[list[FieldWidget]], list[FieldWidget]]


class Widget(ABC):
    """A widget whose source returns the whole list: a search narrows what ``fetch`` returned.

    A widget whose source can search on its own (thousands of subscriptions) implements ``options`` instead.
    """

    id: str

    @abstractmethod
    def matches(self, field: Mapping[str, Any]) -> bool: ...

    @abstractmethod
    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None: ...

    async def options(
        self, field: Mapping[str, Any], ctx: WidgetContext, search: str | None = None
    ) -> Sequence[Option] | None:
        fetched = await self.fetch(field, ctx)
        if fetched is None or search is None:
            return fetched
        return narrow_options(fetched, search)


def names_of(option: Option) -> tuple[str, ...]:
    """Every name a person may know an option by: its label, its value, its aliases."""
    return (option.label, str(option.value), *option.aliases)


def _words(text: str) -> set[str]:
    """The words of a text; an identifier such as ``c-3`` is one word."""
    return set(re.findall(r"\w+(?:-\w+)*", text.casefold()))


def _shared_words(option: Option, wanted: set[str]) -> int:
    return len(wanted & set().union(*map(_words, names_of(option))))


def narrow_options(options: Sequence[Option], words: str, limit: int = MAX_CANDIDATES) -> list[Option]:
    """The options the words could mean: those sharing the most whole words first.

    Only when no option shares a whole word are the options containing the words (an abbreviation, half a name)
    the candidates, so ``ut`` is not also every name with "ut" inside.
    """
    needle = words.strip().casefold()
    if not needle:
        return []
    wanted = _words(words)
    shared = {id(option): _shared_words(option, wanted) for option in options}
    by_word = sorted((o for o in options if shared[id(o)]), key=lambda o: -shared[id(o)])
    by_text = [o for o in options if any(needle in name.casefold() for name in names_of(o))]
    return (by_word or by_text)[:limit]


def field_hint(field: Mapping[str, Any], key: str) -> Any:
    """A hint the form put on the field for its component (``extraProperties``, else the older ``uniforms``)."""
    places = (field.get("extraProperties"), field.get("uniforms"))
    return next((hints[key] for hints in places if isinstance(hints, Mapping) and key in hints), None)


__all__ = [
    "MAX_CANDIDATES",
    "MAX_FULL_READ",
    "MAX_INLINE",
    "WIDGET_MARK",
    "FieldWidget",
    "FieldWidgetExtender",
    "GraphQL",
    "Option",
    "Widget",
    "WidgetContext",
    "field_hint",
    "names_of",
    "narrow_options",
]
