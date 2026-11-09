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

"""What the agent knows about a form field whose options core leaves to a client.

The browser renders such a field with a component that fetches its own options; here the field's ``optionsSource``
says where they are (``widgets.source``), fetched as the person asking. This module holds what every part shares:
an option, the context a fetch runs in, and how a long list is narrowed to the options some words could mean.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from orchestrator_agent.form_fill.skill import CallTool

MAX_INLINE = 10  # at most this many options are written into the page; a longer list is searched
MAX_CANDIDATES = 50  # a search returns at most this many options


@dataclass(frozen=True)
class Option:
    """One option of a widget field: the value sent to core, how a person sees it, other names they may type."""

    value: str | int
    label: str
    aliases: tuple[str, ...] = ()


class Fetch(Protocol):
    """GET a JSON document from core's API as the person asking, by the URL a field's source wrote."""

    async def __call__(self, url: str) -> Any: ...


@dataclass(frozen=True)
class WidgetContext:
    """What a widget may use to find its options: core's tools and API, and the form values known so far."""

    call_tool: CallTool
    fetch: Fetch
    values: Mapping[str, Any] = field(default_factory=dict)


def names_of(option: Option) -> tuple[str, ...]:
    """Every name a person may know an option by: its label, its value, its aliases."""
    return (option.label, str(option.value), *option.aliases)


def _words(text: str) -> set[str]:
    """The words of a text: each plain word and each hyphenated compound, so spaced and hyphenated spellings meet."""
    folded = text.casefold()
    return {*re.findall(r"\w+", folded), *re.findall(r"\w+(?:-\w+)+", folded)}


def _shared_words(option: Option, wanted: set[str]) -> int:
    """How many wanted words the option has; its value is one whole word, so "c-3" shares nothing with "c-1"."""
    known = {str(option.value).casefold()}.union(*map(_words, (option.label, *option.aliases)))
    return len(wanted & known)


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


__all__ = [
    "MAX_CANDIDATES",
    "MAX_INLINE",
    "Fetch",
    "Option",
    "WidgetContext",
    "names_of",
    "narrow_options",
]
