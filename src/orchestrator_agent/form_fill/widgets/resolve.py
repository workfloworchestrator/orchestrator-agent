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

"""What a person typed for a long-list field, resolved to one of its options before core sees it.

Core accepts anything for some of these fields (``CustomerId`` is a plain string), so words never travel as a
value: an exact name is its option, otherwise the interpreter reads the words over the candidates and only an
answer that fits exactly one of them is taken. Several fits are offered to pick from; none is asked again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from orchestrator_agent.form_fill.interpret import Chooser
from orchestrator_agent.form_fill.widgets.base import (
    MAX_FULL_READ,
    MAX_INLINE,
    Option,
    WidgetContext,
    names_of,
    narrow_options,
)
from orchestrator_agent.form_fill.widgets.enrich import LongList


@dataclass(frozen=True)
class Resolution:
    """The outcome for one answer: a value, the options it may mean, or the words nothing fits."""

    value: Any = None
    resolved: bool = False
    candidates: tuple[Option, ...] = ()
    unmatched: tuple[str, ...] = ()


def exact_options(options: Sequence[Option], words: str) -> list[Option]:
    """The options one of whose names (label, value, alias) are the words, but for case and outer spaces."""
    wanted = words.strip().casefold()
    return [option for option in options if any(name.casefold() == wanted for name in names_of(option))]


async def _candidates(long_list: LongList, words: str, ctx: WidgetContext, *, full: bool) -> tuple[Option, ...]:
    """What the words are read against: a short enough list in ``full``, else the options a search finds."""
    if len(long_list.options) <= MAX_FULL_READ:
        return long_list.options if full else tuple(narrow_options(long_list.options, words))
    return tuple(await long_list.widget.options(long_list.field, ctx, search=words) or ())


async def resolve_words(long_list: LongList, words: str, ctx: WidgetContext, chooser: Chooser | None) -> Resolution:
    """One item's words: an exact name, else what the chooser says the words fit among the candidates."""
    exact = exact_options(long_list.options, words)
    if len(exact) == 1:
        return Resolution(value=exact[0].value, resolved=True)
    if exact:
        return Resolution(candidates=tuple(exact))
    candidates = await _candidates(long_list, words, ctx, full=chooser is not None)
    if not candidates:
        return Resolution(unmatched=(words,))
    if chooser is None:
        return Resolution(candidates=candidates) if len(candidates) <= MAX_INLINE else Resolution(unmatched=(words,))
    by_value = {option.value: option for option in candidates}
    picked = await chooser.choose(long_list.title, [(o.value, o.label) for o in candidates], words)
    fits = list(dict.fromkeys(value for value in picked if value in by_value))
    match fits:
        case [value]:
            return Resolution(value=value, resolved=True)
        case []:
            return Resolution(unmatched=(words,))
        case _:
            return Resolution(candidates=tuple(by_value[value] for value in fits))


def _items(answer: Any, multiple: bool) -> list[Any]:
    match answer:
        case list():
            return answer
        case str() if multiple:
            return [part.strip() for part in answer.split(",") if part.strip()]
        case _:
            return [answer]


async def resolve_answer(long_list: LongList, answer: Any, ctx: WidgetContext, chooser: Chooser | None) -> Resolution:
    """A field's answer: a value among the options stands; words are resolved; a list only when every item is."""
    values = {option.value for option in long_list.options}
    outcomes = [
        Resolution(value=item, resolved=True)
        if item in values
        else await resolve_words(long_list, str(item), ctx, chooser)
        for item in _items(answer, long_list.multiple)
    ]
    if all(outcome.resolved for outcome in outcomes):
        resolved = [outcome.value for outcome in outcomes]
        return Resolution(value=resolved if long_list.multiple else resolved[0], resolved=True)
    unmatched = tuple(words for outcome in outcomes for words in outcome.unmatched)
    candidates = tuple(option for outcome in outcomes for option in outcome.candidates)
    return Resolution(candidates=candidates, unmatched=unmatched)


def shown_as(long_list: LongList, value: Any) -> Any:
    """How a resolved value is shown: its option's label (each item's, for a list)."""
    labels = {option.value: option.label for option in long_list.options}
    return [labels.get(item, item) for item in value] if isinstance(value, list) else labels.get(value, value)


__all__ = ["Resolution", "exact_options", "resolve_answer", "resolve_words", "shown_as"]
