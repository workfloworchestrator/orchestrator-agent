"""The widget: a field that says where its options come from (``optionsSource``, pydantic-forms 2.10).

The field's schema says it all: the URL, how the response is read (``items``, ``value``, ``label``), the query
parameter that searches on the server (``search``), and for each ``{name}`` in the URL a pointer into the page's
values (``dependsOn``). The widget fills the URL, fetches it as the person asking, and reads the options; nothing
here knows a ``format``. Core and every deployment describe such fields the same way, so this is the only widget.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

import httpx
from pydantic_forms.utils.pointer import parse_pointer
from pydantic_forms.utils.schema import DEPENDS_ON, OPTIONS_SOURCE

from orchestrator_agent.form_fill.widgets.base import Option, Widget, WidgetContext
from orchestrator_agent.mcp_client import _ContextVarBearerAuth


class SourceError(Exception):
    """A source cannot be used: a host that is not allowed, a pointer this widget cannot follow, a response of another shape."""


def api_base(mcp_url: str, configured: str | None = None) -> str:
    """Where a relative source URL starts: as configured, else the origin of core's MCP endpoint."""
    if configured:
        return configured.rstrip("/")
    parts = urlsplit(mcp_url)
    return f"{parts.scheme}://{parts.netloc}"


class CoreApi:
    """A ``Fetch`` (``widgets.base``) over HTTP, authenticated like the agent's MCP calls.

    A relative URL is resolved against core's base. An absolute one is fetched only from core's host or a host in
    ``allowed_hosts``, so a form cannot send the person's token anywhere it likes. One client serves every fetch;
    the app closes it (``aclose``) when it stops.
    """

    def __init__(
        self,
        base: str,
        *,
        allowed_hosts: Iterable[str] = (),
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base = base.rstrip("/")
        self.hosts = frozenset({urlsplit(self.base).netloc, *allowed_hosts})
        self._client = httpx.AsyncClient(auth=_ContextVarBearerAuth(), transport=transport, timeout=timeout)

    def resolve(self, url: str) -> str:
        """The absolute URL a source URL stands for; refused when its host is neither core's nor allowed."""
        target = urljoin(self.base + "/", url)
        if (host := urlsplit(target).netloc) not in self.hosts:
            raise SourceError(f"{host} is neither core nor an allowed host for a field's options")
        return target

    async def __call__(self, url: str) -> Any:
        response = await self._client.get(self.resolve(url))
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        """Close the client and its connections."""
        await self._client.aclose()


class OptionsSourceWidget(Widget):
    """Any field with an ``optionsSource``: its options are what the URL returns, read as the source says.

    A field whose URL names a value the page does not have yet has no options yet (None): it waits for that
    value, the way ``enrich`` marks it. A source that declares ``search`` is searched on the server; any other
    list is fetched whole and narrowed here.
    """

    id = "optionsSource"

    def matches(self, field: Mapping[str, Any]) -> bool:
        return isinstance(field.get(OPTIONS_SOURCE), Mapping)

    def can_search(self, field: Mapping[str, Any]) -> bool:
        return bool(field[OPTIONS_SOURCE].get("search"))

    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None:
        source = field[OPTIONS_SOURCE]
        url = filled_url(source, ctx.values)
        return None if url is None else options_in(source, await ctx.fetch(url))

    async def options(
        self, field: Mapping[str, Any], ctx: WidgetContext, search: str | None = None
    ) -> Sequence[Option] | None:
        source = field[OPTIONS_SOURCE]
        if search is None or not source.get("search"):
            return await super().options(field, ctx, search)
        url = filled_url(source, ctx.values)
        return None if url is None else options_in(source, await ctx.fetch(searched(url, source["search"], search)))


# --- the request -------------------------------------------------------------------------------------------------


def filled_url(source: Mapping[str, Any], values: Mapping[str, Any]) -> str | None:
    """The source's URL with each ``{name}`` filled from the page's values, URL-encoded; None while one has none yet."""
    pointers: Mapping[str, str] = source.get(DEPENDS_ON) or {}
    found = {name: value_at(values, pointer) for name, pointer in pointers.items()}
    if any(value in (None, "", []) for value in found.values()):
        return None
    return str(source["url"]).format_map({name: quote(_text(value), safe="") for name, value in found.items()})


def searched(url: str, param: str, words: str) -> str:
    """The URL with the search parameter appended, after the query string it already has or starting one."""
    return f"{url}{'&' if '?' in url else '?'}{quote(param, safe='')}={quote(words, safe='')}"


def value_at(values: Mapping[str, Any], pointer: str) -> Any:
    """The page value a ``dependsOn`` pointer points at; None when it has none yet.

    The widget knows the values of the page's own fields, so ``/name`` and ``1/name`` both read the field
    ``name``. A pointer into the field's own value (``0/...``) or above the page cannot be followed here: an error,
    so the field is asked as text rather than waiting for ever.
    """
    parsed = parse_pointer(pointer)
    if parsed is None:
        raise SourceError(f"{pointer!r} is not a JSON pointer")
    if parsed.levels_up not in (None, 1):
        raise SourceError(f"{pointer!r} points outside the page's fields, which is all this widget knows")
    return _descend(values, parsed.segments)


def _descend(node: Any, segments: Sequence[str]) -> Any:
    for segment in segments:
        match node:
            case Mapping():
                node = node.get(segment)
            case list() if segment.isdigit() and int(segment) < len(node):
                node = node[int(segment)]
            case _:
                return None
    return node


def _text(value: Any) -> str:
    """A value as it goes into a URL: a string as it is, anything else as JSON (``true``, ``10000``)."""
    return value if isinstance(value, str) else json.dumps(value)


# --- the response ------------------------------------------------------------------------------------------------


def options_in(source: Mapping[str, Any], document: Any) -> list[Option]:
    """The options a response holds: the list at ``items`` (or the response itself), each row as ``value`` and ``label`` say."""
    rows = document if (pointer := source.get("items")) is None else _at(document, str(pointer))
    if not isinstance(rows, list):
        raise SourceError(f"the response of {source['url']} holds no list where the source expects one")
    return [_option(source, row) for row in rows]


def _at(document: Any, pointer: str) -> Any:
    if (parsed := parse_pointer(pointer)) is None or parsed.levels_up is not None:
        raise SourceError(f"items {pointer!r} is not a JSON pointer into the response")
    return _descend(document, parsed.segments)


def _option(source: Mapping[str, Any], row: Any) -> Option:
    value_key, label = source.get("value"), source.get("label")
    if (value_key is not None or label is not None) and not isinstance(row, Mapping):
        raise SourceError(f"the response of {source['url']} lists {type(row).__name__}s, not objects")
    value = row if value_key is None else row.get(value_key)
    match label:
        case None:
            shown = str(value)
        case str() if "{" in label:
            shown = label.format_map(row)
        case _:
            shown = str(row.get(label))
    return Option(value if isinstance(value, (str, int)) and not isinstance(value, bool) else str(value), shown)


__all__ = [
    "CoreApi",
    "OptionsSourceWidget",
    "SourceError",
    "api_base",
    "filled_url",
    "options_in",
    "searched",
    "value_at",
]
