"""The one widget: a field's ``optionsSource`` filled from the page's values, fetched as the person asking, read as it says."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from typing import Any

import httpx
import pytest

from orchestrator_agent.form_fill.model import fields_in, options_of
from orchestrator_agent.form_fill.widgets import CoreApi, OptionsSourceWidget, SourceError, WidgetContext, api_base
from orchestrator_agent.form_fill.widgets.enrich import enrich
from orchestrator_agent.form_fill.widgets.source import filled_url, options_in, searched, value_at
from orchestrator_agent.mcp_client import bind_outbound_token

from .test_form_widgets import no_tool

CUSTOMERS = {"url": "/api/customers", "value": "customer_id", "label": "{fullname} ({shortcode})"}
ROWS = [
    {"customer_id": "c-1", "fullname": "Universiteit Alder", "shortcode": "UA"},
    {"customer_id": "c-2", "fullname": "Universiteit Birch", "shortcode": "UB"},
]
PORTS = {
    "url": "/api/nodes/{node}/ports?speed={speed}",
    "value": "id",
    "label": "name",
    "dependsOn": {"node": "1/node", "speed": "/service_speed"},
}
WIDGET = OptionsSourceWidget()


class Fetches:
    """A ``Fetch`` answering from a table of URL -> document, remembering what was asked."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers, self.urls = answers, []

    async def __call__(self, url: str) -> Any:
        self.urls.append(url)
        return self.answers[url]


def ctx(fetch: Fetches, **values: Any) -> WidgetContext:
    return WidgetContext(call_tool=no_tool, fetch=fetch, values=values)


def sourced(source: dict[str, Any], type_: str = "string") -> dict[str, Any]:
    return {"type": type_, "optionsSource": source}


@pytest.mark.parametrize(
    "field,matched",
    [
        pytest.param(sourced({"url": "/x"}), True, id="a-source"),
        pytest.param({"type": "string", "format": "customerId"}, False, id="a-format-alone"),
    ],
)
def test_a_field_with_a_source_is_the_widgets(field, matched):
    assert WIDGET.matches(field) is matched


class TestOnAPage:
    """The widget through ``enrich``: where the keyword may sit on a list, and a dependency between two fields."""

    NODES = {
        "url": "/api/locations/{location}/nodes",
        "value": "subscription_id",
        "label": "description",
        "dependsOn": {"location": "1/location_code"},
    }
    ROWS = [{"subscription_id": "n-1", "description": "Node 1"}, {"subscription_id": "n-2", "description": "Node 2"}]

    @pytest.mark.parametrize(
        "nodes",
        [
            pytest.param(
                {"type": "array", "items": {"type": "string"}, "optionsSource": NODES}, id="the-source-on-the-list"
            ),
            pytest.param(
                {"type": "array", "items": {"type": "string", "optionsSource": NODES}}, id="the-source-on-the-items"
            ),
        ],
    )
    async def test_a_list_field_gets_its_items_options_wherever_the_source_sits(self, nodes):
        page = {"properties": {"location_code": {"type": "string"}, "nodes": nodes}, "required": ["nodes"]}
        fetch = Fetches({"/api/locations/AMS001/nodes": self.ROWS})
        waiting = await enrich(page, [WIDGET], ctx(fetch))  # no location yet: the nodes wait
        assert fields_in(waiting.schema)[1].widget == {"id": "optionsSource", "later": True} and fetch.urls == []
        ready = await enrich(page, [WIDGET], ctx(fetch, location_code="AMS001"))
        field = fields_in(ready.schema)[1]
        assert [(o.value, o.label) for o in options_of(field) or []] == [("n-1", "Node 1"), ("n-2", "Node 2")]
        assert ready.choices["nodes"].multiple is True and fetch.urls == ["/api/locations/AMS001/nodes"]


class TestReadingTheResponse:
    async def test_a_list_of_objects_is_read_as_the_source_says(self):
        options = await WIDGET.options(sourced(CUSTOMERS), ctx(Fetches({"/api/customers": ROWS})))
        assert [(o.value, o.label) for o in options] == [
            ("c-1", "Universiteit Alder (UA)"),
            ("c-2", "Universiteit Birch (UB)"),
        ]

    async def test_a_list_of_scalars_is_its_own_value_and_label(self):
        fetch = Fetches({"/api/locations": ["AMS001", "UTR001"]})
        options = await WIDGET.options(sourced({"url": "/api/locations"}), ctx(fetch))
        assert [(o.value, o.label) for o in options] == [("AMS001", "AMS001"), ("UTR001", "UTR001")]

    async def test_the_list_may_sit_inside_the_response(self):
        fetch = Fetches({"/api/nodes": {"data": {"nodes": [{"id": 7, "name": "n7"}]}}})
        source = {"url": "/api/nodes", "items": "/data/nodes", "value": "id", "label": "name"}
        (option,) = await WIDGET.options(sourced(source, "integer"), ctx(fetch))
        assert (option.value, option.label) == (7, "n7")

    @pytest.mark.parametrize(
        "source,document",
        [
            pytest.param({"url": "/x"}, {"not": "a list"}, id="no-list"),
            pytest.param({"url": "/x", "value": "id"}, ["scalar"], id="scalars-where-objects-are-due"),
            pytest.param({"url": "/x", "items": "nope"}, [], id="items-no-pointer"),
        ],
    )
    def test_a_response_of_another_shape_is_an_error(self, source, document):
        with pytest.raises(SourceError):
            options_in(source, document)


class TestDependencies:
    async def test_the_url_is_filled_from_the_pages_values_url_encoded(self):
        fetch = Fetches({"/api/nodes/UT%20001/ports?speed=10000": [{"id": 1, "name": "p1"}]})
        options = await WIDGET.options(sourced(PORTS, "integer"), ctx(fetch, node="UT 001", service_speed=10000))
        assert [o.value for o in options] == [1] and fetch.urls == ["/api/nodes/UT%20001/ports?speed=10000"]

    async def test_a_field_whose_dependency_has_no_value_yet_waits(self):
        fetch = Fetches({})
        assert await WIDGET.options(sourced(PORTS, "integer"), ctx(fetch, service_speed=10000)) is None
        assert fetch.urls == []

    @pytest.mark.parametrize(
        "value,expected",
        [
            pytest.param(True, "true", id="bool"),
            pytest.param(10, "10", id="int"),
            pytest.param("a/b", "a%2Fb", id="slash"),
        ],
    )
    def test_a_value_goes_into_the_url_as_text(self, value, expected):
        assert filled_url({"url": "/x/{v}", "dependsOn": {"v": "/v"}}, {"v": value}) == f"/x/{expected}"

    def test_a_pointer_reads_into_nested_values(self):
        assert value_at({"row": {"node": 7}}, "/row/node") == 7
        assert value_at({"rows": [{"node": 8}]}, "/rows/0/node") == 8
        assert value_at({"rows": [{"node": 8}]}, "/rows/3/node") is None
        assert value_at({"node": 9}, "1/node") == 9  # the page is one level up from any of its fields

    @pytest.mark.parametrize(
        "pointer",
        [
            pytest.param("0/x", id="into-the-fields-own-value"),
            pytest.param("2/x", id="above-the-page"),
            pytest.param("x", id="no-pointer"),
        ],
    )
    def test_a_pointer_this_widget_cannot_follow_is_an_error(self, pointer):
        with pytest.raises(SourceError):
            value_at({"x": 1}, pointer)


class TestSearching:
    async def test_a_source_that_searches_is_asked_with_the_words(self):
        source = {**CUSTOMERS, "url": "/api/customers?status=active", "search": "q"}
        fetch = Fetches({"/api/customers?status=active&q=alder": ROWS[:1]})
        options = await WIDGET.options(sourced(source), ctx(fetch), search="alder")
        assert [o.value for o in options] == ["c-1"] and WIDGET.can_search(sourced(source))

    async def test_a_source_that_cannot_search_is_narrowed_here(self):
        options = await WIDGET.options(sourced(CUSTOMERS), ctx(Fetches({"/api/customers": ROWS})), search="birch")
        assert [o.value for o in options] == ["c-2"] and not WIDGET.can_search(sourced(CUSTOMERS))

    def test_the_search_parameter_joins_the_query_string(self):
        assert searched("/a", "q", "x y") == "/a?q=x%20y"
        assert searched("/a?s=1", "q", "x") == "/a?s=1&q=x"


class TestCoreApi:
    @pytest.mark.parametrize(
        "mcp,configured,expected",
        [
            pytest.param("http://core:8080/mcp", None, "http://core:8080", id="derived"),
            pytest.param("http://core:8080/mcp/", None, "http://core:8080", id="trailing-slash"),
            pytest.param(
                "http://core:8080/mcp", "https://api.example.org/", "https://api.example.org", id="configured"
            ),
        ],
    )
    def test_the_api_base(self, mcp, configured, expected):
        assert api_base(mcp, configured) == expected

    @staticmethod
    def _api(handler, **kwargs) -> tuple[CoreApi, list[httpx.Request]]:
        seen: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        return CoreApi("http://core:8080", transport=httpx.MockTransport(record), **kwargs), seen

    async def test_a_relative_url_is_fetched_from_core_with_the_persons_token(self):
        api, seen = self._api(lambda request: httpx.Response(200, json=["a"]))
        with bind_outbound_token("tok-1"):
            assert await api("/api/surf/locations") == ["a"]
        assert str(seen[0].url) == "http://core:8080/api/surf/locations"
        assert seen[0].headers["Authorization"] == "Bearer tok-1"

    @pytest.mark.parametrize(
        "url,allowed,ok",
        [
            pytest.param("https://crm.example.org/customers", (), False, id="another-host-refused"),
            pytest.param("https://crm.example.org/customers", ("crm.example.org",), True, id="an-allowed-host"),
            pytest.param("http://core:8080/api/x", (), True, id="cores-own-host"),
        ],
    )
    async def test_an_absolute_url_needs_an_allowed_host(self, url, allowed, ok):
        api, _ = self._api(lambda request: httpx.Response(200, json=[]), allowed_hosts=allowed)
        with bind_outbound_token("tok-1"):
            if ok:
                assert await api(url) == []
            else:
                with pytest.raises(SourceError):
                    await api(url)

    async def test_an_http_error_is_raised(self):
        api, _ = self._api(lambda request: httpx.Response(503))
        with bind_outbound_token("tok-1"), pytest.raises(httpx.HTTPStatusError):
            await api("/api/x")
