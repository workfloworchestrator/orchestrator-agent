"""The built-in widgets and the skill against a real orchestrator-core: its GraphQL schema, its MCP tools, its forms."""

from __future__ import annotations

import pytest

from orchestrator_agent.form_fill.core_bridge import choices, page_model, widget_mark
from orchestrator_agent.form_fill.skill import FormFillSkill
from orchestrator_agent.form_fill.widgets import BUILTIN_WIDGETS, CoreGraphQL, CustomerIdWidget, ProductIdWidget
from orchestrator_agent.form_fill.widgets.enrich import enrich
from orchestrator_agent.state import Decision, SearchState

from ..test_form_fill import open_form, turn

P1, P2 = "d8a3f9b0-1111-4000-8000-000000000001", "d8a3f9b0-1111-4000-8000-000000000002"
KEY = "widget_demo"


class Chooser:
    def __init__(self, picks):
        self.picks, self.calls = picks, []

    async def choose(self, title, options, words):
        self.calls.append([value for value, _ in options])
        return self.picks


class Refuses:
    async def choose(self, title, options, words):
        raise AssertionError("no model call expected")


def skill(graphql, choose=None) -> FormFillSkill:
    return FormFillSkill(widgets=list(BUILTIN_WIDGETS), graphql=graphql, choose=choose)


async def test_customers_come_from_cores_graphql_schema(ctx):
    options = await CustomerIdWidget().options({"type": "string", "format": "customerId"}, ctx)
    assert len(options) == 30 and options[0].value == "cust-00" and options[0].label == "Customer 00 (C00)"


@pytest.mark.parametrize(
    "words,first",
    [
        pytest.param("Testaccount", "cust-07", id="name"),
        pytest.param("C12", "cust-12", id="shortcode"),
        pytest.param("customer 02", "cust-02", id="most-words-first"),
        pytest.param("nobody", None, id="nothing"),
    ],
)
async def test_searching_the_customers(ctx, words, first):
    options = await CustomerIdWidget().options({"type": "string", "format": "customerId"}, ctx, search=words)
    assert (options[0].value if options else None) == first


async def test_products_come_from_list_products_restricted_to_the_fields_ids(ctx):
    field = {"type": "string", "format": "productId", "extraProperties": {"productIds": [P1, P2]}}
    options = await ProductIdWidget().options(field, ctx)
    assert sorted(option.label for option in options) == ["Widget Port 10G", "Widget Port 1G"]


async def test_the_real_page_is_enriched(core, ctx):
    page = await core.direct_call_tool("get_workflow_form", {"workflow_key": KEY, "page_inputs": []})
    enriched = await enrich(page["schema"], BUILTIN_WIDGETS, ctx)
    fields = page_model(enriched.schema).model_fields
    assert sorted(choices(fields["product_id"])) == sorted([P1, P2])  # two products: chips
    assert widget_mark(fields["customer_id"]) == {"id": "customerId", "total": 30}  # thirty customers: typed
    assert set(enriched.choices) == {"customer_id", "customers", "product_id"}  # every widget field with options
    assert {name for name, info in fields.items() if "total" in (widget_mark(info) or {})} == {
        "customer_id",
        "customers",
    }
    assert widget_mark(fields["note"]) is None


async def test_an_exact_name_walks_to_the_approval_and_core_runs_it(core, graphql):
    state, form = SearchState(), skill(graphql, Refuses())
    reply = await open_form(form, core.direct_call_tool, state, KEY, {"customer_id": "testaccount", "product_id": P1})
    assert reply.status == "confirming" and reply.values["customer_id"] == "cust-07"
    assert reply.labels["customer_id"] == "Testaccount (TA)" and reply.labels["product_id"] == "Widget Port 10G"
    started = await turn(form, core.direct_call_tool, state, Decision.START)  # the human's approval
    assert started.status == "started" and started.process_id


async def test_the_chooser_picks_among_the_candidates(core, graphql):
    chooser = Chooser(["cust-12"])
    answers = {"customer_id": "twelve", "product_id": P1}
    reply = await open_form(skill(graphql, chooser), core.direct_call_tool, SearchState(), KEY, answers)
    assert reply.values["customer_id"] == "cust-12" and len(chooser.calls[0]) == 30  # all thirty: a short list


async def test_several_fits_are_asked_again_as_chips(core, graphql):
    chooser = Chooser(["cust-01", "cust-02"])
    answers = {"customer_id": "one or two", "product_id": P1}
    reply = await open_form(skill(graphql, chooser), core.direct_call_tool, SearchState(), KEY, answers)
    customer = reply.question("customer_id")
    assert reply.status == "gathering" and customer.values == ("cust-01", "cust-02")


async def test_a_list_of_customers_resolves_each_item(core, graphql):
    answers = {"customer_id": "C03", "customers": "testaccount, C05", "product_id": P2}
    reply = await open_form(skill(graphql, Refuses()), core.direct_call_tool, SearchState(), KEY, answers)
    assert reply.values["customers"] == ["cust-07", "cust-05"]


async def test_unreachable_graphql_leaves_customer_free_text(core):
    unreachable = CoreGraphQL("http://127.0.0.1:9/api/graphql", timeout=2)
    reply = await open_form(skill(unreachable), core.direct_call_tool, SearchState(), KEY, {"product_id": P1})
    customer = reply.question("customer_id")
    assert customer.choices == () and customer.hint == ""  # as before widgets: typed, nothing resolved
