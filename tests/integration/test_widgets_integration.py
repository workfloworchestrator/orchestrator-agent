"""The options-source reader and the page tools against a real orchestrator-core: its routes, its MCP tools, its forms."""

from __future__ import annotations

from orchestrator_agent.form_fill.model import fields_in, options_of
from orchestrator_agent.form_fill.skill import FormFillSkill
from orchestrator_agent.form_fill.widgets import CoreApi, source_options
from orchestrator_agent.form_fill.widgets.enrich import enrich
from orchestrator_agent.state import SearchState

P1, P2 = "d8a3f9b0-1111-4000-8000-000000000001", "d8a3f9b0-1111-4000-8000-000000000002"
KEY = "widget_demo"
# As the demo core's fields declare them (tests/integration/core/demo.py).
CUSTOMER = {
    "type": "string",
    "optionsSource": {"url": "/api/demo/customers", "value": "customer_id", "label": "{fullname} ({shortcode})"},
}
PRODUCT = {"type": "string", "optionsSource": {"url": "/api/products/", "value": "product_id", "label": "name"}}


async def test_customers_come_from_the_route_the_field_names(ctx):
    options = await source_options(CUSTOMER, ctx)
    assert len(options) == 30 and options[0].value == "cust-00" and options[0].label == "Customer 00 (C00)"


async def test_searching_the_customers(ctx):
    assert (await source_options(CUSTOMER, ctx, search="Testaccount"))[0].value == "cust-07"
    assert (await source_options(CUSTOMER, ctx, search="C12"))[0].value == "cust-12"
    assert await source_options(CUSTOMER, ctx, search="nobody") == []


async def test_products_come_from_cores_own_route(ctx):
    options = await source_options(PRODUCT, ctx)
    assert {"Widget Port 10G", "Widget Port 1G"} <= {option.label for option in options}


async def test_the_real_page_is_enriched(core, ctx):
    page = await core.direct_call_tool("get_workflow_form", {"workflow_key": KEY, "page_inputs": []})
    enriched = await enrich(page["schema"], ctx)
    fields = {field.name: field for field in fields_in(enriched.schema)}
    assert {P1, P2} <= {str(o.value) for o in options_of(fields["product_id"]) or ()}  # a few products: in the schema
    assert fields["customer_id"].widget == {"id": "optionsSource", "total": 30}  # thirty customers: searched
    assert set(enriched.choices) == {"customer_id", "customers", "product_id"}  # every sourced field with options
    assert fields["note"].widget is None


async def test_a_searched_value_walks_to_the_approval_and_core_runs_it(core, fetch):
    """The page tools over the real core: search, submit with a found value, approve, start."""
    state, skill = SearchState(), FormFillSkill(fetch=fetch)
    state.hitl = True
    opened = await skill.open(state, core.direct_call_tool, KEY)
    assert opened["status"] == "opened" and "customer_id" in opened["page"]["fields"]
    session = state.form_fill
    found = await skill.search(session, core.direct_call_tool, "customer_id", "testaccount")
    assert [o["value"] for o in found["options"]] == ["cust-07"]
    strange = await skill.submit(session, core.direct_call_tool, {"customer_id": "cust-99", "product_id": P1})
    assert strange["status"] == "rejected" and strange["errors"][0]["field"] == "customer_id"  # not found by a search
    outcome = await skill.submit(session, core.direct_call_tool, {"customer_id": "cust-07", "product_id": P1})
    assert outcome["status"] in ("accepted", "complete")
    while outcome["status"] == "accepted":  # any further page: submit it empty, core's defaults apply
        outcome = await skill.submit(session, core.direct_call_tool, {})
    assert outcome["status"] == "complete"
    summary = skill.summary(session)
    assert summary.approval is not None and "cust-07" in summary.text and "Testaccount (TA)" in summary.text
    started = await skill.start(session, core.direct_call_tool)
    assert '"status":"started"' in started.text and "process_id" in started.text


async def test_an_unreachable_source_leaves_the_field_to_the_person(core):
    unreachable = CoreApi("http://127.0.0.1:9", timeout=2)
    state, skill = SearchState(), FormFillSkill(fetch=unreachable)
    opened = await skill.open(state, core.direct_call_tool, KEY)
    assert "customer_id" in opened["page"]["fields"]  # the field is asked without its options
    found = await skill.search(state.form_fill, core.direct_call_tool, "customer_id", "x")
    assert "note" in found or "error" in found
