"""Tests for the form-fill skill: the page tools over core's pages, the approval and the start, the person's decision."""

from __future__ import annotations

import os
from typing import Literal

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from orchestrator.core.schemas.mcp_tools import WorkflowFormPage
from pydantic_ai import ModelRetry
from pydantic_forms.core import FormFieldError

from orchestrator_agent.form_fill import session as fs
from orchestrator_agent.form_fill.model import ACCEPTED, FieldSpec, askable, fields_in
from orchestrator_agent.form_fill.pages import describe, obvious, tool_schema
from orchestrator_agent.form_fill.skill import COMPLETE_NOTE, FormFillSkill
from orchestrator_agent.form_fill.tools import BUDGET_SPENT, MAX_CALLS, PageToolset
from orchestrator_agent.state import Decision, FormFillSession, FormInput, FormReply, Reply, SearchState
from orchestrator_agent.tool_names import (
    CANCEL_FORM_TOOL,
    RESTART_FROM_PAGE_TOOL,
    SEARCH_OPTIONS_TOOL,
    SUBMIT_PAGE_TOOL,
)

from .test_form_widgets import Fetches

PRODUCT = "a19904e6-fa9f-4dd3-90c1-08dbd6d0821e"


# --- pages as core's form tool describes them: one JSON Schema each, as pydantic-forms renders it ------------

Prop = tuple[str, dict, bool]  # a property of a page schema: its name, its schema, whether the page requires it


def field(
    name,
    type_="string",
    *,
    title=None,
    required=True,
    default=None,
    options=None,
    nullable=False,
    format=None,
    read_only=False,
    item=None,
    constraints=None,
    fields=None,
    data=None,
) -> Prop:
    """A field of a page as its schema property; ``options`` as ``{value: label}``, ``fields`` a nested object's."""
    prop = _shape(type_, item, constraints or {}, fields or [])
    if format == "accept":
        prop["enum"] = ["ACCEPTED", "INCOMPLETE"]  # the checkbox's two states; only ACCEPTED passes
    elif options is not None:
        prop["enum"], prop["options"] = list(options), dict(options)
    marks = (
        ("format", format),
        ("readOnly", read_only or None),
        ("uniforms", {"data": data} if data is not None else None),
    )
    prop.update({key: value for key, value in marks if value})
    if nullable:
        prop = {"anyOf": [prop, {"type": "null"}]}
    if shown := title or name.replace("_", " ").title():
        prop["title"] = shown
    if not required:
        prop["default"] = default
    return name, prop, required


def _shape(type_, item, constraints, fields) -> dict:
    """The type of a property, with what a list (its item and size) or an object (its fields) carries."""
    if type_ == "list":
        sizes = {"minItems": constraints.get("min_length"), "maxItems": constraints.get("max_length")}
        items = {"items": item[1]} if item is not None else {}
        return {"type": "array", **items, **{key: value for key, value in sizes.items() if value is not None}}
    if type_ == "object" and fields:
        return {
            "type": "object",
            "properties": {n: s for n, s, _ in fields},
            "required": [n for n, _, r in fields if r],
        }
    return {} if type_ == "any" else {"type": type_}


def choice_list(name, options, **extra) -> Prop:
    """A multi-select of ``options`` (the options are on the list's item)."""
    return field(name, "list", item=field("", required=False, options=options), **extra)


def schema_of(title, fields) -> dict:
    """A page's schema as core sends it: one object, every property inlined."""
    return {
        "type": "object",
        "title": title or "unknown",
        "additionalProperties": False,
        "properties": {name: prop for name, prop, _ in fields},
        "required": [name for name, _, required in fields if required],
    }


def specs(fields) -> list[FieldSpec]:
    """The fields as the skill reads them off a page schema."""
    return fields_in(schema_of(None, fields))


def errors(problems: dict[str, str], type_="value_error") -> list[FormFieldError]:
    """Core's verdict on a page: one error per field, in core's words."""
    return [FormFieldError(loc=[name], msg=msg, type=type_) for name, msg in problems.items()]


def page(index: int, title, fields, verdict=()) -> dict:
    """A page as the tool returns it: the next one to fill, or with ``verdict`` the page core rejected."""
    status: Literal["next", "rejected"] = "rejected" if verdict else "next"
    result = WorkflowFormPage(
        page=index, complete=False, status=status, title=title, schema=schema_of(title, fields), errors=list(verdict)
    )
    return result.model_dump(mode="json", by_alias=True)


def complete(index: int) -> dict:
    return WorkflowFormPage(page=index, complete=True, status="complete").model_dump(mode="json", by_alias=True)


def raw_page(index: int, schema: dict, verdict=(), title=None) -> dict:
    """A page as the tool returns it, from a page schema written out in full (the shape core sends)."""
    status: Literal["next", "rejected"] = "rejected" if verdict else "next"
    shown = title if title is not None else (schema.get("title") if schema.get("title") != "unknown" else None)
    result = WorkflowFormPage(
        page=index, complete=False, status=status, title=shown, schema=schema, errors=list(verdict)
    )
    return result.model_dump(mode="json", by_alias=True)


def missing_in(schema: dict, submitted: dict) -> list[FormFieldError]:
    """Core's verdict on a submitted page that lacks a required field, for a page schema written out in full."""
    lacking = {
        name: "Field required"
        for name in schema.get("required", [])
        if name not in submitted and not (schema["properties"].get(name) or {}).get("readOnly")
    }
    return errors(lacking, "missing")


def rejection(problems: dict[str, str]) -> str:
    """The tool error text fastmcp builds from core's 400 on ``create_workflow``."""
    listed = [{"loc": (name,), "msg": msg, "type": "value_error"} for name, msg in problems.items()]
    return f"HTTP error 400: Bad Request - {{'type': 'FormValidationError', 'validation_errors': {listed!r}, 'status': 400}}"


SPEED = {"1000": "1 Gbit/s", "10000": "10 Gbit/s", "100000": "100 Gbit/s"}
REDUNDANCY = {"protected": "Protected", "unprotected": "Unprotected"}
PRODUCT_PAGE = (None, [field("product", options={PRODUCT: "Demo Lightpath"}, format="productId")])
LIGHTPATH_FIELDS = [
    field("header", required=False, default="Details", format="label", read_only=True),
    field("locked", required=False, default="fixed", read_only=True),
    field("customer_name"),
    field("speed", options=SPEED),
    field("speed_policer", "boolean", required=False, default=False),
]
LIGHTPATH_PAGE = ("Demo Lightpath", LIGHTPATH_FIELDS)
REDUNDANCY_PAGE = (
    "Redundancy and ticket",
    [field("redundancy", options=REDUNDANCY), field("ticket_id", required=False, default="")],
)
TICKET_PAGE = ("Ticket", [field("ticket_id", required=False, default="")])
ACCEPT = field("confirm", format="accept")
ACCEPT_PAGE = ("Dry run", [ACCEPT])
# The last page of a form that ends in core's summary form (``base_summary``): tables to show, nothing to fill.
SUMMARY_TABLE = {
    "columns": [["Dune", "10 Gbit/s", "Protected"]],
    "headers": [],
    "labels": ["customer_name", "speed", "redundancy"],
}
SUMMARY_PAGE = (
    "Demo Lightpath Summary",
    [
        field("divider_1", required=False, nullable=True, format="divider", read_only=True),
        field("product_summary", "object", required=False, format="summary", read_only=True, data=SUMMARY_TABLE),
    ],
)


class FakeCore:
    """Core's three form tools, with the demo lightpath's dynamic generator: 10 Gbit/s+ adds a redundancy page.

    Like core, it does not accept a submitted page that lacks a required field (``Field required`` per field),
    and it answers a rejected page as a result with its verdict (the skill asks for that: ``verdict="result"``).
    """

    def __init__(self, reject=None, accept_page=False, summary_page=False):
        # (n_pages, verdict): once n pages are submitted, reject the last with the verdict (a ``{field: message}``),
        # or raise the text as a tool error when the verdict is a string (a failure that is no verdict at all).
        self.reject = reject
        self.accept_page = accept_page
        self.summary_page = summary_page  # the form ends in the workflow's own summary (core's summary form)
        self.created: list[dict] = []

    # Rows as core's ``list_workflows`` returns them (``WorkflowSchema``).
    WORKFLOWS = [
        {
            "name": "create_demo_lightpath",
            "description": "Create a demo lightpath",
            "target": "CREATE",
            "workflow_id": "0a6c2b8e-8d9c-4b16-9c5e-1e8a2c6d3f01",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "name": "create_nsi_light_path",
            "description": "Create NSI Light Path",
            "target": "CREATE",
            "workflow_id": "0a6c2b8e-8d9c-4b16-9c5e-1e8a2c6d3f02",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "name": "modify_demo_lightpath",
            "description": "Modify a demo lightpath",
            "target": "MODIFY",
            "workflow_id": "0a6c2b8e-8d9c-4b16-9c5e-1e8a2c6d3f03",
            "created_at": "2026-01-01T00:00:00Z",
        },
    ]
    PROCESS_ID = "7c1f0e5a-2b3d-4c6e-8f90-a1b2c3d4e5f6"

    SUB = "9df1beb7-0183-4fb9-8d66-504dfbe85a25"
    OUT_OF_SYNC = "00000000-0000-4000-8000-00000000beef"  # a subscription core will not run a modify on
    NOT_IN_SYNC = "This workflow cannot be started: related subscriptions are not insync"  # core's own message
    SUBSCRIPTION_PAGE = (
        None,
        [field("subscription_id", format="uuid"), field("version", "integer", required=False, nullable=True)],
    )
    NOTE_PAGE = ("Modify note", [field("note", required=False, nullable=True, format="long")])

    @staticmethod
    def missing(fields, submitted: dict) -> list[FormFieldError]:
        """Core's verdict on a submitted page that lacks a required field."""
        lacking = {
            f.name: "Field required" for f in specs(fields) if askable(f) and f.required and f.name not in submitted
        }
        return errors(lacking, "missing")

    def walk(self, pages: list[tuple], inputs: list[dict]) -> dict:
        """The form tool over ``pages``: the page ``inputs`` reach, or the last one rejected for what it lacks."""
        if inputs:
            title, fields = pages[len(inputs) - 1]
            if verdict := self.missing(fields, inputs[-1]):
                return page(len(inputs) - 1, title, fields, verdict)
        if len(inputs) >= len(pages):
            return complete(len(inputs))
        title, fields = pages[len(inputs)]
        return page(len(inputs), title, fields)

    def pages(self, inputs: list[dict]) -> list[tuple]:
        pages = [PRODUCT_PAGE, LIGHTPATH_PAGE]
        if len(inputs) >= 2:
            pages.append(REDUNDANCY_PAGE if inputs[1].get("speed") in ("10000", "100000") else TICKET_PAGE)
        if self.accept_page:
            pages.append(ACCEPT_PAGE)
        if self.summary_page:
            pages.append(SUMMARY_PAGE)
        return pages

    async def __call__(self, name, args):
        if name == "list_workflows":
            return self.WORKFLOWS
        if name == "create_workflow":
            self.created.append(args)
            return {"id": self.PROCESS_ID}  # ``ProcessIdSchema``
        assert name == "get_workflow_form"
        key, inputs = args["workflow_key"], args["page_inputs"]
        if key not in {wf["name"] for wf in self.WORKFLOWS}:
            raise ModelRetry(f"Workflow {key!r} not found")
        if key == "modify_demo_lightpath":
            if inputs and inputs[0].get("subscription_id") == self.OUT_OF_SYNC:  # core's subscription page validator
                title, fields = self.SUBSCRIPTION_PAGE
                return page(0, title, fields, errors({"subscription_id": self.NOT_IN_SYNC}))
            return self.walk([self.SUBSCRIPTION_PAGE, self.NOTE_PAGE], inputs)
        if self.reject and len(inputs) >= self.reject[0]:
            n, verdict = self.reject
            if isinstance(verdict, str):
                raise ModelRetry(verdict)
            title, fields = self.pages(inputs)[n - 1]
            return page(n - 1, title, fields, errors(verdict))
        return self.walk(self.pages(inputs), inputs)


APPROVE, REJECT = Decision.START, Decision.CANCEL  # the person's decision on the start, as the transport hands it over


class Stop:
    """A skill reply as the tests read it: the ``FormReply`` data, plus the approval it stops at."""

    def __init__(self, reply: Reply) -> None:
        self.reply, self.data = reply, FormReply.model_validate_json(reply.text)

    def __getattr__(self, name):
        return getattr(self.data, name)

    @property
    def approval(self):
        return self.reply.approval


def data(reply: Reply | None) -> Stop | None:
    return None if reply is None else Stop(reply)


def verdict(reply: Stop) -> list[tuple]:
    """Core's errors on the reply as ``(field, message, type)``."""
    return [(str(error.loc[0]) if error.loc else "", error.msg, error.type) for error in reply.rejected]


async def opened(core, key, skill=None, **kwargs) -> tuple[SearchState, FormFillSession, dict]:
    """What follows the model's ``start_workflow_form``: the state with its open form, and what the model was told."""
    state = SearchState(hitl=True)
    result = await (skill or FormFillSkill()).open(state, core, key, **kwargs)
    assert state.form_fill is not None
    return state, state.form_fill, result


async def decide(skill, core, state, given: Decision | str | None, *, hitl=True) -> Stop | None:
    """One turn at the approval: the person's decision (``str``: a rejection with those words), or a plain message."""
    state.hitl = hitl
    if given is None:
        state.form_input = None
    elif isinstance(given, Decision):
        state.form_input = FormInput(decision=given)
    else:
        state.form_input = FormInput(decision=REJECT, reason=given)
    return data(await skill.handle(state, core))


FILLED = {"customer_name": "UA", "speed": "10000", "speed_policer": True}

# --- the page as the model sees it ---------------------------------------------------------------------------


class TestToolSchema:
    PAGE = raw_page(
        1,
        schema_of(
            "Details",
            [
                field("header", required=False, default="Details", format="label", read_only=True),
                field("locked", required=False, default="fixed", read_only=True),
                field("customer_name"),
                field("speed", options=SPEED),
                field("vlan", "integer", required=False, nullable=True),
                ACCEPT,
                field("divider", required=False, nullable=True, format="divider", read_only=True),
            ],
        ),
    )

    def test_only_what_the_model_may_fill_is_a_parameter(self):
        schema = tool_schema(WorkflowFormPage.model_validate(self.PAGE))
        assert list(schema["properties"]) == ["customer_name", "speed", "vlan"]  # shown-only and consent are not
        assert schema["required"] == ["customer_name", "speed"] and schema["additionalProperties"] is False
        assert (
            schema["properties"]["speed"]["enum"] == list(SPEED) and schema["properties"]["speed"]["options"] == SPEED
        )
        assert schema["properties"]["vlan"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]

    def test_the_frontends_hints_and_the_source_are_stripped(self):
        prop = {
            "type": "string",
            "title": "Customer",
            "enum": ["c-1"],
            "options": {"c-1": "Alder"},
            "uniforms": {"x": 1},
            "extraProperties": {"y": 2},
            "optionsSource": {"url": "/customers"},
            "x-widget": {"id": "optionsSource"},
        }
        page = WorkflowFormPage(page=0, complete=False, status="next", schema={"properties": {"customer": prop}})
        assert tool_schema(page)["properties"]["customer"] == {
            "type": "string",
            "title": "Customer",
            "enum": ["c-1"],
            "options": {"c-1": "Alder"},
        }

    def test_a_long_list_is_a_value_to_search_for(self):
        long = {
            "type": "string",
            "title": "Customer",
            "optionsSource": {"url": "/c"},
            "x-widget": {"id": "optionsSource", "total": 784},
        }
        later = {
            "type": "integer",
            "title": "Port",
            "description": "A free port.",
            "optionsSource": {"url": "/ports/{node}", "dependsOn": {"node": "/node"}},
            "x-widget": {"id": "optionsSource", "later": True},
        }
        page = WorkflowFormPage(
            page=0,
            complete=False,
            status="next",
            schema={"properties": {"customer": long, "port": later}, "required": ["customer", "port"]},
        )
        schema = tool_schema(page)["properties"]
        assert schema["customer"] == {
            "type": "string",
            "title": "Customer",
            "description": "784 options: find the value with search_options and give one it returned.",
        }
        assert schema["port"]["type"] == "integer" and "depend on node" in schema["port"]["description"]
        assert schema["port"]["description"].startswith("A free port. ")
        described = describe(page)
        assert described["fields"] == {
            "customer": "Customer; required; find the value with search_options",
            "port": "Port; required; find the value with search_options",
        }

    def test_describe_says_each_field_in_one_line(self):
        described = describe(WorkflowFormPage.model_validate(self.PAGE))
        assert described["page"] == 1 and described["title"] == "Details"
        assert described["fields"] == {
            "customer_name": "Customer Name; required; string",
            "speed": "Speed; required; one of 1000 (1 Gbit/s), 10000 (10 Gbit/s), 100000 (100 Gbit/s)",
            "vlan": "Vlan; optional; integer",
        }


class TestObvious:
    def test_consent_is_filled_by_code_and_a_required_single_option_is_taken(self):
        fields = [
            ACCEPT,
            field("product", options={PRODUCT: "Demo Lightpath"}),
            choice_list("node", {"n1": "Node 1"}, constraints={"max_length": 1}),
            field("speed", options=SPEED),
        ]
        page = WorkflowFormPage.model_validate(raw_page(0, schema_of(None, fields)))
        assert obvious(page) == {"confirm": ACCEPTED, "product": PRODUCT, "node": ["n1"]}

    def test_an_optional_or_sourced_single_option_is_the_persons(self):
        sourced = {
            "type": "string",
            "enum": ["p-1"],
            "options": {"p-1": "The one free port"},
            "x-widget": {"id": "optionsSource"},
        }
        schema = {
            "properties": {"port": sourced, **dict([field("ticket", options={"t": "T"}, required=False)[:2]])},
            "required": ["port"],
        }
        page = WorkflowFormPage(page=0, complete=False, status="next", schema=schema)
        assert obvious(page) == {}


# --- the page tools over core --------------------------------------------------------------------------------


class TestOpen:
    async def test_the_form_opens_on_cores_first_page(self):
        core = FakeCore()
        state, session, told = await opened(core, "create_demo_lightpath")
        assert told["status"] == "opened" and told["workflow"] == "create_demo_lightpath"
        assert told["page"]["page"] == 0 and list(told["page"]["fields"]) == ["product"]
        assert session.status == "gathering" and fs.current(session) is not None and session.page_inputs == []

    async def test_a_subscription_passed_by_the_model_is_submitted_by_code_and_core_judges_it(self):
        core = FakeCore()
        _, session, told = await opened(core, "modify_demo_lightpath", subscription_id=FakeCore.SUB)
        assert told["status"] == "accepted" and told["next_page"]["title"] == "Modify note"
        assert session.page_inputs == [{"subscription_id": FakeCore.SUB}]
        _, session, refused = await opened(core, "modify_demo_lightpath", subscription_id=FakeCore.OUT_OF_SYNC)
        assert refused["status"] == "rejected" and refused["errors"][0]["message"] == FakeCore.NOT_IN_SYNC
        assert session.page_inputs == [] and session.draft == {"subscription_id": FakeCore.OUT_OF_SYNC}

    async def test_a_create_workflow_ignores_a_subscription_passed_along(self):
        _, session, told = await opened(FakeCore(), "create_demo_lightpath", subscription_id=FakeCore.SUB)
        assert told["status"] == "opened" and session.page_inputs == []

    async def test_a_key_core_does_not_know_is_refused_and_no_form_opens(self):
        state = SearchState()
        told = await FormFillSkill().open(state, FakeCore(), "create_unicorn")
        assert told["status"] == "refused" and "not found" in told["reason"] and state.form_fill is None

    async def test_the_workflow_catalogue_is_fetched_once(self):
        calls = []

        async def counting(name, args):
            calls.append(name)
            return await FakeCore()(name, args)

        skill = FormFillSkill()
        first = await skill.workflows(counting)
        assert first["create_demo_lightpath"] == "Create a demo lightpath" and "modify_demo_lightpath" in first
        assert await skill.workflows(counting) == first and calls == ["list_workflows"]


class TestSubmit:
    async def test_an_accepted_page_advances_the_form_and_its_values_are_kept(self):
        core = FakeCore()
        state, session, _ = await opened(core, "create_demo_lightpath")
        skill = FormFillSkill()
        told = await skill.submit(session, core, {"product": PRODUCT})
        assert told["status"] == "accepted" and told["next_page"]["title"] == "Demo Lightpath"
        assert session.page_inputs == [{"product": PRODUCT}] and fs.current(session).title == "Demo Lightpath"
        told = await skill.submit(session, core, FILLED)
        assert told["status"] == "accepted" and told["next_page"]["title"] == "Redundancy and ticket"  # 10G: redundancy
        told = await skill.submit(session, core, {"redundancy": "protected"})
        assert told == {"status": "complete", "note": COMPLETE_NOTE}
        assert session.status == "confirming" and fs.current(session) is None
        assert fs.given(session) == {"product": PRODUCT, **FILLED, "redundancy": "protected"}

    async def test_a_rejected_page_stays_current_with_cores_verdict_and_the_draft(self):
        core = FakeCore()
        _, session, _ = await opened(core, "create_demo_lightpath")
        skill = FormFillSkill()
        await skill.submit(session, core, {"product": PRODUCT})
        told = await skill.submit(session, core, {"customer_name": "UA"})
        assert told["status"] == "rejected" and told["page"] == 1
        assert told["errors"] == [{"field": "speed", "message": "Field required"}]
        assert session.page_inputs == [{"product": PRODUCT}] and session.draft == {"customer_name": "UA"}
        assert [e.loc for e in session.rejected] == [["speed"]]
        told = await skill.submit(session, core, {"customer_name": "UA", "speed": "1000"})
        assert told["status"] == "accepted" and session.draft == {} and session.rejected == []

    async def test_what_the_model_sends_for_fields_the_page_lacks_is_dropped_and_said(self):
        core = FakeCore()
        _, session, _ = await opened(core, "create_demo_lightpath")
        told = await FormFillSkill().submit(session, core, {"product": PRODUCT, "colour": "blue", "header": "x"})
        assert told["status"] == "accepted" and told["ignored"] == ["colour", "header"]
        assert session.page_inputs == [{"product": PRODUCT}]

    async def test_consent_is_filled_by_code_so_the_approval_is_the_persons(self):
        core = FakeCore(accept_page=True)
        _, session, _ = await opened(core, "create_demo_lightpath")
        skill = FormFillSkill()
        await skill.submit(session, core, {"product": PRODUCT})
        await skill.submit(session, core, {**FILLED, "speed": "1000"})  # 1G: the ticket page
        told = await skill.submit(session, core, {})
        assert (
            told["status"] == "accepted"
            and told["next_page"]["title"] == "Dry run"
            and told["next_page"]["fields"] == {}
        )
        told = await skill.submit(session, core, {})  # nothing for the model to fill: code consents
        assert told["status"] == "complete" and session.page_inputs[-1] == {"confirm": ACCEPTED}

    async def test_a_searched_fields_value_must_come_from_a_search(self):
        customers = [{"value": "c-1", "label": "Alder (UA)"}]
        customers += [{"value": f"c-{n}", "label": f"Customer {n}"} for n in range(2, 14)]
        core = SourcedCore(customers)
        skill = FormFillSkill(fetch=Fetches({"/customers": customers}))
        _, session, told = await opened(core, "create_sourced", skill)
        assert "search_options" in told["page"]["fields"]["customer_id"]
        strange = await skill.submit(session, core, {"customer_id": "c-1"})
        assert strange["status"] == "rejected" and core.submitted == []  # refused here, never sent
        assert "search_options" in strange["errors"][0]["message"]
        found = await skill.search(session, core, "customer_id", "alder")
        assert found == {
            "field": "customer_id",
            "options": [{"value": "c-1", "label": "Alder (UA)"}],
            "truncated": False,
        }
        accepted = await skill.submit(session, core, {"customer_id": "c-1"})
        assert accepted["status"] == "complete" and core.submitted == [[{"customer_id": "c-1"}]]
        assert session.found == {"customer_id": {"c-1": "Alder (UA)"}}

    async def test_a_short_sourced_list_is_in_the_schema_and_its_values_stand(self):
        customers = [{"value": "c-1", "label": "Alder (UA)"}, {"value": "c-2", "label": "Birch (UB)"}]
        core = SourcedCore(customers)
        skill = FormFillSkill(fetch=Fetches({"/customers": customers}))
        _, session, told = await opened(core, "create_sourced", skill)
        assert told["page"]["fields"]["customer_id"] == "Customer; required; one of c-1 (Alder (UA)), c-2 (Birch (UB))"
        assert (await skill.submit(session, core, {"customer_id": "c-3"}))["status"] == "rejected"  # not an option
        assert (await skill.submit(session, core, {"customer_id": "c-2"}))["status"] == "complete"

    async def test_a_refusal_that_is_no_verdict_is_said_and_the_form_stays(self):
        core = FakeCore(reject=(1, "core is having a moment"))
        _, session, _ = await opened(core, "create_demo_lightpath")
        told = await FormFillSkill().submit(session, core, {"product": PRODUCT})
        assert told["status"] == "refused" and told["reason"] == "core is having a moment"
        assert session.status == "gathering" and session.page_inputs == []

    async def test_nothing_is_submitted_once_the_form_is_complete(self):
        core = FakeCore()
        _, session, _ = await opened(core, "modify_demo_lightpath", subscription_id=FakeCore.SUB)
        skill = FormFillSkill()
        assert (await skill.submit(session, core, {}))["status"] == "complete"
        assert "error" in await skill.submit(session, core, {"note": "x"})
        assert "error" in await skill.search(session, core, "note", "x")


class TestSearch:
    async def test_the_pages_own_options_are_narrowed_here(self):
        core = FakeCore()
        _, session, _ = await opened(core, "create_demo_lightpath")
        skill = FormFillSkill()
        await skill.submit(session, core, {"product": PRODUCT})
        found = await skill.search(session, core, "speed", "10 gbit")
        assert (
            found["options"][0] == {"value": "10000", "label": "10 Gbit/s"} and found["total"] == 3
        )  # the best fit first
        assert (await skill.search(session, core, "speed", "100"))["options"] == [
            {"value": "100000", "label": "100 Gbit/s"}
        ]
        assert len((await skill.search(session, core, "speed", ""))["options"]) == 3

    async def test_a_field_without_options_or_not_on_the_page_is_said(self):
        core = FakeCore()
        _, session, _ = await opened(core, "create_demo_lightpath")
        skill = FormFillSkill()
        await skill.submit(session, core, {"product": PRODUCT})
        assert (await skill.search(session, core, "customer_name", "x"))["note"].startswith(
            "Customer Name has no option list"
        )
        missing = await skill.search(session, core, "colour", "x")
        assert missing["error"].startswith("No field 'colour'") and "customer_name" in missing["error"]
        assert "error" in await skill.search(session, core, "header", "x")  # shown-only

    async def test_a_source_that_waits_for_another_value_says_so(self):
        source = {"url": "/ports/{node}", "dependsOn": {"node": "/node"}, "value": "value", "label": "label"}
        ports = {"type": "integer", "title": "Port", "optionsSource": source}
        node = field("node")
        core = SourcedCore([], extra={"node": node[1], "port": ports}, required=["node", "port"])
        skill = FormFillSkill(fetch=Fetches({"/ports/n1": [{"value": 7, "label": "Port 7"}]}))
        _, session, told = await opened(core, "create_sourced", skill)
        assert "depend on node" in told["page"]["fields"]["port"] or "search_options" in told["page"]["fields"]["port"]
        waiting = await skill.search(session, core, "port", "7")
        assert waiting["error"].startswith("The options of Port depend on another field")
        rejected = await skill.submit(session, core, {"node": "n1"})  # port missing: core rejects, the draft keeps node
        assert rejected["status"] == "rejected"
        found = await skill.search(session, core, "port", "7")
        assert found["options"] == [{"value": 7, "label": "Port 7"}]
        assert (await skill.submit(session, core, {"node": "n1", "port": 7}))["status"] == "complete"

    async def test_an_unreachable_source_is_said_not_raised(self):
        core = SourcedCore([{"value": "c-1", "label": "A"}] * 20)

        async def down(url):
            raise RuntimeError("connection refused")

        skill = FormFillSkill(fetch=down)
        _, session, told = await opened(core, "create_sourced", skill)
        assert told["page"]["fields"]["customer_id"].endswith("find the value with search_options")
        assert "could not be fetched" in (await skill.search(session, core, "customer_id", "a"))["error"]
        refused = await skill.submit(session, core, {"customer_id": "made-up"})  # never free text: nothing was found
        assert refused["status"] == "rejected" and core.submitted == []


class TestRestart:
    async def test_an_earlier_page_is_current_again_and_later_pages_are_regenerated(self):
        core = FakeCore()
        _, session, _ = await opened(core, "create_demo_lightpath")
        skill = FormFillSkill()
        await skill.submit(session, core, {"product": PRODUCT})
        await skill.submit(session, core, FILLED)
        told = await skill.restart(session, core, 1)
        assert told["status"] == "restarted" and told["page"]["title"] == "Demo Lightpath"
        assert session.page_inputs == [{"product": PRODUCT}] and session.status == "gathering"
        told = await skill.submit(session, core, {**FILLED, "speed": "1000"})
        assert told["next_page"]["title"] == "Ticket"  # 1G: no redundancy page

    async def test_a_page_that_is_not_accepted_cannot_be_restarted(self):
        core = FakeCore()
        _, session, _ = await opened(core, "create_demo_lightpath")
        skill = FormFillSkill()
        assert "error" in await skill.restart(session, core, 0)  # nothing accepted yet
        await skill.submit(session, core, {"product": PRODUCT})
        assert "error" in await skill.restart(session, core, 1) and "error" in await skill.restart(session, core, "x")
        assert (await skill.restart(session, core, 0))["status"] == "restarted"


# --- the summary, the start, the decision ----------------------------------------------------------------------


async def completed(core=None, **core_kwargs) -> tuple[FakeCore, SearchState, FormFillSession, FormFillSkill]:
    """A form every page of which core accepted: a 10 Gbit/s lightpath with redundancy."""
    core = core or FakeCore(**core_kwargs)
    skill = FormFillSkill()
    state, session, _ = await opened(core, "create_demo_lightpath", skill)
    await skill.submit(session, core, {"product": PRODUCT})
    await skill.submit(session, core, FILLED)
    told = await skill.submit(session, core, {"redundancy": "protected"})
    while told["status"] == "accepted":  # the optional extra pages (consent, the summary) take nothing
        told = await skill.submit(session, core, {})
    assert told["status"] == "complete"
    return core, state, session, skill


class TestSummaryAndStart:
    async def test_the_summary_carries_the_values_with_their_labels_and_the_defaults(self):
        _, _, session, skill = await completed()
        summary = Stop(skill.summary(session))
        assert summary.status == "confirming"
        assert summary.values == {"product": PRODUCT, **FILLED, "redundancy": "protected"}
        assert summary.labels == {"product": "Demo Lightpath", "speed": "10 Gbit/s", "redundancy": "Protected"}
        assert summary.defaults == {"ticket_id": ""}
        assert summary.approval.tool_name == "create_workflow"
        assert (
            summary.approval.args["json_data"] == session.page_inputs
            and summary.approval.args["labels"] == summary.labels
        )

    async def test_a_searched_value_is_labelled_as_the_search_labelled_it(self):
        customers = [{"value": f"c-{n}", "label": f"Customer {n}"} for n in range(12)]
        core = SourcedCore(customers)
        skill = FormFillSkill(fetch=Fetches({"/customers": customers}))
        _, session, _ = await opened(core, "create_sourced", skill)
        await skill.search(session, core, "customer_id", "customer 7")
        await skill.submit(session, core, {"customer_id": "c-7"})
        assert Stop(skill.summary(session)).labels == {"customer_id": "Customer 7"}

    async def test_the_workflows_own_summary_page_is_carried_to_the_approval(self):
        _, _, session, skill = await completed(summary_page=True)
        assert Stop(skill.summary(session)).summary == [SUMMARY_TABLE]

    async def test_the_start_sends_exactly_the_accepted_pages_once(self):
        core, state, session, skill = await completed()
        started = await decide(skill, core, state, APPROVE)
        assert started is None  # no approval is pending: the transport has not shown one yet
        fs.paused(session, pending_approval("req-1"), unseen=False)
        started = await decide(skill, core, state, APPROVE)
        assert started.status == "started" and started.process_id == FakeCore.PROCESS_ID
        assert core.created == [{"workflow_key": "create_demo_lightpath", "json_data": session.page_inputs}]
        assert state.form_fill is None

    async def test_a_rejection_cancels_and_a_rejection_with_words_reopens_the_form(self):
        core, state, session, skill = await completed()
        fs.paused(session, pending_approval("req-1"), unseen=False)
        assert (await decide(skill, core, state, REJECT)).status == "cancelled" and state.form_fill is None
        core, state, session, skill = await completed()
        fs.paused(session, pending_approval("req-1"), unseen=False)
        assert await decide(skill, core, state, "make it 1G") is None  # the model's turn
        assert session.status == "gathering" and session.reason == "make it 1G" and session.pending is None
        assert fs.current(session) is None and session.page_inputs  # every page still accepted: restart to change one

    async def test_a_message_instead_of_a_decision_reopens_the_form_unless_the_stop_was_not_shown(self):
        core, state, session, skill = await completed()
        fs.paused(session, pending_approval("req-1"), unseen=True)
        again = await decide(skill, core, state, None)
        assert again.status == "confirming" and session.status == "confirming"  # shown again, nothing else
        fs.paused(session, pending_approval("req-2"), unseen=False)
        assert await decide(skill, core, state, None) is None and session.status == "gathering"

    async def test_an_unreadable_response_shows_the_approval_again(self):
        core, state, session, skill = await completed()
        fs.paused(session, pending_approval("req-1"), unseen=False)
        state.form_input = FormInput()
        again = data(await skill.handle(state, core))
        assert again.status == "confirming" and again.approval is not None

    async def test_a_start_core_refuses_reopens_the_page_it_rejects_with_its_words(self):
        core, state, session, skill = await completed()
        fs.paused(session, pending_approval("req-1"), unseen=False)

        async def refusing(name, args):
            if name == "create_workflow":
                raise ModelRetry(rejection({"speed": "Speed no longer offered"}))
            if name == "get_workflow_form" and len(args["page_inputs"]) == 3:
                return page(1, "Demo Lightpath", LIGHTPATH_FIELDS, errors({"speed": "Speed no longer offered"}))
            return await core(name, args)

        reopened = await decide(skill, refusing, state, APPROVE)
        assert reopened.status == "gathering" and reopened.page == 1 and reopened.title == "Demo Lightpath"
        assert verdict(reopened) == [("speed", "Speed no longer offered", "value_error")]
        assert session.status == "gathering" and session.page_inputs == [{"product": PRODUCT}]
        assert session.draft == FILLED and fs.current(session).title == "Demo Lightpath"

    async def test_a_start_that_fails_without_a_verdict_closes_the_form(self):
        core, state, session, skill = await completed()
        fs.paused(session, pending_approval("req-1"), unseen=False)

        async def failing(name, args):
            if name == "create_workflow":
                raise RuntimeError("core unreachable")
            return await core(name, args)

        failed = await decide(skill, failing, state, APPROVE)
        assert failed.status == "failed" and failed.reason == "core unreachable" and state.form_fill is None

    async def test_a_start_core_answered_is_never_sent_again_whatever_the_answer_looks_like(self):
        core, state, session, skill = await completed()
        fs.paused(session, pending_approval("req-1"), unseen=False)

        async def odd(name, args):
            return {"weird": True} if name == "create_workflow" else await core(name, args)

        started = await decide(skill, odd, state, APPROVE)
        assert started.status == "started" and started.process_id is None and "weird" in started.reason
        assert state.form_fill is None

    async def test_a_caller_that_cannot_approve_gets_the_form_prepared_not_started(self):
        core, _, session, skill = await completed()
        prepared = Stop(skill.prepared(session))
        assert (
            prepared.status == "prepared"
            and prepared.values["speed"] == "10000"
            and prepared.labels["speed"] == "10 Gbit/s"
        )
        assert session.status == "done" and core.created == []

    async def test_session_survives_a_json_round_trip(self):
        core, _, session, skill = await completed()
        await skill.search(session, core, "speed", "1")
        restored = FormFillSession.model_validate_json(session.model_dump_json())
        assert restored == session and fs.given(restored) == fs.given(session)


# --- the toolset: the tools are the state --------------------------------------------------------------------


class _Ctx:
    """What a toolset reads off the run context: the state."""

    def __init__(self, state: SearchState) -> None:
        self.deps = SimpleNamespace(state=state)


def names(tools: dict) -> set[str]:
    return set(tools)


class TestPageToolset:
    async def test_the_tools_follow_the_form(self):
        core = FakeCore()
        skill = FormFillSkill()
        toolset = PageToolset(skill, core)
        state = SearchState()
        assert await toolset.get_tools(_Ctx(state)) == {}  # no form: no tools (the handoff is another toolset's)
        state, session, _ = await opened(core, "create_demo_lightpath", skill)
        tools = await toolset.get_tools(_Ctx(state))
        assert names(tools) == {SUBMIT_PAGE_TOOL, SEARCH_OPTIONS_TOOL, CANCEL_FORM_TOOL}
        assert list(tools[SUBMIT_PAGE_TOOL].tool_def.parameters_json_schema["properties"]) == ["product"]
        await skill.submit(session, core, {"product": PRODUCT})
        tools = await toolset.get_tools(_Ctx(state))
        assert names(tools) == {SUBMIT_PAGE_TOOL, SEARCH_OPTIONS_TOOL, RESTART_FROM_PAGE_TOOL, CANCEL_FORM_TOOL}
        assert set(tools[SUBMIT_PAGE_TOOL].tool_def.parameters_json_schema["properties"]) == {
            "customer_name",
            "speed",
            "speed_policer",
        }
        await skill.submit(session, core, FILLED)
        await skill.submit(session, core, {"redundancy": "protected"})
        assert names(await toolset.get_tools(_Ctx(state))) == {RESTART_FROM_PAGE_TOOL, CANCEL_FORM_TOOL}  # complete
        fs.close(session)
        assert await toolset.get_tools(_Ctx(state)) == {}

    async def test_the_budget_of_a_page_is_spent_by_calls_without_progress(self):
        core = FakeCore()
        skill = FormFillSkill()
        toolset = PageToolset(skill, core)
        state, session, _ = await opened(core, "create_demo_lightpath", skill)
        ctx = _Ctx(state)
        tool = (await toolset.get_tools(ctx))[SEARCH_OPTIONS_TOOL]
        for _ in range(MAX_CALLS):
            assert "error" not in await toolset.call_tool(
                SEARCH_OPTIONS_TOOL, {"field": "product", "words": ""}, ctx, tool
            )
        spent = await toolset.call_tool(SUBMIT_PAGE_TOOL, {"product": PRODUCT}, ctx, tool)
        assert spent == {"error": BUDGET_SPENT} and session.page_inputs == []
        assert (await toolset.call_tool(CANCEL_FORM_TOOL, {}, ctx, tool))["status"] == "cancelled"  # always possible


class SourcedCore:
    """A one-page form whose customer field says where its options come from (``optionsSource``)."""

    def __init__(self, customers, extra=None, required=None):
        self.customers = customers
        self.submitted: list[list[dict]] = []
        self.extra = extra or {}
        self.required = required or ["customer_id"]

    def _schema(self) -> dict:
        customer = {
            "type": "string",
            "title": "Customer",
            "optionsSource": {"url": "/customers", "value": "value", "label": "label"},
        }
        properties = {"customer_id": customer, **self.extra} if "customer_id" in self.required else dict(self.extra)
        return {
            "type": "object",
            "title": "Sourced",
            "properties": properties,
            "required": self.required,
            "additionalProperties": False,
        }

    async def __call__(self, name, args):
        if name == "list_workflows":
            return [{**FakeCore.WORKFLOWS[0], "name": "create_sourced"}]
        if name == "create_workflow":
            return {"id": FakeCore.PROCESS_ID}
        assert name == "get_workflow_form"
        inputs = args["page_inputs"]
        if not inputs:
            return raw_page(0, self._schema(), title="Sourced")
        self.submitted.append(inputs)
        if lacking := missing_in(self._schema(), inputs[0]):
            return raw_page(0, self._schema(), lacking, title="Sourced")
        return complete(1)


from types import SimpleNamespace  # noqa: E402

from orchestrator_agent.form_fill.pending import pending_approval  # noqa: E402
