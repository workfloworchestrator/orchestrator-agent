"""Tests for the deterministic form-fill skill: every reply is core's data (``FormReply``), with fakes for core and the interpreter."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from pydantic_ai import ModelRetry

from orchestrator_agent.form_fill.core_bridge import (
    choices,
    form_errors,
    is_accept,
    is_list,
    is_single_pick,
    item_bounds,
    item_type,
    labels,
    page_model,
)
from orchestrator_agent.form_fill.interpret import Interpretation
from orchestrator_agent.form_fill.skill import FormFillSkill, questions
from orchestrator_agent.state import Decision, FormFillSession, FormReply, Reply, SearchState, values_in

PRODUCT = "a19904e6-fa9f-4dd3-90c1-08dbd6d0821e"
SPEED = {
    "enum": ["1000", "10000", "100000"],
    "options": {"1000": "1 Gbit/s", "10000": "10 Gbit/s", "100000": "100 Gbit/s"},
    "type": "string",
}
REDUNDANCY = {
    "enum": ["protected", "unprotected"],
    "options": {"protected": "Protected", "unprotected": "Unprotected"},
    "type": "string",
}

PRODUCT_PAGE = {
    "$defs": {"ProductChoice": {"enum": [PRODUCT], "options": {PRODUCT: "Demo Lightpath"}, "type": "string"}},
    "properties": {"product": {"$ref": "#/$defs/ProductChoice", "format": "productId"}},
    "required": ["product"],
    "title": "unknown",
}
LIGHTPATH_PAGE = {
    "$defs": {"Speed": SPEED},
    "properties": {
        "header": {"format": "label", "default": "Details", "type": "string"},
        "locked": {
            "const": "fixed",
            "enum": ["fixed"],
            "default": "fixed",
            "uniforms": {"disabled": True},
            "type": "string",
        },
        "customer_name": {"title": "Customer Name", "type": "string"},
        "speed": {"$ref": "#/$defs/Speed"},
        "speed_policer": {"default": False, "title": "Speed Policer", "type": "boolean"},
    },
    "required": ["customer_name", "speed"],
    "title": "Demo Lightpath",
}
REDUNDANCY_PAGE = {
    "$defs": {"Redundancy": REDUNDANCY},
    "properties": {"redundancy": {"$ref": "#/$defs/Redundancy"}, "ticket_id": {"default": "", "type": "string"}},
    "required": ["redundancy"],
    "title": "Redundancy and ticket",
}
TICKET_PAGE = {"properties": {"ticket_id": {"default": "", "type": "string"}}, "required": [], "title": "Ticket"}
ACCEPT_PAGE = {
    "properties": {"confirm": {"enum": ["ACCEPTED", "INCOMPLETE"], "format": "accept", "type": "string"}},
    "required": ["confirm"],
    "title": "Dry run",
}


def rejection(problems: dict[str, str]) -> str:
    """The tool error text fastmcp builds from core's 400 body, naming each rejected field."""
    errors = ", ".join(
        f"{{'loc': ('{name}',), 'msg': \"{msg}\", 'type': 'value_error'}}" for name, msg in problems.items()
    )
    return f"HTTP error 400: Bad Request - {{'type': 'FormValidationError', 'validation_errors': [{errors}], 'status': 400}}"


class FakeCore:
    """Core's three form tools, with the demo lightpath's dynamic generator: 10 Gbit/s+ adds a redundancy page.

    Like core, it does not accept a submitted page that lacks a required field (``Field required`` per field).
    """

    def __init__(self, reject=None, accept_page=False):
        self.reject = reject  # (page_index, message): raise when that page's inputs are submitted
        self.accept_page = accept_page
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
    SUBSCRIPTION_PAGE = {
        "properties": {
            "subscription_id": {"format": "uuid", "type": "string"},
            "version": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
        },
        "required": ["subscription_id"],
        "title": "unknown",
    }
    NOTE_PAGE = {
        "properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "format": "long"}},
        "required": [],
        "title": "Modify note",
    }

    @staticmethod
    def require(schema: dict, submitted: dict) -> None:
        """Reject a submitted page the way pydantic-forms does when a required field is missing."""
        if missing := [name for name in schema.get("required") or [] if name not in submitted]:
            raise ModelRetry(rejection(dict.fromkeys(missing, "Field required")))

    async def __call__(self, name, args):  # noqa: C901 - a scripted stand-in for four core tools
        if name == "list_workflows":
            return self.WORKFLOWS
        if name == "get_workflow_form" and args["workflow_key"] not in {wf["name"] for wf in self.WORKFLOWS}:
            raise ModelRetry(f"Workflow {args['workflow_key']!r} not found")
        if name == "get_workflow_form" and args["workflow_key"] == "modify_demo_lightpath":
            inputs = args["page_inputs"]
            pages = [self.SUBSCRIPTION_PAGE, self.NOTE_PAGE]
            if inputs:
                self.require(pages[len(inputs) - 1], inputs[-1])
            if inputs and inputs[0].get("subscription_id") == self.OUT_OF_SYNC:  # core's subscription page validator
                raise ModelRetry(rejection({"subscription_id": self.NOT_IN_SYNC}))
            if len(inputs) >= len(pages):
                return {"page": len(inputs), "complete": True, "schema": None}
            return {"page": len(inputs), "complete": False, "schema": pages[len(inputs)]}
        if name == "create_workflow":
            self.created.append(args)
            return {"id": self.PROCESS_ID}  # ``ProcessIdSchema``
        assert name == "get_workflow_form"
        inputs = args["page_inputs"]
        if self.reject and len(inputs) >= self.reject[0]:  # (n_pages, message): reject when n pages are submitted
            raise ModelRetry(self.reject[1])
        pages = [PRODUCT_PAGE, LIGHTPATH_PAGE]
        if len(inputs) >= 2:
            pages.append(REDUNDANCY_PAGE if inputs[1].get("speed") in ("10000", "100000") else TICKET_PAGE)
        if self.accept_page:
            pages.append(ACCEPT_PAGE)
        if inputs:
            self.require(pages[len(inputs) - 1], inputs[-1])
        if len(inputs) >= len(pages):
            return {"page": len(inputs), "complete": True, "schema": None}
        return {"page": len(inputs), "complete": False, "schema": pages[len(inputs)]}


class WordsInterpreter:
    """Stand-in for the model that reads a message: exactly `yes` / `no` / `cancel` decide, nothing else is read."""

    DECISIONS = {"yes": Decision.START, "no": Decision.CANCEL, "cancel": Decision.CANCEL}

    async def answers(self, form, words):
        return {}

    async def message(self, form, text, decisions):
        decision = self.DECISIONS.get(text)
        return Interpretation(values={}, decision=decision if decision in decisions else None)


def make_skill() -> FormFillSkill:
    return FormFillSkill(interpret=WordsInterpreter())


def data(reply: Reply | None) -> FormReply | None:
    """The reply as the caller reads it: one JSON object."""
    return None if reply is None else FormReply.model_validate_json(reply.text)


def rejected(reply: FormReply) -> list[str]:
    """The fields core rejected, in core's order."""
    return [str(error["loc"][0]) for error in reply.rejected if error["loc"]]


async def turn(skill, core, state, text) -> FormReply | None:
    return data(await skill.handle(text, state, core))


async def open_form(skill, core, state, key, text, subscription_id=None) -> FormReply | None:
    """What the executor does after the model called ``start_workflow_form(key, subscription_id)`` on ``text``."""
    values = {"subscription_id": subscription_id} if subscription_id else {}
    state.form_fill = FormFillSession(workflow_key=key, status="opening", request=text, values=values)
    return data(await skill.open(state, core))


# --- the page model ------------------------------------------------------------------------------------


class TestPageModel:
    """Core's browser-oriented page schema becomes one pydantic model per page: the artifact every reply works from."""

    def test_types_required_defaults_and_display_only(self):
        fields = page_model(LIGHTPATH_PAGE).model_fields
        assert list(fields) == ["customer_name", "speed", "speed_policer"]  # display-only and read-only fields left out
        assert fields["customer_name"].annotation is str and fields["customer_name"].is_required()
        assert choices(fields["speed"]) == ("1000", "10000", "100000") and labels(fields["speed"]) == SPEED["options"]
        assert fields["speed_policer"].annotation is bool and fields["speed_policer"].default is False
        assert page_model(LIGHTPATH_PAGE) is page_model(dict(LIGHTPATH_PAGE))  # pure: one model per schema

    def test_product_picker_and_accept(self):
        (product,) = page_model(PRODUCT_PAGE).model_fields.values()
        assert choices(product) == (PRODUCT,) and labels(product) == {PRODUCT: "Demo Lightpath"}
        (confirm,) = page_model(ACCEPT_PAGE).model_fields.values()
        assert is_accept(confirm) and choices(confirm) == ("ACCEPTED",)

    def test_structured_fields_are_nested_models_and_the_schema_carries_them(self):
        schema = {
            "$defs": {
                "PortChoice": {
                    "enum": ["p-1", "p-2"],
                    "options": {"p-1": "ACE SP DT010A", "p-2": "ACE SP ASD002A"},
                    "type": "string",
                },
                "ServicePort": {
                    "properties": {
                        "subscription_id": {"$ref": "#/$defs/PortChoice"},
                        "vlan": {"default": "0", "type": "string"},
                    },
                    "required": ["subscription_id"],
                    "type": "object",
                },
            },
            "properties": {
                "service_ports": {
                    "items": {"$ref": "#/$defs/ServicePort"},
                    "minItems": 2,
                    "maxItems": 2,
                    "type": "array",
                },
                "subscription_id": {"format": "uuid", "type": "string"},
                "note": {"format": "long", "type": "string"},
            },
            "required": ["service_ports", "subscription_id"],
        }
        model = page_model(schema)
        ports = model.model_fields["service_ports"]
        assert is_list(ports) and item_bounds(ports) == (2, 2)
        port = item_type(ports).model_fields
        assert choices(port["subscription_id"]) == ("p-1", "p-2") and port["vlan"].default == "0"
        # An agent reads the page's JSON schema: nested shapes, counts, labels and formats as data.
        json_schema = model.model_json_schema()
        assert json_schema["required"] == ["service_ports", "subscription_id"]
        assert json_schema["properties"]["service_ports"] == {
            "items": {"$ref": "#/$defs/ServicePort"},
            "maxItems": 2,
            "minItems": 2,
            "title": "service_ports",
            "type": "array",
        }
        assert json_schema["$defs"]["ServicePort"]["properties"]["subscription_id"]["labels"] == {
            "p-1": "ACE SP DT010A",
            "p-2": "ACE SP ASD002A",
        }
        assert json_schema["properties"]["subscription_id"]["format"] == "uuid"
        assert (
            json_schema["properties"]["note"]["format"] == "long"
            and json_schema["properties"]["note"]["default"] is None
        )
        # A person gets one question per field: title, whether it is required, core's message when it rejected the answer.
        errors = form_errors(rejection({"service_ports": "Field required", "subscription_id": "Field required"}))
        assert [q.question for q in questions(model, {}, errors)] == [
            "service_ports (`service_ports`, required) — Field required",
            "subscription_id (`subscription_id`, required) — Field required",
            "note (`note`, optional — leave empty to keep the default)",
        ]

    def test_single_select_list_is_a_list_of_one(self):
        schema = {
            "$defs": {"N": {"enum": ["a", "b"], "type": "string"}},
            "properties": {"n": {"items": {"$ref": "#/$defs/N"}, "maxItems": 1, "type": "array"}},
        }
        (field,) = page_model(schema).model_fields.values()
        assert is_list(field) and item_bounds(field) == (None, 1) and choices(field) == ("a", "b")
        assert is_single_pick(field)


FIELDS_PAGE = {
    "$defs": {"Speed": SPEED},
    "properties": {
        "speed": {"$ref": "#/$defs/Speed"},
        "speed_policer": {"type": "boolean"},
        "customer_name": {"type": "string"},
        "vlan": {"type": "integer"},
        "nodes": {
            "items": {"enum": ["a", "b"], "options": {"a": "Node A", "b": "Node B"}, "type": "string"},
            "type": "array",
        },
        "confirm": {"enum": ["ACCEPTED", "INCOMPLETE"], "format": "accept", "type": "string"},
        "header": {"format": "label", "type": "string"},
    },
    "required": ["speed", "customer_name"],
}


class TestParseReply:
    @pytest.mark.parametrize(
        "text, expected",
        [
            pytest.param('{"speed": "10000"}', {"speed": "10000"}, id="value"),
            pytest.param(
                '{"speed": 10000, "speed_policer": true}',
                {"speed": 10000, "speed_policer": True},
                id="json-types-as-sent",
            ),
            pytest.param('{"speed": "10 Gbit/s"}', {"speed": "10 Gbit/s"}, id="a-label-goes-to-core-as-sent"),
            pytest.param('```json\n{"speed": "1000"}\n```', None, id="a-fenced-object-is-not-the-contract"),
            pytest.param(
                '{"customer_name": "Universiteit Twente", "vlan": 12}',
                {"customer_name": "Universiteit Twente", "vlan": 12},
                id="json",
            ),
            pytest.param('{"nodes": ["a", "b"]}', {"nodes": ["a", "b"]}, id="multi"),
            pytest.param('{"confirm": "ACCEPTED"}', {"confirm": "ACCEPTED"}, id="accept"),
            pytest.param(
                '{"vlan": "", "colour": "blue"}', {"vlan": "", "colour": "blue"}, id="as-sent-even-when-unknown"
            ),
            pytest.param("speed: 10000", None, id="a-line-is-not-the-contract"),
            pytest.param('Here you go: {"speed": "10000"}', None, id="json-inside-prose-is-not-the-contract"),
            pytest.param("Sure, protected please", None, id="prose-has-no-answers"),
        ],
    )
    def test_parse(self, text, expected):
        assert values_in(text) == expected

    def test_a_page_takes_its_own_fields_exactly_as_sent(self):
        session = FormFillSession(
            workflow_key="w",
            values={
                "speed": "10 Gbit/s",
                "Speed": "1000",
                "vlan": "",
                "header": "x",
                "colour": "blue",
                "ticket_id": "T-1",
            },
        )
        page = FormFillSkill._see_page(session, page_model(FIELDS_PAGE), page_index=0)
        assert page == {
            "speed": "10 Gbit/s",
            "vlan": "",
        }  # as sent, core judges it; names exact; display-only, absent: not
        assert session.values["ticket_id"] == "T-1"  # a value for a later page waits


class TestCoreErrors:
    def test_cores_errors_are_read_out_of_the_tool_error_text_as_they_are(self):
        raw = (
            "Error calling tool 'get_workflow_form': HTTP error 400: Bad Request - {'type': 'FormValidationError', "
            "'detail': '1 validation error for ModifySubscriptionPage', 'validation_errors': [{'type': 'value_error', "
            "'loc': ['subscription_id'], 'msg': 'This workflow cannot be started: related subscriptions are not insync'}], 'status': 400}"
        )
        (error,) = form_errors(raw)
        assert error["loc"] == ("subscription_id",) and error["type"] == "value_error"
        assert error["msg"] == "This workflow cannot be started: related subscriptions are not insync"
        assert form_errors("plain failure") == []

    def test_a_rejection_core_pins_on_no_field_is_one_free_question(self):
        error = "HTTP error 400: Bad Request - {'validation_errors': [{'loc': (), 'msg': 'ports must differ', 'type': 'value_error'}]}"
        model = page_model(LIGHTPATH_PAGE)
        (free, policer) = questions(model, {"customer_name": "UT", "speed": "1000"}, form_errors(error))
        assert free.name is None and "ports must differ" in free.question
        assert policer.name == "speed_policer"  # still untouched: asked along


# --- the skill --------------------------------------------------------------------------------------------


class TestWalk:
    """The walk after a handoff: values in, pages submitted to core as known, core's verdict out; the summary; the start."""

    VALUES = '{"speed": "10000", "speed_policer": true}'

    async def test_full_flow_ask_answer_confirm_start(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        # 1. product is the only option and taken; page 1 goes to core without a customer name: core says so.
        reply = await open_form(skill, core, state, "create_demo_lightpath", self.VALUES)
        assert reply.status == "gathering" and reply.page == 1 and reply.title == "Demo Lightpath"
        assert rejected(reply) == ["customer_name"] and reply.rejected[0]["msg"] == "Field required"
        assert reply.values == {"product": PRODUCT, "speed": "10000", "speed_policer": True}
        assert reply.schema_["required"] == ["customer_name", "speed"]
        assert state.form_fill.status == "gathering"
        # 2. The caller answers; page 2 (10 Gbit/s -> redundancy) needs a value nobody gave; the ticket waits.
        reply = await turn(skill, core, state, '{"customer_name": "Universiteit Twente", "ticket_id": "JIRA-4821"}')
        assert reply.page == 2 and rejected(reply) == ["redundancy"]
        assert reply.schema_["properties"]["redundancy"]["labels"] == REDUNDANCY["options"]
        assert reply.values["customer_name"] == "Universiteit Twente"
        # 3. The summary lists every page, including the ticket sent early; a start decision starts with exactly that.
        reply = await turn(skill, core, state, '{"redundancy": "protected"}')
        assert reply.status == "confirming" and reply.values["ticket_id"] == "JIRA-4821"
        assert state.form_fill.status == "confirming"
        reply = await turn(skill, core, state, "yes")
        assert reply.status == "started" and reply.process_id == FakeCore.PROCESS_ID and state.form_fill is None
        (created,) = core.created
        assert created["json_data"] == [
            {"product": PRODUCT},
            {"customer_name": "Universiteit Twente", "speed": "10000", "speed_policer": True},
            {"redundancy": "protected", "ticket_id": "JIRA-4821"},
        ]

    async def test_the_summary_carries_the_values_and_the_defaults_that_apply(self):
        core, state = FakeCore(), SearchState()
        reply = await open_form(
            make_skill(),
            core,
            state,
            "create_demo_lightpath",
            '{"customer_name": "UT", "speed": "10000", "redundancy": "protected"}',
        )
        assert reply.status == "confirming"
        assert reply.values == {"product": PRODUCT, "customer_name": "UT", "speed": "10000", "redundancy": "protected"}
        assert reply.defaults == {
            "speed_policer": False,
            "ticket_id": "",
        }  # optional, never set: the form default applies

    async def test_correction_while_confirming_rewalks_and_regenerates_later_pages(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(
            skill,
            core,
            state,
            "create_demo_lightpath",
            '{"customer_name": "UT", "speed": "10000", "speed_policer": true}',
        )
        await turn(skill, core, state, '{"redundancy": "protected"}')
        assert state.form_fill.status == "confirming"
        # 1 Gbit/s has no redundancy page: the earlier answer is simply no longer part of the form, and the
        # ticket-only page was already offered, so it is not asked again: straight to the summary.
        reply = await turn(skill, core, state, '{"speed": "1000"}')
        assert reply.status == "confirming" and reply.values["speed"] == "1000" and "redundancy" not in reply.values
        assert state.form_fill.page_inputs == [
            {"product": PRODUCT},
            {"customer_name": "UT", "speed": "1000", "speed_policer": True},
            {},
        ]

    async def test_cancel_while_gathering(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", self.VALUES)
        reply = await turn(skill, core, state, "cancel")
        assert reply.status == "cancelled" and state.form_fill is None

    async def test_a_reply_that_is_not_the_contract_on_an_optional_only_page_moves_on_with_defaults(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(
            skill, core, state, "modify_demo_lightpath", "change the note on it", subscription_id=FakeCore.SUB
        )
        reply = await turn(skill, core, state, "no thanks, leave it")  # not a JSON object, not a cancel
        assert reply.status == "confirming"  # asked once, never again

    async def test_read_only_single_option_field_is_never_submitted(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", self.VALUES)
        await turn(skill, core, state, '{"customer_name": "UT"}')
        assert "locked" not in state.form_fill.page_inputs[1] and "locked" not in state.form_fill.values

    async def test_a_bare_uuid_does_not_steal_the_subscription_slot(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(
            skill, core, state, "modify_demo_lightpath", "change the note on it", subscription_id=FakeCore.SUB
        )
        other = "11111111-2222-4333-8444-555555555555"
        await turn(skill, core, state, f'{{"customer_id": "{other}"}}')  # a uuid claimed by another field
        assert state.form_fill.values["subscription_id"] == FakeCore.SUB  # not overwritten
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB}

    async def test_the_workflow_catalogue_is_fetched_once(self):
        core, calls = FakeCore(), []

        async def counting(name, args):
            calls.append(name)
            return await core(name, args)

        skill = make_skill()
        assert await skill.workflows(counting) == await skill.workflows(counting)
        assert calls.count("list_workflows") == 1

    async def test_a_form_that_never_completes_fails_and_closes_instead_of_looping(self):
        class EndlessCore(FakeCore):
            async def __call__(self, name, args):
                if name == "get_workflow_form":
                    return {
                        "page": len(args["page_inputs"]),
                        "complete": False,
                        "schema": {"properties": {}, "required": []},
                    }
                return await super().__call__(name, args)

        state = SearchState()
        reply = await open_form(make_skill(), EndlessCore(), state, "create_demo_lightpath", self.VALUES)
        assert reply.status == "failed" and "did not complete" in reply.reason and state.form_fill is None

    async def test_core_rejecting_a_page_is_relayed_as_its_errors_and_the_session_stays_open(self):
        message = "Input should be '1000', '10000' or '100000'"
        core, state = FakeCore(reject=(2, rejection({"speed": message}))), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", self.VALUES)
        reply = await turn(skill, core, state, '{"customer_name": "UT"}')
        assert reply.status == "gathering" and reply.title == "Demo Lightpath" and reply.page == 1
        assert reply.rejected == [{"loc": ("speed",), "msg": message, "type": "value_error"}] and reply.reason is None
        assert reply.values == {
            "product": PRODUCT,
            "customer_name": "UT",
            "speed_policer": True,
        }  # the rejected value is not
        assert state.form_fill.status == "gathering" and state.form_fill.page_inputs == [{"product": PRODUCT}]

    async def test_a_refusal_that_is_no_form_error_travels_as_the_reason(self):
        core, state = FakeCore(reject=(2, "Error calling tool: something else entirely")), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", self.VALUES)
        reply = await turn(skill, core, state, '{"customer_name": "UT"}')
        assert reply.status == "gathering" and reply.rejected == []
        assert reply.reason == "Error calling tool: something else entirely"

    async def test_accept_field_is_asked_and_only_an_explicit_accept_fills_it(self):
        core, state = FakeCore(accept_page=True), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", '{"customer_name": "UT", "speed": "10000"}')
        reply = await turn(skill, core, state, '{"redundancy": "protected"}')
        assert rejected(reply) == ["confirm"]
        assert reply.schema_["properties"]["confirm"] == {
            "const": "ACCEPTED",
            "format": "accept",
            "title": "confirm",
            "type": "string",
        }
        reply = await turn(skill, core, state, '{"confirm": "ACCEPTED"}')
        assert reply.status == "confirming" and state.form_fill.page_inputs[-1] == {"confirm": "ACCEPTED"}

    async def test_session_survives_a_json_round_trip(self):
        core, state = FakeCore(), SearchState()
        await open_form(make_skill(), core, state, "create_demo_lightpath", self.VALUES)
        restored = SearchState.model_validate(state.model_dump(mode="json"))
        assert restored.form_fill == state.form_fill


class TestStructuredReplies:
    """The same stops as questions and approvals, for adapters that render natively (a human-in-the-loop transport)."""

    VALUES = '{"speed": "10000", "speed_policer": true}'

    async def test_a_stop_carries_ask_fields_with_choices(self):
        core, state = FakeCore(), SearchState()
        state.form_fill = FormFillSession(workflow_key="create_demo_lightpath", status="opening", request=self.VALUES)
        reply = await make_skill().open(state, core)
        assert [f.name for f in reply.ask] == ["customer_name"]  # page 1: the only thing left
        assert reply.ask[0].choices == ()
        assert reply.ask[0].question == "Customer Name (`customer_name`, required) — Field required"
        reply = await make_skill().handle('{"customer_name": "UT"}', state, core)
        (redundancy, ticket) = reply.ask
        assert redundancy.name == "redundancy" and redundancy.choices == ("Protected", "Unprotected")  # labels
        assert redundancy.values == ("protected", "unprotected")  # ...the values behind the chips
        assert ticket.name == "ticket_id" and "optional" in ticket.question and ticket.choices == ()
        assert reply.approval is None

    async def test_summary_carries_the_create_call_to_approve(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", '{"customer_name": "UT", "speed": "10000"}')
        reply = await skill.handle('{"redundancy": "protected"}', state, core)
        assert reply.ask is None
        assert reply.approval.tool_name == "create_workflow"
        assert reply.approval.args == {
            "workflow_key": "create_demo_lightpath",
            "json_data": state.form_fill.page_inputs,
        }
        assert reply.approval.hint == "Start workflow `create_demo_lightpath` with the values shown?"

    async def test_an_approval_arrives_as_a_decision(self):
        # Words reach the skill only through the interpreter (a stand-in here); kagent's approval arrives as a decision.
        for text, expect_created, expect_status in (
            ("yes", 1, None),
            ('{"speed": "1000"}', 0, "confirming"),
            ("no", 0, None),
        ):
            core, state = FakeCore(), SearchState()
            skill = make_skill()
            await open_form(skill, core, state, "create_demo_lightpath", '{"customer_name": "UT", "speed": "10000"}')
            await skill.handle('{"redundancy": "protected"}', state, core)  # the ticket was offered on this page
            reply = await skill.handle(text, state, core)
            assert len(core.created) == expect_created
            assert (state.form_fill.status if state.form_fill else None) == expect_status
            if text == "yes":
                assert data(reply).status == "started"
            elif expect_status == "confirming":
                assert data(reply).values["speed"] == "1000" and reply.approval is not None
            else:
                assert data(reply).status == "cancelled"


async def handoff(state, key, subscription_id=None, core=None):
    """The model's ``start_workflow_form`` call, through the checked tool (the catalogue comes from ``core``)."""
    from types import SimpleNamespace

    from orchestrator_agent.form_fill.handoff import build_handoff_toolset
    from orchestrator_agent.tool_names import START_WORKFLOW_FORM_TOOL

    tool = build_handoff_toolset(make_skill(), core or FakeCore()).tools[START_WORKFLOW_FORM_TOOL].function
    return await tool(SimpleNamespace(deps=SimpleNamespace(state=state)), key, subscription_id=subscription_id)


class TestHandoff:
    """The model routes: ``start_workflow_form`` marks the session, ``open`` walks it in the same turn."""

    async def test_the_tool_marks_the_session_and_keeps_the_request(self):
        state = SearchState(user_input="create a lightpath for UT")
        out = await handoff(state, "create_demo_lightpath")
        assert state.form_fill.status == "opening" and state.form_fill.workflow_key == "create_demo_lightpath"
        assert state.form_fill.request == "create a lightpath for UT"
        assert "create_demo_lightpath" in out

    async def test_a_subscription_passed_by_the_model_fills_the_first_page_and_core_judges_it(self):
        core = FakeCore()
        state = SearchState(user_input="change its note")  # the id was said earlier
        await handoff(state, "modify_demo_lightpath", subscription_id=FakeCore.SUB, core=core)
        reply = data(await make_skill().open(state, core))
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB}
        assert reply.status == "gathering" and reply.rejected == [] and list(reply.schema_["properties"]) == ["note"]
        # A subscription core will not run the workflow on: core's own subscription page says so, nothing here pre-checks.
        state = SearchState(user_input="change its note anyway")
        await handoff(state, "modify_demo_lightpath", subscription_id=FakeCore.OUT_OF_SYNC, core=core)
        reply = data(await make_skill().open(state, core))
        assert reply.status == "gathering" and reply.page == 0 and rejected(reply) == ["subscription_id"]
        assert reply.rejected[0]["msg"] == FakeCore.NOT_IN_SYNC and reply.values == {}
        assert state.form_fill.status == "gathering"  # the caller may correct the id, or cancel

    async def test_a_create_workflow_ignores_a_subscription_passed_along(self):
        state = SearchState(user_input="create a new lightpath for ACE")
        await handoff(state, "create_demo_lightpath", subscription_id=FakeCore.SUB)
        reply = data(await make_skill().open(state, FakeCore()))
        assert state.form_fill.status == "gathering" and rejected(reply) == ["customer_name", "speed"]

    async def test_open_walks_the_first_pages_with_nothing_prefilled(self):
        state = SearchState()
        reply = await open_form(make_skill(), FakeCore(), state, "create_demo_lightpath", "create a lightpath for UT")
        assert state.form_fill.status == "gathering"
        assert rejected(reply) == ["customer_name", "speed"]  # nothing read from prose
        assert reply.values == {"product": PRODUCT}  # the single-option product page is still taken
        assert state.form_fill.request == "create a lightpath for UT"  # only the caller's words are kept

    async def test_values_in_the_request_and_the_subscription_the_model_passed_are_taken(self):
        core, state = FakeCore(), SearchState()
        reply = await open_form(
            make_skill(), core, state, "modify_demo_lightpath", "change the note on it", subscription_id=FakeCore.SUB
        )
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB}
        assert reply.page == 1 and reply.rejected == [] and "note" in reply.schema_["properties"]
        state = SearchState()
        reply = await open_form(
            make_skill(), core, state, "create_demo_lightpath", '{"customer_name": "UT", "speed": "10000"}'
        )
        assert reply.page == 2 and rejected(reply) == ["redundancy"]

    async def test_a_key_core_does_not_know_is_cores_refusal_and_the_form_closes(self):
        state = SearchState()  # the handoff tool checks the key; a stale session may still reach core with one
        reply = await open_form(make_skill(), FakeCore(), state, "create_unicorn", "make me a unicorn")
        assert reply.status == "failed" and reply.reason == "Workflow 'create_unicorn' not found"
        assert state.form_fill is None

    async def test_the_tool_rejects_a_key_core_does_not_know_so_the_model_corrects_itself(self):
        state = SearchState(user_input="create a lightpath for UT")
        with pytest.raises(ModelRetry, match="Unknown workflow key 'create_demo_lightpth'"):
            await handoff(state, "create_demo_lightpth")
        assert state.form_fill is None
        await handoff(state, "create_demo_lightpath")
        assert state.form_fill.status == "opening" and state.form_fill.workflow_key == "create_demo_lightpath"

    async def test_a_handoff_never_walked_is_discarded_on_the_next_message(self):
        state = SearchState(form_fill=FormFillSession(workflow_key="create_demo_lightpath", status="opening"))
        assert await turn(make_skill(), FakeCore(), state, "hello") is None
        assert state.form_fill is None


class TestLiteralContract:
    """After the handoff the reply is data (a JSON object) or the interpreter's reading of it; no word is matched."""

    REQUEST = "create a lightpath for Universiteit Twente"

    async def test_without_an_asker_no_message_is_routed_by_the_skill(self):
        state = SearchState()
        assert await turn(make_skill(), FakeCore(), state, "please create a lightpath for UT") is None
        assert (
            await turn(make_skill(), FakeCore(), state, "create_demo_lightpath for UT") is None
        )  # keys are not routing
        assert state.form_fill is None

    async def test_yes_starts_no_cancels_and_anything_else_reshows_the_summary(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", self.REQUEST)
        await turn(skill, core, state, '{"customer_name": "UT", "speed": "10000"}')
        reply = await turn(skill, core, state, '{"redundancy": "protected"}')
        assert reply.status == "confirming"
        reply = await turn(skill, core, state, "looks right to me")  # not literal: no judgement without an asker
        assert reply.status == "confirming" and not core.created
        reply = await turn(skill, core, state, "Yes.")  # not the exact token: the summary again
        assert reply.status == "confirming" and not core.created
        reply = await turn(skill, core, state, "yes")
        assert reply.status == "started" and len(core.created) == 1

    async def test_no_at_confirmation_cancels(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(
            skill,
            core,
            state,
            "create_demo_lightpath",
            '{"customer_name": "UT", "speed": "10000", "redundancy": "protected"}',
        )
        assert state.form_fill.status == "confirming"
        reply = await turn(skill, core, state, "no")
        assert reply.status == "cancelled" and state.form_fill is None and not core.created

    async def test_cancel_mid_form_and_other_prose_is_re_asked(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", self.REQUEST)
        reply = await turn(skill, core, state, "what do you need again?")
        assert rejected(reply) == ["customer_name", "speed"]  # no asker: the same stop, not a guess
        reply = await turn(skill, core, state, "cancel")
        assert reply.status == "cancelled" and state.form_fill is None


class TestReviewRegressions:
    """Scenarios from the final review; each one used to misbehave."""

    async def test_without_an_asker_and_without_a_form_nothing_is_called(self):
        calls = []

        async def counting(name, args):
            calls.append(name)
            return await FakeCore()(name, args)

        assert await make_skill().handle("hello", SearchState(), counting) is None and calls == []
        assert not make_skill().wants(SearchState())
        assert make_skill().wants(SearchState(form_fill=FormFillSession(workflow_key="w")))

    async def test_a_nested_object_is_that_fields_value_not_top_level_answers(self):
        other = "11111111-2222-4333-8444-555555555555"
        pairs = values_in(f'{{"port": {{"subscription_id": "{other}", "vlan": "10"}}}}')
        assert set(pairs) == {"port"}
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(
            skill, core, state, "modify_demo_lightpath", "change the note on it", subscription_id=FakeCore.SUB
        )
        await turn(skill, core, state, f'{{"note": {{"subscription_id": "{other}"}}}}')
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB}  # not clobbered

    async def test_a_failure_mid_walk_leaves_the_last_good_pages(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(
            skill,
            core,
            state,
            "create_demo_lightpath",
            '{"customer_name": "UT", "speed": "10000", "redundancy": "protected"}',
        )
        assert state.form_fill.status == "confirming"
        pages_before = list(state.form_fill.page_inputs)

        class DownError(Exception):
            pass

        async def failing(name, args):
            if name == "get_workflow_form" and len(args["page_inputs"]) == 1:
                raise DownError("core unreachable")
            return await core(name, args)

        with pytest.raises(DownError):
            await skill.handle('{"speed": "1000"}', state, failing)
        assert state.form_fill.page_inputs == pages_before and state.form_fill.status == "confirming"

    async def test_the_models_subscription_beats_another_id_in_the_text(self):
        other = "11111111-2222-4333-8444-555555555555"
        state = SearchState(user_input=f"change its note; it replaces port {other}")
        await handoff(state, "modify_demo_lightpath", subscription_id=FakeCore.SUB)
        await make_skill().open(state, FakeCore())
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB}

    async def test_a_stale_choice_is_reported_by_core_when_the_page_regenerates(self):
        class DependentCore(FakeCore):
            async def __call__(self, name, args):
                if name == "get_workflow_form" and args["workflow_key"] == "create_demo_lightpath":
                    inputs = args["page_inputs"]
                    node = {"properties": {"node": {"enum": ["A", "B"], "type": "string"}}, "required": ["node"]}
                    if not inputs:
                        return {"page": 0, "complete": False, "schema": node}
                    ports = ["A1", "A2"] if inputs[0]["node"] == "A" else ["B1", "B2"]
                    port = {"properties": {"port": {"enum": ports, "type": "string"}}, "required": ["port"]}
                    if len(inputs) >= 2 and inputs[1].get("port") not in ports:  # core validates the value
                        raise ModelRetry(
                            "HTTP error 400: Bad Request - {'validation_errors': [{'loc': ('port',), "
                            f"'msg': \"Input should be {' or '.join(repr(p) for p in ports)}\", 'type': 'literal_error'}}]}}"
                        )
                    if len(inputs) == 1:
                        return {"page": 1, "complete": False, "schema": port}
                    return {"page": 2, "complete": True, "schema": {}}
                return await super().__call__(name, args)

        core, state = DependentCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", '{"node": "A", "port": "A1"}')
        assert state.form_fill.status == "confirming"
        reply = await turn(skill, core, state, '{"node": "B"}')
        # The stale port goes to core as sent; core's own message names the new options, and the form stays open.
        assert reply.rejected == [{"loc": ("port",), "msg": "Input should be 'B1' or 'B2'", "type": "literal_error"}]
        assert reply.schema_["properties"]["port"]["enum"] == ["B1", "B2"]
        assert state.form_fill is not None and state.form_fill.status == "gathering"

    async def test_consent_is_given_per_page(self):
        accept = {
            "properties": {"confirm": {"enum": ["ACCEPTED", "INCOMPLETE"], "format": "accept", "type": "string"}},
            "required": ["confirm"],
        }

        class TwoConsents(FakeCore):
            async def __call__(self, name, args):
                if name == "get_workflow_form" and args["workflow_key"] == "create_demo_lightpath":
                    n = len(args["page_inputs"])
                    if n:
                        self.require(accept, args["page_inputs"][-1])
                    return {"page": n, "complete": n == 2, "schema": {**accept, "title": f"Step {n}"}}
                return await super().__call__(name, args)

        core, state = TwoConsents(), SearchState()
        skill = make_skill()
        reply = await open_form(skill, core, state, "create_demo_lightpath", "go")
        assert reply.title == "Step 0" and rejected(reply) == ["confirm"]
        reply = await turn(skill, core, state, '{"confirm": "ACCEPTED"}')
        assert reply.title == "Step 1" and rejected(reply) == ["confirm"]  # the second consent is asked, not assumed
        reply = await turn(skill, core, state, '{"confirm": "ACCEPTED"}')
        assert reply.status == "confirming"

    async def test_a_rejection_with_an_unparseable_correction_re_walks_instead_of_cancelling(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(
            skill,
            core,
            state,
            "create_demo_lightpath",
            '{"customer_name": "UT", "speed": "10000", "redundancy": "protected"}',
        )
        from orchestrator_agent.adapters.a2a_hitl import resume
        from orchestrator_agent.adapters.kagent_hitl import ToolApproval, ToolApprovalResponse

        state.form_fill.hitl_request = {"id": "r", "kind": "approval"}

        async def rejected_with(
            reason,
        ):  # what the executor does with a kagent rejection: data and decision on the state
            response = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=False, rejection_reason=reason)])
            resumed = resume(state.form_fill, response, "Human input supplied")
            state.form_decision, state.form_values = resumed.decision, resumed.values
            return data(await skill.handle(resumed.text, state, core))

        reply = await rejected_with('{"speed": "1G"}')
        assert reply.status != "cancelled" and state.form_fill is not None  # a correction is walked with
        state.form_decision = state.form_values = None
        await skill.handle('{"speed": "10000"}', state, core)  # a good correction: the summary again
        assert state.form_fill.status == "confirming"
        reply = await rejected_with("not now")
        assert reply.status == "cancelled" and state.form_fill is None

    async def test_a_start_that_fails_without_an_answer_closes_the_form(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(
            skill,
            core,
            state,
            "create_demo_lightpath",
            '{"customer_name": "UT", "speed": "10000", "redundancy": "protected"}',
        )

        async def timing_out(name, args):
            if name == "create_workflow":
                raise TimeoutError("no answer")
            return await core(name, args)

        reply = await turn(skill, timing_out, state, "yes")
        assert reply.status == "failed" and reply.reason == "no answer" and state.form_fill is None  # never retried

    async def test_a_start_core_refuses_reopens_the_form_with_its_errors(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(
            skill,
            core,
            state,
            "create_demo_lightpath",
            '{"customer_name": "UT", "speed": "10000", "redundancy": "protected"}',
        )

        async def refusing(name, args):
            if name == "create_workflow":
                raise ModelRetry(rejection({"customer_name": "Customer no longer exists"}))
            return await core(name, args)

        reply = await turn(skill, refusing, state, "yes")
        assert reply.status == "gathering" and reply.page is None and rejected(reply) == ["customer_name"]
        assert "redundancy" in reply.schema_["properties"] and reply.values["speed"] == "10000"
        assert state.form_fill.status == "gathering"
