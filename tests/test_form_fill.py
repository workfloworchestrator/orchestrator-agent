"""Tests for the deterministic form-fill skill: the caller contract and the turn logic, with fakes for core and the asker."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import uuid

import pytest
from pydantic_ai import ModelRetry

from orchestrator_agent.form_fill.contract import (
    FormCommand,
    command,
    label,
    need_input,
    raw_pairs,
    render_summary,
)
from orchestrator_agent.form_fill.core_bridge import page_fields
from orchestrator_agent.form_fill.skill import FormFillSkill
from orchestrator_agent.state import FormField, FormFillSession, SearchState

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


class FakeCore:
    """Core's three form tools, with the demo lightpath's dynamic generator: 10 Gbit/s+ adds a redundancy page."""

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

    @staticmethod
    def lists(**groups: list[dict]) -> dict:
        """A ``get_subscription_available_workflows`` result in core's shape (``SubscriptionWorkflowListsSchema``)."""
        return {"create": [], "modify": [], "terminate": [], "system": [], "reconcile": [], **groups}

    @staticmethod
    def row(name: str, description: str, target: str) -> dict:
        """One more ``list_workflows`` row in core's shape (``WorkflowSchema``)."""
        return {
            "name": name,
            "description": description,
            "target": target,
            "workflow_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, name)),
            "created_at": "2026-01-01T00:00:00Z",
        }

    SUB = "9df1beb7-0183-4fb9-8d66-504dfbe85a25"
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

    async def __call__(self, name, args):  # noqa: C901 - a scripted stand-in for four core tools
        if name == "list_workflows":
            return self.WORKFLOWS
        if name == "get_subscription_available_workflows":
            if args["subscription_id"] != self.SUB:
                raise ModelRetry("Subscription not found")
            return {  # ``SubscriptionWorkflowListsSchema``
                "create": [],
                "modify": [{"name": "modify_demo_lightpath"}, {"name": "modify_note", "reason": "blocked"}],
                "terminate": [{"name": "terminate_demo_lightpath", "reason": "subscription.not_in_sync"}],
                "system": [],
                "reconcile": [],
            }
        if name == "get_workflow_form" and args["workflow_key"] == "modify_demo_lightpath":
            inputs = args["page_inputs"]
            pages = [self.SUBSCRIPTION_PAGE, self.NOTE_PAGE]
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
        if len(inputs) >= len(pages):
            return {"page": len(inputs), "complete": True, "schema": None}
        return {"page": len(inputs), "complete": False, "schema": pages[len(inputs)]}


# --- contract -----------------------------------------------------------------------------------------


class TestPageFields:
    def test_kinds_required_and_display_only(self):
        by_name = {f.name: f for f in page_fields(LIGHTPATH_PAGE)}
        assert by_name["header"].display_only and by_name["header"].kind == "text"
        assert by_name["customer_name"].kind == "text" and by_name["customer_name"].required
        assert by_name["speed"].kind == "choice" and by_name["speed"].options == SPEED["options"]
        assert by_name["speed_policer"].kind == "boolean" and not by_name["speed_policer"].required

    def test_product_picker_and_accept(self):
        (product,) = page_fields(PRODUCT_PAGE)
        assert product.kind == "choice" and product.options == {PRODUCT: "Demo Lightpath"}
        (confirm,) = page_fields(ACCEPT_PAGE)
        assert confirm.kind == "accept"

    def test_structured_fields_get_a_shape_and_uuid_fields_a_hint(self):
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
        by_name = {f.name: f for f in page_fields(schema)}
        ports = by_name["service_ports"]
        assert ports.kind == "json"
        assert ports.shape == (
            "a JSON list of exactly 2 objects, each with keys `subscription_id` (required: one of `p-1` (ACE SP DT010A), "
            "`p-2` (ACE SP ASD002A)), `vlan` (optional: text, default '0')"
        )
        text = need_input(
            FormFillSession(workflow_key="w"),
            page=0,
            title="t",
            fields=list(by_name.values()),
            values={},
            missing=[ports, by_name["subscription_id"]],
        ).text
        assert "- service_ports (required): a JSON list of exactly 2 objects" in text
        assert "- subscription_id (required): an id (UUID) of the subscription — find it with the search skill" in text
        assert "- note (optional): free text (multi-line allowed)" in text

    def test_single_select_list_is_a_choice_returned_as_list(self):
        schema = {
            "$defs": {"N": {"enum": ["a", "b"], "type": "string"}},
            "properties": {"n": {"items": {"$ref": "#/$defs/N"}, "maxItems": 1, "type": "array"}},
        }
        (field,) = page_fields(schema)
        assert field.kind == "choice" and field.as_list


SPEED_FIELD = FormField(name="speed", title="Speed", kind="choice", required=True, options=SPEED["options"])
FIELDS = {
    "speed": SPEED_FIELD,
    "speed_policer": FormField(name="speed_policer", title="Speed Policer", kind="boolean"),
    "customer_name": FormField(name="customer_name", title="Customer Name", kind="text", required=True),
    "vlan": FormField(name="vlan", title="Vlan", kind="integer"),
    "nodes": FormField(name="nodes", title="Nodes", kind="multi", options={"a": "Node A", "b": "Node B"}),
    "confirm": FormField(name="confirm", title="Confirm", kind="accept"),
    "header": FormField(name="header", title="Header", kind="text", display_only=True),
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
            pytest.param('```json\n{"speed": "1000"}\n```', {"speed": "1000"}, id="fenced"),
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
            pytest.param("speed: 10000", {}, id="a-line-is-not-the-contract"),
            pytest.param('Here you go: {"speed": "10000"}', {}, id="json-inside-prose-is-not-the-contract"),
            pytest.param("Sure, protected please", {}, id="prose-has-no-answers"),
        ],
    )
    def test_parse(self, text, expected):
        assert raw_pairs(text) == expected

    def test_a_page_takes_its_own_fields_as_sent_and_leaves_empty_values_to_their_defaults(self):
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
        page = FormFillSkill._see_page(session, list(FIELDS.values()), page_index=0)
        assert page == {"speed": "10 Gbit/s"}  # as sent (core judges it); names exact; empty, display-only, absent: not
        assert session.values["ticket_id"] == "T-1"  # a value for a later page waits


class TestRendering:
    def test_need_input_lists_missing_optional_filled_and_labels(self):
        session = FormFillSession(workflow_key="create_demo_lightpath", page_inputs=[{"product": PRODUCT}])
        session.fields = {
            "product": FormField(name="product", title="Product", kind="choice", options={PRODUCT: "Demo Lightpath"})
        }
        fields = page_fields(LIGHTPATH_PAGE)
        text = need_input(
            session,
            page=1,
            title="Demo Lightpath",
            fields=fields,
            values={"speed": "10000"},
            missing=[f for f in fields if f.name == "customer_name"],
        ).text
        assert "- customer_name (required): free text" in text
        assert "- speed_policer (optional, default False): `true` or `false`" in text
        assert "header" not in text  # display-only fields are never asked for
        assert f"product: {PRODUCT} (Demo Lightpath)" in text and "speed: 10000" in text
        assert 'Reply with a JSON object keyed by field name, e.g. {"customer_name": "..."}' in text

    def test_summary_lists_defaults_the_caller_never_set_and_renders_structured_values_as_json(self):
        fields = {f.name: f for f in page_fields(LIGHTPATH_PAGE)}
        fields["ports"] = FormField(name="ports", title="Ports", kind="json")
        session = FormFillSession(
            workflow_key="w",
            page_inputs=[
                {"customer_name": "UT", "speed": "10000", "ports": [{"subscription_id": "p-1", "vlan": "10"}]}
            ],
            fields=fields,
        )
        summary = render_summary(session)
        assert '- ports: [{"subscription_id": "p-1", "vlan": "10"}]' in summary
        assert "- speed_policer: False (default)" in summary  # optional, never set: the form default applies
        assert label(None, []) == "(empty)" and label(None, None) == "(empty)"
        assert "header" not in summary  # display-only, never listed

    def test_a_blocked_workflow_says_why_and_what_the_subscription_can_run(self):
        from orchestrator_agent.form_fill.contract import render_blocked

        text = render_blocked("terminate_x", "Terminate X", "subscription.not_in_sync", {"modify_note": "Modify note"})
        assert text.startswith("Cannot start a workflow on this subscription now:")
        assert "- `terminate_x`: Terminate X — cannot run on this subscription now: subscription.not_in_sync" in text
        assert "Workflows this subscription can run now:\n- `modify_note`: Modify note\n" in text
        assert "Workflows this subscription can run now" not in render_blocked("terminate_x", "Terminate X", "why", {})

    def test_summary_lists_the_defaults_of_the_last_walks_fields_the_caller_never_set(self):
        from orchestrator_agent.form_fill.contract import defaults_applying

        ticket = FormField(name="ticket_id", title="Ticket", kind="text", has_default=True, default="")
        session = FormFillSession(workflow_key="w", fields={"ticket_id": ticket}, page_inputs=[{"note": "n"}])
        assert defaults_applying(session) == ["ticket_id: (empty) (default)"]

    def test_summary_shows_labels_and_asks_for_yes(self):
        session = FormFillSession(
            workflow_key="create_demo_lightpath",
            page_inputs=[{"product": PRODUCT}, {"speed": "10000"}],
            fields={"speed": SPEED_FIELD},
        )
        summary = render_summary(session)
        assert "- speed: 10000 (10 Gbit/s)" in summary and "`yes`" in summary


# --- the skill --------------------------------------------------------------------------------------------


async def turn(skill, core, state, text):
    reply = await skill.handle(text, state, core)
    return None if reply is None else reply.text


class TestWalk:
    """The walk after a handoff: values in, pages regenerated by core, the summary, the start."""

    VALUES = '{"speed": "10000", "speed_policer": true}'

    async def test_full_flow_ask_answer_confirm_start(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
        # 1. product is the only option and taken; customer_name is free text -> asked.
        reply = await open_form(skill, core, state, "create_demo_lightpath", self.VALUES)
        assert "page 1" in reply and "- customer_name (required): free text" in reply
        assert "speed: 10000 (10 Gbit/s)" in reply and "speed_policer: True" in reply
        assert state.form_fill.status == "gathering"
        # 2. The caller answers; page 2 (10 Gbit/s -> redundancy) needs a value nobody gave; the ticket waits.
        reply = await turn(skill, core, state, '{"customer_name": "Universiteit Twente", "ticket_id": "JIRA-4821"}')
        assert "page 2" in reply and "- redundancy (required): one of `protected` (Protected)" in reply
        assert "customer_name: Universiteit Twente" in reply
        # 3. The summary lists every page, including the ticket sent early; yes starts with exactly that.
        reply = await turn(skill, core, state, '{"redundancy": "protected"}')
        assert reply.startswith("All pages of workflow `create_demo_lightpath`") and "ticket_id: JIRA-4821" in reply
        assert state.form_fill.status == "confirming"
        reply = await turn(skill, core, state, "yes")
        assert reply.startswith("Started workflow") and state.form_fill is None
        (created,) = core.created
        assert created["json_data"] == [
            {"product": PRODUCT},
            {"customer_name": "Universiteit Twente", "speed": "10000", "speed_policer": True},
            {"redundancy": "protected", "ticket_id": "JIRA-4821"},
        ]

    async def test_correction_while_confirming_rewalks_and_regenerates_later_pages(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
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
        assert "- speed: 1000 (1 Gbit/s)" in reply and "redundancy" not in reply
        assert state.form_fill.page_inputs == [
            {"product": PRODUCT},
            {"customer_name": "UT", "speed": "1000", "speed_policer": True},
            {},
        ]

    async def test_cancel_while_gathering(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
        await open_form(skill, core, state, "create_demo_lightpath", self.VALUES)
        reply = await turn(skill, core, state, "cancel")
        assert reply.startswith("Cancelled") and state.form_fill is None

    async def test_a_reply_that_is_not_the_contract_on_an_optional_only_page_moves_on_with_defaults(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
        await open_form(skill, core, state, "modify_demo_lightpath", f"change the note on {FakeCore.SUB}")
        reply = await turn(skill, core, state, "no thanks, leave it")  # not a JSON object, not a cancel
        assert reply.startswith("All pages of workflow `modify_demo_lightpath`")  # asked once, never again

    async def test_read_only_single_option_field_is_never_submitted(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
        await open_form(skill, core, state, "create_demo_lightpath", self.VALUES)
        await turn(skill, core, state, '{"customer_name": "UT"}')
        assert "locked" not in state.form_fill.page_inputs[1] and "locked" not in state.form_fill.values

    async def test_a_bare_uuid_does_not_steal_the_subscription_slot(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
        await open_form(skill, core, state, "modify_demo_lightpath", f"change the note on {FakeCore.SUB}")
        other = "11111111-2222-4333-8444-555555555555"
        await turn(skill, core, state, f'{{"customer_id": "{other}"}}')  # a uuid claimed by another field
        assert state.form_fill.values["subscription_id"] == FakeCore.SUB  # not overwritten
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB}

    async def test_the_workflow_catalogue_is_fetched_once(self):
        core, state = FakeCore(), SearchState()
        calls = []

        async def counting(name, args):
            calls.append(name)
            return await core(name, args)

        skill = FormFillSkill()
        await open_form(skill, counting, state, "create_demo_lightpath", self.VALUES)
        await turn(skill, counting, state, '{"customer_name": "UT"}')
        assert calls.count("list_workflows") == 1

    async def test_a_form_that_never_completes_is_rejected_not_looped(self):
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
        reply = await open_form(FormFillSkill(), EndlessCore(), state, "create_demo_lightpath", self.VALUES)
        assert reply.startswith("The orchestrator rejected") and "did not complete" in reply

    async def test_core_rejecting_a_page_is_relayed_and_the_session_stays_open(self):
        core, state = FakeCore(reject=(2, "speed: Input should be '1000', '10000' or '100000'")), SearchState()
        skill = FormFillSkill()
        await open_form(skill, core, state, "create_demo_lightpath", self.VALUES)
        reply = await turn(skill, core, state, '{"customer_name": "UT"}')
        assert reply.startswith("The orchestrator rejected") and "speed" in reply
        assert state.form_fill.status == "gathering"

    async def test_accept_field_is_asked_and_only_an_explicit_accept_fills_it(self):
        core, state = FakeCore(accept_page=True), SearchState()
        skill = FormFillSkill()
        await open_form(skill, core, state, "create_demo_lightpath", '{"customer_name": "UT", "speed": "10000"}')
        reply = await turn(skill, core, state, '{"redundancy": "protected"}')
        assert "- confirm (required): `ACCEPTED` once the user has approved this step" in reply
        reply = await turn(skill, core, state, '{"confirm": "ACCEPTED"}')
        assert reply.startswith("All pages of workflow") and state.form_fill.page_inputs[-1] == {"confirm": "ACCEPTED"}

    async def test_session_survives_a_json_round_trip(self):
        core, state = FakeCore(), SearchState()
        await open_form(FormFillSkill(), core, state, "create_demo_lightpath", self.VALUES)
        restored = SearchState.model_validate(state.model_dump(mode="json"))
        assert restored.form_fill == state.form_fill

    def test_rejection_text_is_the_validation_messages(self):
        from orchestrator_agent.form_fill.core_bridge import error_detail

        raw = (
            "Error calling tool 'get_workflow_form': HTTP error 400: Bad Request - {'type': 'FormValidationError', "
            "'detail': '1 validation error for ModifySubscriptionPage', 'validation_errors': [{'type': 'value_error', "
            "'loc': ['subscription_id'], 'msg': 'This workflow cannot be started: related subscriptions are not insync'}], 'status': 400}"
        )
        assert (
            error_detail(raw)
            == "subscription_id: This workflow cannot be started: related subscriptions are not insync"
        )
        assert error_detail("plain failure") == "plain failure"


class TestStructuredReplies:
    """The same stops as data, for adapters that render natively (a human-in-the-loop transport)."""

    VALUES = '{"speed": "10000", "speed_policer": true}'

    async def test_need_input_carries_ask_fields_with_choices(self):
        core, state = FakeCore(), SearchState()
        state.form_fill = FormFillSession(workflow_key="create_demo_lightpath", status="opening", request=self.VALUES)
        reply = await FormFillSkill().open(state, core)
        assert [f.name for f in reply.ask] == ["customer_name"]  # page 1: the only thing left
        assert reply.ask[0].choices == () and "free text" in reply.ask[0].question
        reply = await FormFillSkill().handle('{"customer_name": "UT"}', state, core)
        (redundancy, ticket) = reply.ask
        assert redundancy.name == "redundancy" and redundancy.choices == ("Protected", "Unprotected")  # labels
        assert redundancy.values == ("protected", "unprotected")  # ...the values behind the chips
        assert ticket.name == "ticket_id" and "optional" in ticket.question and ticket.choices == ()
        assert reply.approval is None

    async def test_summary_carries_the_create_call_to_approve(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
        await open_form(skill, core, state, "create_demo_lightpath", '{"customer_name": "UT", "speed": "10000"}')
        reply = await skill.handle('{"redundancy": "protected"}', state, core)
        assert reply.ask is None
        assert reply.approval.tool_name == "create_workflow"
        assert reply.approval.args == {
            "workflow_key": "create_demo_lightpath",
            "json_data": state.form_fill.page_inputs,
        }
        assert reply.approval.hint == reply.text

    async def test_an_approval_arrives_as_the_contract_words(self):
        # A kagent Approve/Reject is mapped to `yes`, a JSON object of corrections or `no` by the adapter; the skill sees text.
        for text, expect_created, expect_status in (
            ("yes", 1, None),
            ('{"speed": "1000"}', 0, "confirming"),
            ("no", 0, None),
        ):
            core, state = FakeCore(), SearchState()
            skill = FormFillSkill()
            await open_form(skill, core, state, "create_demo_lightpath", '{"customer_name": "UT", "speed": "10000"}')
            await skill.handle('{"redundancy": "protected"}', state, core)  # the ticket was offered on this page
            reply = await skill.handle(text, state, core)
            assert len(core.created) == expect_created
            assert (state.form_fill.status if state.form_fill else None) == expect_status
            if text == "yes":
                assert reply.text.startswith("Started workflow")
            elif expect_status == "confirming":
                assert "- speed: 1000 (1 Gbit/s)" in reply.text and reply.approval is not None
            else:
                assert reply.text.startswith("Cancelled")


async def handoff(state, key, subscription_id=None, core=None):
    """The model's ``start_workflow_form`` call, through the checked tool (the catalogue comes from ``core``)."""
    from types import SimpleNamespace

    from orchestrator_agent.form_fill.handoff import build_handoff_toolset
    from orchestrator_agent.tool_names import START_WORKFLOW_FORM_TOOL

    tool = build_handoff_toolset(FormFillSkill(), core or FakeCore()).tools[START_WORKFLOW_FORM_TOOL].function
    return await tool(SimpleNamespace(deps=SimpleNamespace(state=state)), key, subscription_id=subscription_id)


async def open_form(skill, core, state, key, text):
    """What the executor does after the model called ``start_workflow_form(key)`` on ``text``."""
    state.form_fill = FormFillSession(workflow_key=key, status="opening", request=text)
    reply = await skill.open(state, core)
    return None if reply is None else reply.text


class TestHandoff:
    """The model routes: ``start_workflow_form`` marks the session, ``open`` walks it in the same turn."""

    async def test_the_tool_marks_the_session_and_keeps_the_request(self):

        state = SearchState(user_input="create a lightpath for UT")
        out = await handoff(state, "create_demo_lightpath")
        assert state.form_fill.status == "opening" and state.form_fill.workflow_key == "create_demo_lightpath"
        assert state.form_fill.request == "create a lightpath for UT"
        assert "create_demo_lightpath" in out

    async def test_a_subscription_passed_by_the_model_is_checked_and_filled(self):

        core = FakeCore()
        core.WORKFLOWS = core.WORKFLOWS + [
            FakeCore.row("terminate_demo_lightpath", "Terminate a demo lightpath", "TERMINATE")
        ]
        state = SearchState(user_input="start the terminate workflow for it anyway")  # the id was said earlier
        await handoff(state, "terminate_demo_lightpath", subscription_id=FakeCore.SUB, core=core)
        reply = await FormFillSkill().open(state, core)
        assert reply.text.startswith("Cannot start a workflow") and "subscription.not_in_sync" in reply.text
        assert state.form_fill is None
        state = SearchState(user_input="change its note")
        await handoff(state, "modify_demo_lightpath", subscription_id=FakeCore.SUB, core=core)
        reply = await FormFillSkill().open(state, core)
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB} and "- note (optional" in reply.text

    async def test_a_create_workflow_ignores_a_subscription_passed_along(self):

        state = SearchState(user_input="create a new lightpath for ACE")
        await handoff(state, "create_demo_lightpath", subscription_id=FakeCore.SUB)
        reply = await FormFillSkill().open(state, FakeCore())
        assert state.form_fill.status == "gathering" and "- customer_name (required)" in reply.text

    async def test_nothing_runnable_closes_with_the_reason_instead_of_asking(self):
        core, state = FakeCore(), SearchState()
        core.WORKFLOWS = core.WORKFLOWS + [
            FakeCore.row("terminate_demo_lightpath", "Terminate a demo lightpath", "TERMINATE")
        ]

        async def only_blocked(name, args):
            if name == "get_subscription_available_workflows":
                return FakeCore.lists(
                    terminate=[{"name": "terminate_demo_lightpath", "reason": "subscription.not_in_sync"}]
                )
            return await core(name, args)

        reply = await open_form(
            FormFillSkill(), only_blocked, state, "terminate_demo_lightpath", f"terminate {FakeCore.SUB}"
        )
        assert reply.startswith("Cannot start a workflow on this subscription now:")
        assert (
            "`terminate_demo_lightpath`: Terminate a demo lightpath — cannot run on this subscription now: subscription.not_in_sync"
            in reply
        )
        assert state.form_fill is None  # not left in a choosing loop

    async def test_open_walks_the_first_pages_with_nothing_prefilled(self):
        state = SearchState()
        reply = await open_form(
            FormFillSkill(), FakeCore(), state, "create_demo_lightpath", "create a lightpath for UT"
        )
        assert state.form_fill.status == "gathering"
        assert "- customer_name (required)" in reply and "- speed (required)" in reply  # nothing read from prose
        assert "Filled so far: product:" in reply  # the single-option product page is still taken
        assert state.form_fill.request == "create a lightpath for UT"  # only the caller's words are kept

    async def test_values_and_a_subscription_id_in_the_request_are_taken(self):
        core, state = FakeCore(), SearchState()
        reply = await open_form(
            FormFillSkill(), core, state, "modify_demo_lightpath", f"change the note on {FakeCore.SUB}"
        )
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB}
        assert "- note (optional" in reply
        state = SearchState()
        reply = await open_form(
            FormFillSkill(), core, state, "create_demo_lightpath", '{"customer_name": "UT", "speed": "10000"}'
        )
        assert "page 2" in reply and "- redundancy (required)" in reply

    async def test_an_unknown_key_is_dropped_and_the_model_answers(self):
        state = SearchState()
        assert await open_form(FormFillSkill(), FakeCore(), state, "create_unicorn", "make me a unicorn") is None
        assert state.form_fill is None

    async def test_the_tool_rejects_a_key_core_does_not_know_so_the_model_corrects_itself(self):

        state = SearchState(user_input="create a lightpath for UT")
        with pytest.raises(ModelRetry, match="Unknown workflow key 'create_demo_lightpth'"):
            await handoff(state, "create_demo_lightpth")
        assert state.form_fill is None
        await handoff(state, "create_demo_lightpath")
        assert state.form_fill.status == "opening" and state.form_fill.workflow_key == "create_demo_lightpath"

    async def test_a_workflow_core_will_not_run_on_the_subscription_is_reported(self):
        core, state = FakeCore(), SearchState()
        core.WORKFLOWS = core.WORKFLOWS + [
            FakeCore.row("terminate_demo_lightpath", "Terminate a demo lightpath", "TERMINATE")
        ]
        skill = FormFillSkill()
        reply = await open_form(skill, core, state, "terminate_demo_lightpath", f"terminate {FakeCore.SUB}")
        assert reply.startswith("Cannot start a workflow on this subscription now:")
        assert (
            "`terminate_demo_lightpath`: Terminate a demo lightpath — cannot run on this subscription now: subscription.not_in_sync"
            in reply
        )
        assert (
            "Workflows this subscription can run now:" in reply
            and "- `modify_demo_lightpath`: Modify a demo lightpath" in reply
        )
        assert state.form_fill is None  # nothing to choose here: the caller restates its request

    async def test_a_handoff_never_walked_is_discarded_on_the_next_message(self):
        state = SearchState(form_fill=FormFillSession(workflow_key="create_demo_lightpath", status="opening"))
        assert await turn(FormFillSkill(), FakeCore(), state, "hello") is None
        assert state.form_fill is None


class TestLiteralContract:
    """After the handoff, without a decision engine, the contract is literal: a JSON object, yes / no / cancel."""

    REQUEST = "create a lightpath for Universiteit Twente"

    async def test_without_an_asker_no_message_is_routed_by_the_skill(self):
        state = SearchState()
        assert await turn(FormFillSkill(), FakeCore(), state, "please create a lightpath for UT") is None
        assert (
            await turn(FormFillSkill(), FakeCore(), state, "create_demo_lightpath for UT") is None
        )  # keys are not routing
        assert state.form_fill is None

    async def test_yes_starts_no_cancels_and_anything_else_reshows_the_summary(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
        await open_form(skill, core, state, "create_demo_lightpath", self.REQUEST)
        await turn(skill, core, state, '{"customer_name": "UT", "speed": "10000"}')
        reply = await turn(skill, core, state, '{"redundancy": "protected"}')
        assert reply.startswith("All pages of workflow `create_demo_lightpath`")
        reply = await turn(skill, core, state, "looks right to me")  # not literal: no judgement without an asker
        assert reply.startswith("All pages of workflow `create_demo_lightpath`") and not core.created
        reply = await turn(skill, core, state, "Yes.")  # not the exact token: the summary again
        assert reply.startswith("All pages of workflow `create_demo_lightpath`") and not core.created
        reply = await turn(skill, core, state, "yes")
        assert reply.startswith("Started workflow `create_demo_lightpath`") and len(core.created) == 1

    async def test_no_at_confirmation_cancels(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
        await open_form(
            skill,
            core,
            state,
            "create_demo_lightpath",
            '{"customer_name": "UT", "speed": "10000", "redundancy": "protected"}',
        )
        assert state.form_fill.status == "confirming"
        reply = await turn(skill, core, state, "no")
        assert reply.startswith("Cancelled") and state.form_fill is None and not core.created

    async def test_cancel_mid_form_and_other_prose_is_re_asked(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
        await open_form(skill, core, state, "create_demo_lightpath", self.REQUEST)
        reply = await turn(skill, core, state, "what do you need again?")
        assert "- customer_name (required)" in reply  # no asker: the same question, not a guess
        reply = await turn(skill, core, state, "cancel")
        assert reply.startswith("Cancelled") and state.form_fill is None


class TestReviewRegressions:
    """Scenarios from the final review; each one used to misbehave."""

    async def test_without_an_asker_and_without_a_form_nothing_is_called(self):
        calls = []

        async def counting(name, args):
            calls.append(name)
            return await FakeCore()(name, args)

        assert await FormFillSkill().handle("hello", SearchState(), counting) is None and calls == []
        assert not FormFillSkill().wants(SearchState())
        assert FormFillSkill().wants(SearchState(form_fill=FormFillSession(workflow_key="w")))

    async def test_a_nested_object_is_that_fields_value_not_top_level_answers(self):
        other = "11111111-2222-4333-8444-555555555555"
        pairs = raw_pairs(f'{{"port": {{"subscription_id": "{other}", "vlan": "10"}}}}')
        assert set(pairs) == {"port"}
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
        await open_form(skill, core, state, "modify_demo_lightpath", f"change the note on {FakeCore.SUB}")
        await turn(skill, core, state, f'{{"note": {{"subscription_id": "{other}"}}}}')
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB}  # not clobbered

    async def test_a_failure_mid_walk_leaves_the_last_good_pages(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
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
        await FormFillSkill().open(state, FakeCore())
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
        skill = FormFillSkill()
        await open_form(skill, core, state, "create_demo_lightpath", '{"node": "A", "port": "A1"}')
        assert state.form_fill.status == "confirming"
        reply = await turn(skill, core, state, '{"node": "B"}')
        # The stale port goes to core as sent; core's own message names the new options, and the form stays open.
        assert (
            reply.startswith("The orchestrator rejected the values") and "port: Input should be 'B1' or 'B2'" in reply
        )
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
                    return {"page": n, "complete": n == 2, "schema": {**accept, "title": f"Step {n}"}}
                return await super().__call__(name, args)

        core, state = TwoConsents(), SearchState()
        skill = FormFillSkill()
        reply = await open_form(skill, core, state, "create_demo_lightpath", "go")
        assert 'Form "Step 0"' in reply
        reply = await turn(skill, core, state, '{"confirm": "ACCEPTED"}')
        assert 'Form "Step 1"' in reply and "- confirm (required)" in reply  # the second consent is asked, not assumed
        reply = await turn(skill, core, state, '{"confirm": "ACCEPTED"}')
        assert reply.startswith("All pages of workflow")

    async def test_a_rejection_with_an_unparseable_correction_re_walks_instead_of_cancelling(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
        await open_form(
            skill,
            core,
            state,
            "create_demo_lightpath",
            '{"customer_name": "UT", "speed": "10000", "redundancy": "protected"}',
        )
        from orchestrator_agent.form_fill.hitl import PendingAsk, ToolApproval, ToolApprovalResponse, approval_as_text

        pending = PendingAsk(id="r", kind="approval")

        def rejection(reason):
            return ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=False, rejection_reason=reason)])

        reply = await skill.handle(approval_as_text(pending, rejection('{"speed": "1G"}')), state, core)
        assert not reply.text.startswith("Cancelled") and state.form_fill is not None  # a correction is walked with
        await skill.handle('{"speed": "10000"}', state, core)  # a good correction: the summary again
        assert state.form_fill.status == "confirming"
        reply = await skill.handle(approval_as_text(pending, rejection("not now")), state, core)
        assert reply.text.startswith("Cancelled") and state.form_fill is None

    def test_commands_are_exact_tokens(self):
        assert command(" cancel\n") is FormCommand.CANCEL and command("yes") is FormCommand.YES
        assert command("Yes.") is None and command("`no`") is None and command("go ahead") is None

    async def test_a_start_that_fails_without_an_answer_closes_the_form(self):
        core, state = FakeCore(), SearchState()
        skill = FormFillSkill()
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
        assert "whether a process was started is unknown" in reply and state.form_fill is None  # never retried blindly
