"""Tests for the deterministic form-fill skill: its stops are questions or an approval, its input the human's response."""

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
    summaries,
)
from orchestrator_agent.form_fill.skill import FormFillSkill, questions
from orchestrator_agent.state import AskField, Decision, FormFillSession, FormInput, FormReply, Reply, SearchState

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
# The last page of a form that ends in core's summary form (``base_summary``), as core 5.4 renders it: tables to
# show, nothing to fill.
SUMMARY_TABLE = {
    "columns": [["SURF", "10 Gbit/s", "Protected"]],
    "headers": [],
    "labels": ["customer_name", "speed", "redundancy"],
}
SUMMARY_PAGE = {
    "$defs": {"MigrationSummaryValue": {"properties": {}, "title": "MigrationSummaryValue", "type": "object"}},
    "additionalProperties": False,
    "properties": {
        "divider_1": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "format": "divider",
            "title": "Divider 1",
            "type": "string",
        },
        "product_summary": {
            "$ref": "#/$defs/MigrationSummaryValue",
            "default": None,
            "extraProperties": {"data": SUMMARY_TABLE},
            "format": "summary",
            "type": "string",
            "uniforms": {"data": SUMMARY_TABLE},
        },
    },
    "title": "Demo Lightpath Summary",
    "type": "object",
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

    def __init__(self, reject=None, accept_page=False, summary_page=False):
        self.reject = reject  # (page_index, message): raise when that page's inputs are submitted
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
        if self.summary_page:
            pages.append(SUMMARY_PAGE)
        if inputs:
            self.require(pages[len(inputs) - 1], inputs[-1])
        if len(inputs) >= len(pages):
            return {"page": len(inputs), "complete": True, "schema": None}
        return {"page": len(inputs), "complete": False, "schema": pages[len(inputs)]}


APPROVE, REJECT = Decision.START, Decision.CANCEL  # the human's decision on the start, as the transport hands it over
FULL = {"customer_name": "UT", "speed": "10000", "redundancy": "protected"}


class Stop:
    """A skill reply as the tests read it: the ``FormReply`` data, plus the stop it is (questions or an approval)."""

    def __init__(self, reply: Reply) -> None:
        self.reply, self.data = reply, FormReply.model_validate_json(reply.text)

    def __getattr__(self, name):
        return getattr(self.data, name)

    @property
    def asked(self) -> list[str]:
        """The fields asked, in order ([] when the stop is not questions)."""
        return [field.name for field in self.reply.ask or []]

    def question(self, name: str) -> AskField:
        return next(field for field in self.reply.ask or [] if field.name == name)

    @property
    def approval(self):
        return self.reply.approval


def make_skill() -> FormFillSkill:
    return FormFillSkill()


def data(reply: Reply | None) -> Stop | None:
    return None if reply is None else Stop(reply)


def rejected(reply: Stop) -> list[str]:
    """The fields core rejected, in core's order."""
    return [str(error["loc"][0]) for error in reply.rejected if error["loc"]]


async def turn(skill, core, state, given, *, hitl=True) -> Stop | None:
    """One turn of an open form.

    ``given`` is what the transport made of the message: the human's answers (a dict), their decision on the
    start, or None for a message that carries no response.
    """
    state.hitl = hitl
    if given is None:
        state.form_input = None
    else:
        state.form_input = FormInput(decision=given) if isinstance(given, Decision) else FormInput(values=given)
    return data(await skill.handle(state, core))


async def open_form(skill, core, state, key, values=None, subscription_id=None) -> Stop | None:
    """What follows the model's ``start_workflow_form(key, subscription_id)``; ``values`` are then answered."""
    opened = {"subscription_id": subscription_id} if subscription_id else {}
    state.hitl = True
    state.form_fill = FormFillSession(workflow_key=key, status="opening", values=opened)
    reply = data(await skill.open(state, core))
    return await turn(skill, core, state, values) if values and state.form_fill is not None else reply


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

    def test_a_summary_page_has_nothing_to_fill_but_tables_to_show(self):
        assert page_model(SUMMARY_PAGE).model_fields == {}
        assert summaries(SUMMARY_PAGE) == [SUMMARY_TABLE]
        assert summaries(LIGHTPATH_PAGE) == []  # labels and dividers are no summary

    def test_single_select_list_is_a_list_of_one(self):
        schema = {
            "$defs": {"N": {"enum": ["a", "b"], "type": "string"}},
            "properties": {"n": {"items": {"$ref": "#/$defs/N"}, "maxItems": 1, "type": "array"}},
        }
        (field,) = page_model(schema).model_fields.values()
        assert is_list(field) and item_bounds(field) == (None, 1) and choices(field) == ("a", "b")
        assert is_single_pick(field)

    def test_a_choice_core_has_no_option_for_is_still_a_page(self):
        # ``Literal`` cannot be empty; the field keeps its base type and the schema still says nothing is allowed.
        schema = {
            "$defs": {"Port": {"enum": [], "options": {}, "type": "string"}},
            "properties": {
                "port": {"$ref": "#/$defs/Port"},
                "ports": {"items": {"$ref": "#/$defs/Port"}, "type": "array"},
            },
            "required": ["port"],
        }
        model = page_model(schema)
        assert model.model_fields["port"].annotation is str and choices(model.model_fields["port"]) is None
        properties = model.model_json_schema()["properties"]
        assert properties["port"]["enum"] == [] and properties["ports"]["items"] == {"type": "string", "enum": []}

    def test_a_field_with_one_allowed_value_is_a_field_core_still_requires(self):
        schema = {
            "properties": {"kind": {"const": "port", "type": "string"}, "amount": {"const": 2, "type": "integer"}},
            "required": ["kind", "amount"],
        }
        fields = page_model(schema).model_fields
        assert choices(fields["kind"]) == ("port",) and choices(fields["amount"]) == (2,)  # as core gave them
        assert fields["kind"].is_required() and is_single_pick(fields["kind"])


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


class TestQuestions:
    """A stop is questions a person can answer: one per field, options as chips, core's message where it rejected."""

    def test_structured_fields_are_nested_models_and_asked_as_one_question_each(self):
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
        # The model's JSON schema is what the interpreter reads: nested shapes, counts, labels and formats as data.
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
        # One question per field: its title and whether it is required; a missing value needs no message.
        errors = form_errors(rejection({"service_ports": "Field required", "note": "Too long"}))
        errors[0]["type"] = "missing"
        assert [q.question for q in questions(model, {}, errors)] == [
            "service_ports (`service_ports`, required)",
            "note (`note`, optional — leave empty to keep the default) — Too long",
            "subscription_id (`subscription_id`, required)",
        ]

    def test_options_are_chips_shown_by_label_and_a_boolean_is_two_chips(self):
        speed, policer, name, _vlan, nodes, confirm = questions(page_model(FIELDS_PAGE), {}, [])
        assert speed.choices == ("1 Gbit/s", "10 Gbit/s", "100 Gbit/s") and speed.values == ("1000", "10000", "100000")
        assert policer.choices == ("True", "False") and policer.values == (True, False)  # the value is a boolean
        assert name.choices == () and not name.multiple  # free: a person types
        assert nodes.multiple and nodes.choices == ("Node A", "Node B") and nodes.values == ("a", "b")
        assert confirm.choices == ("ACCEPTED",) and confirm.values == ("ACCEPTED",)

    def test_only_what_is_rejected_or_unanswered_is_asked(self):
        model = page_model(LIGHTPATH_PAGE)
        errors = form_errors(rejection({"speed": "Input should be '1000', '10000' or '100000'"}))
        asked = questions(model, {"customer_name": "UT"}, errors)
        assert [q.name for q in asked] == ["speed", "speed_policer"]  # the name was accepted: not asked again
        assert asked[0].question.endswith("— Input should be '1000', '10000' or '100000'")

    def test_a_rejection_core_pins_on_no_field_asks_the_page_again_with_what_core_said(self):
        error = "HTTP error 400: Bad Request - {'validation_errors': [{'loc': (), 'msg': 'ports must differ', 'type': 'value_error'}]}"
        model = page_model(LIGHTPATH_PAGE)
        asked = questions(model, {"customer_name": "UT", "speed": "1000"}, form_errors(error))
        assert [q.name for q in asked] == ["customer_name", "speed", "speed_policer"]
        assert asked[0].question.endswith("— ports must differ") and "—" not in asked[1].question
        # A refusal that is no form verdict at all is asked the same way, with the refusal as it came.
        asked = questions(model, {"customer_name": "UT", "speed": "1000"}, [], "HTTP error 502: Bad Gateway")
        assert [q.name for q in asked] == ["customer_name", "speed", "speed_policer"]
        assert asked[0].question.endswith("— HTTP error 502: Bad Gateway")
        # On the whole form (core refused the start) there is no page to ask again: nothing is asked.
        assert questions(model, {}, form_errors(error), untouched=False) == []


# --- the skill --------------------------------------------------------------------------------------------


class TestSeePage:
    def test_a_page_takes_its_own_fields_exactly_as_given(self):
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
        page = FormFillSkill._see_page(session, page_model(FIELDS_PAGE), [])
        assert page == {
            "speed": "10 Gbit/s",
            "vlan": "",
        }  # as given, core judges it; names exact; display-only, absent: not
        assert session.values["ticket_id"] == "T-1"  # a value for a later page waits


class TestWalk:
    """The walk after a handoff: pages submitted to core as known, its verdict as questions; the approval; the start."""

    VALUES = {"speed": "10000", "speed_policer": True}

    async def test_full_flow_ask_answer_approve_start(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        # 1. product is the only option and taken; the lightpath page has nothing yet: its fields are the questions.
        reply = await open_form(skill, core, state, "create_demo_lightpath")
        assert reply.status == "gathering" and reply.page == 1 and reply.title == "Demo Lightpath"
        assert reply.asked == ["customer_name", "speed", "speed_policer"] and reply.approval is None
        assert rejected(reply) == ["customer_name", "speed"] and reply.rejected[0]["msg"] == "Field required"
        assert reply.values == {"product": PRODUCT} and reply.labels == {"product": "Demo Lightpath"}
        # 2. The human answers two of them; core still misses the name, and only that is asked again.
        reply = await turn(skill, core, state, self.VALUES)
        assert reply.asked == ["customer_name"] and reply.question("customer_name").choices == ()
        assert reply.question("customer_name").question == "Customer Name (`customer_name`, required) — Field required"
        assert reply.values == {"product": PRODUCT, "speed": "10000", "speed_policer": True}
        # 3. Page 2 (10 Gbit/s -> redundancy): a choice shown by label, and the optional ticket offered along.
        reply = await turn(skill, core, state, {"customer_name": "Universiteit Twente"})
        assert reply.page == 2 and rejected(reply) == ["redundancy"] and reply.asked == ["redundancy", "ticket_id"]
        redundancy = reply.question("redundancy")
        assert redundancy.choices == ("Protected", "Unprotected") and redundancy.values == ("protected", "unprotected")
        assert "optional" in reply.question("ticket_id").question
        # 4. The summary is the start to approve: the call as it will be made, and what its ids stand for.
        reply = await turn(skill, core, state, {"redundancy": "protected", "ticket_id": "JIRA-4821"})
        assert reply.status == "confirming" and reply.asked == [] and state.form_fill.status == "confirming"
        assert reply.approval.tool_name == "create_workflow"
        assert reply.approval.hint == "Start workflow `create_demo_lightpath` with these values?"
        assert reply.approval.args == {
            "workflow_key": "create_demo_lightpath",
            "json_data": state.form_fill.page_inputs,
            "labels": {"product": "Demo Lightpath", "speed": "10 Gbit/s", "redundancy": "Protected"},
        }
        # 5. The approval starts, with exactly the validated pages.
        reply = await turn(skill, core, state, APPROVE)
        assert reply.status == "started" and reply.process_id == FakeCore.PROCESS_ID and state.form_fill is None
        (created,) = core.created
        assert created == {
            "workflow_key": "create_demo_lightpath",
            "json_data": [
                {"product": PRODUCT},
                {"customer_name": "Universiteit Twente", "speed": "10000", "speed_policer": True},
                {"redundancy": "protected", "ticket_id": "JIRA-4821"},
            ],
        }

    async def test_the_summary_carries_the_values_and_the_defaults_that_apply(self):
        core, state = FakeCore(), SearchState()
        reply = await open_form(make_skill(), core, state, "create_demo_lightpath", FULL)
        assert reply.status == "confirming"
        assert reply.values == {"product": PRODUCT, "customer_name": "UT", "speed": "10000", "redundancy": "protected"}
        assert reply.defaults == {
            "speed_policer": False,
            "ticket_id": "",
        }  # optional, never set: the form default applies
        assert reply.summary == []  # this form has no summary page of its own

    async def test_the_workflows_own_summary_page_is_carried_to_the_approval(self):
        """A form that ends in core's summary form: its tables are what the workflow's author wants confirmed."""
        core, state = FakeCore(summary_page=True), SearchState()
        reply = await open_form(make_skill(), core, state, "create_demo_lightpath", FULL)
        # The summary page asks nothing, so it is no stop of its own: it is submitted as it is.
        assert reply.status == "confirming" and state.form_fill.page_inputs[-1] == {}
        assert reply.summary == [SUMMARY_TABLE]
        assert reply.values["customer_name"] == "UT"  # the values as they will be sent are still on the reply

    async def test_a_changed_answer_rewalks_and_regenerates_later_pages(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", {**FULL, "speed_policer": True})
        assert state.form_fill.status == "confirming"
        # 1 Gbit/s has no redundancy page: the earlier answer is simply no longer part of the form. The page
        # that replaces it has only an optional ticket nobody was offered yet, so that is asked — once.
        reply = await turn(skill, core, state, {"speed": "1000"})
        assert reply.status == "gathering" and reply.asked == ["ticket_id"] and reply.rejected == []
        reply = await turn(skill, core, state, {})
        assert reply.status == "confirming" and reply.values["speed"] == "1000" and "redundancy" not in reply.values
        assert state.form_fill.page_inputs == [
            {"product": PRODUCT},
            {"customer_name": "UT", "speed": "1000", "speed_policer": True},
            {},
        ]

    async def test_a_rejection_cancels_wherever_the_form_is(self):
        for answered in (self.VALUES, FULL):  # gathering, confirming
            core, state = FakeCore(), SearchState()
            skill = make_skill()
            await open_form(skill, core, state, "create_demo_lightpath", answered)
            reply = await turn(skill, core, state, REJECT)
            assert reply.status == "cancelled" and state.form_fill is None and not core.created

    async def test_an_optional_only_page_is_asked_once_and_an_empty_answer_moves_on_with_defaults(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        reply = await open_form(skill, core, state, "modify_demo_lightpath", subscription_id=FakeCore.SUB)
        assert reply.page == 1 and reply.asked == ["note"] and reply.rejected == []
        assert state.form_fill.page_inputs == [{"subscription_id": FakeCore.SUB}]
        reply = await turn(skill, core, state, {})  # every question left empty
        assert reply.status == "confirming" and reply.defaults == {"version": None, "note": None}

    async def test_read_only_single_option_field_is_never_submitted(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", {**self.VALUES, "customer_name": "UT"})
        assert "locked" not in state.form_fill.page_inputs[1] and "locked" not in state.form_fill.values

    async def test_a_value_for_another_field_does_not_touch_the_subscription(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "modify_demo_lightpath", subscription_id=FakeCore.SUB)
        other = "11111111-2222-4333-8444-555555555555"
        await turn(skill, core, state, {"customer_id": other})  # a uuid for another field
        assert state.form_fill.values["subscription_id"] == FakeCore.SUB  # not overwritten
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB}

    async def test_the_workflow_catalogue_is_fetched_once(self):
        core, calls = FakeCore(), []

        async def counting(name, args):
            calls.append(name)
            return await core(name, args)

        skill = make_skill()
        assert await skill.workflows(counting) == await skill.workflows(counting)
        assert calls.count("list_workflows") == 2  # user-facing workflows and tasks: one call each, once

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
        reply = await open_form(make_skill(), EndlessCore(), state, "create_demo_lightpath")
        assert reply.status == "failed" and "did not complete" in reply.reason and state.form_fill is None

    async def test_core_rejecting_a_page_is_asked_again_with_its_message_and_the_session_stays_open(self):
        message = "Input should be '1000', '10000' or '100000'"
        core, state = FakeCore(reject=(2, rejection({"speed": message}))), SearchState()
        skill = make_skill()
        reply = await open_form(skill, core, state, "create_demo_lightpath", {**self.VALUES, "customer_name": "UT"})
        assert reply.status == "gathering" and reply.title == "Demo Lightpath" and reply.page == 1
        assert reply.rejected == [{"loc": ("speed",), "msg": message, "type": "value_error"}] and reply.reason is None
        assert reply.asked == ["speed"] and reply.question("speed").question.endswith(f"— {message}")
        assert reply.values == {
            "product": PRODUCT,
            "customer_name": "UT",
            "speed_policer": True,
        }  # the rejected value is not
        assert state.form_fill.status == "gathering" and state.form_fill.page_inputs == [{"product": PRODUCT}]

    async def test_a_refusal_that_is_no_form_error_travels_as_the_reason_and_the_form_stays_open(self):
        # A passing failure in core must not cost the human the form: the page is asked again, with what core said.
        core, state = FakeCore(reject=(2, "Error calling tool: something else entirely")), SearchState()
        skill = make_skill()
        reply = await open_form(skill, core, state, "create_demo_lightpath", {**self.VALUES, "customer_name": "UT"})
        assert reply.status == "gathering" and reply.rejected == [] and reply.page == 1
        assert reply.reason == "Error calling tool: something else entirely"
        assert reply.asked == ["customer_name", "speed", "speed_policer"]
        assert reply.question("customer_name").question.endswith("— Error calling tool: something else entirely")
        assert state.form_fill.status == "gathering" and state.form_fill.page_inputs == [{"product": PRODUCT}]
        core.reject = None  # core is back: answering again goes on
        reply = await turn(skill, core, state, {"customer_name": "UT"})
        assert reply.page == 2 and reply.asked == ["redundancy", "ticket_id"]

    async def test_accept_field_is_asked_and_only_an_explicit_accept_fills_it(self):
        core, state = FakeCore(accept_page=True), SearchState()
        skill = make_skill()
        reply = await open_form(skill, core, state, "create_demo_lightpath", FULL)
        assert rejected(reply) == ["confirm"] and reply.asked == ["confirm"]
        assert reply.question("confirm").choices == ("ACCEPTED",)
        reply = await turn(skill, core, state, {"confirm": "ACCEPTED"})
        assert reply.status == "confirming" and state.form_fill.page_inputs[-1] == {"confirm": "ACCEPTED"}

    async def test_session_survives_a_json_round_trip(self):
        core, state = FakeCore(), SearchState()
        await open_form(make_skill(), core, state, "create_demo_lightpath", self.VALUES)
        state.form_fill.pending = {"id": "r", "kind": "ask", "questions": [{"field": "speed", "options": {"x": True}}]}
        restored = SearchState.model_validate(state.model_dump(mode="json"))
        assert restored.form_fill == state.form_fill


class TestOnlyAResponseContinuesAForm:
    """No chat text is read: a form goes on with the human's response to its stop, and with nothing else."""

    async def test_a_message_without_a_response_ends_the_form_and_is_the_models(self):
        for answered in (None, FULL):  # at a page's questions, at the approval
            core, state = FakeCore(), SearchState()
            skill = make_skill()
            await open_form(skill, core, state, "create_demo_lightpath", answered)
            assert state.form_fill is not None
            # Whatever the message says — "yes", "cancel", a question — it is not the form's.
            assert await turn(skill, core, state, None) is None
            assert state.form_fill is None and not core.created

    async def test_a_stop_not_shown_yet_is_shown_again_by_the_next_message(self):
        # A parent runtime pauses once per call: the stop that answered a response reaches the human only
        # when the parent calls again. The transport marks it; the skill then shows it instead of closing.
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        first = await open_form(skill, core, state, "create_demo_lightpath", FULL)
        state.form_fill.unseen = True
        again = await turn(skill, core, state, None)
        assert again.status == "confirming" and again.approval == first.approval and state.form_fill is not None
        state.form_fill.unseen = False  # shown now: the next message without a response ends it
        assert await turn(skill, core, state, None) is None and state.form_fill is None

    async def test_a_caller_that_cannot_pause_gets_no_form(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", FULL)
        state.form_fill.unseen = True
        assert await turn(skill, core, state, APPROVE, hitl=False) is None  # not even with a decision on the state
        assert state.form_fill is None and not core.created

    async def test_an_approval_is_only_a_start_at_the_summary(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", TestWalk.VALUES)
        reply = await turn(skill, core, state, APPROVE)  # a stale approval while a page is still open
        assert reply.status == "gathering" and reply.asked == ["customer_name"] and not core.created

    async def test_without_a_form_nothing_is_called(self):
        calls = []

        async def counting(name, args):
            calls.append(name)
            return await FakeCore()(name, args)

        assert await turn(make_skill(), counting, SearchState(), APPROVE) is None and calls == []

    async def test_a_handoff_never_walked_is_discarded_on_the_next_message(self):
        state = SearchState(form_fill=FormFillSession(workflow_key="create_demo_lightpath", status="opening"))
        assert await turn(make_skill(), FakeCore(), state, {"speed": "1000"}) is None
        assert state.form_fill is None


async def handoff(state, key, subscription_id=None, core=None, hitl=True):
    """The model's ``start_workflow_form`` call, through the checked tool (the catalogue comes from ``core``)."""
    from types import SimpleNamespace

    from orchestrator_agent.form_fill.handoff import build_handoff_toolset
    from orchestrator_agent.tool_names import START_WORKFLOW_FORM_TOOL

    state.hitl = hitl
    tool = build_handoff_toolset(make_skill(), core or FakeCore()).tools[START_WORKFLOW_FORM_TOOL].function
    return await tool(SimpleNamespace(deps=SimpleNamespace(state=state)), key, subscription_id=subscription_id)


class TestHandoff:
    """The model routes: ``start_workflow_form`` marks the session, ``open`` walks it in the same turn."""

    async def test_the_tool_marks_the_session(self):
        state = SearchState(user_input="create a lightpath for UT")
        out = await handoff(state, "create_demo_lightpath")
        assert state.form_fill.status == "opening" and state.form_fill.workflow_key == "create_demo_lightpath"
        assert state.form_fill.values == {} and "create_demo_lightpath" in out

    async def test_for_a_caller_that_cannot_pause_the_tool_opens_nothing_and_says_so(self):
        from orchestrator_agent.form_fill.handoff import NO_HITL

        calls = []

        async def counting(name, args):
            calls.append(name)
            return await FakeCore()(name, args)

        state = SearchState(user_input="create a lightpath for UT")
        assert await handoff(state, "create_demo_lightpath", core=counting, hitl=False) == NO_HITL
        assert state.form_fill is None and calls == []

    async def test_a_subscription_passed_by_the_model_fills_the_first_page_and_core_judges_it(self):
        core = FakeCore()
        state = SearchState(user_input="change its note")  # the id was said earlier
        await handoff(state, "modify_demo_lightpath", subscription_id=FakeCore.SUB, core=core)
        reply = data(await make_skill().open(state, core))
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB}
        assert reply.status == "gathering" and reply.rejected == [] and reply.asked == ["note"]
        # A subscription core will not run the workflow on: core's own subscription page says so, nothing here pre-checks.
        state = SearchState(user_input="change its note anyway")
        await handoff(state, "modify_demo_lightpath", subscription_id=FakeCore.OUT_OF_SYNC, core=core)
        reply = data(await make_skill().open(state, core))
        assert reply.status == "gathering" and reply.page == 0 and rejected(reply) == ["subscription_id"]
        assert reply.rejected[0]["msg"] == FakeCore.NOT_IN_SYNC and reply.values == {}
        assert reply.question("subscription_id").question.endswith(f"— {FakeCore.NOT_IN_SYNC}")
        assert state.form_fill.status == "gathering"  # the human may give another id, or reject

    async def test_a_create_workflow_ignores_a_subscription_passed_along(self):
        state = SearchState(user_input="create a new lightpath for ACE")
        await handoff(state, "create_demo_lightpath", subscription_id=FakeCore.SUB)
        reply = data(await make_skill().open(state, FakeCore()))
        assert state.form_fill.status == "gathering" and rejected(reply) == ["customer_name", "speed"]

    async def test_open_walks_the_first_pages_with_nothing_prefilled(self):
        state = SearchState(user_input='create a lightpath for UT {"customer_name": "UT"}')
        await handoff(state, "create_demo_lightpath")
        reply = data(await make_skill().open(state, FakeCore()))
        assert state.form_fill.status == "gathering"
        assert rejected(reply) == ["customer_name", "speed"]  # nothing is read from the request, data or words
        assert reply.values == {"product": PRODUCT}  # the single-option product page is still taken

    async def test_a_key_core_does_not_know_is_cores_refusal_and_the_form_closes(self):
        state = SearchState()  # the handoff tool checks the key; a stale session may still reach core with one
        reply = await open_form(make_skill(), FakeCore(), state, "create_unicorn")
        assert reply.status == "failed" and reply.reason == "Workflow 'create_unicorn' not found"
        assert state.form_fill is None

    async def test_the_tool_rejects_a_key_core_does_not_know_so_the_model_corrects_itself(self):
        state = SearchState(user_input="create a lightpath for UT")
        with pytest.raises(ModelRetry, match="Unknown workflow key 'create_demo_lightpth'"):
            await handoff(state, "create_demo_lightpth")
        assert state.form_fill is None
        await handoff(state, "create_demo_lightpath")
        assert state.form_fill.status == "opening" and state.form_fill.workflow_key == "create_demo_lightpath"

    async def test_a_task_is_a_workflow_the_model_may_hand_off(self):
        task = {**FakeCore.WORKFLOWS[0], "name": "task_validate_products", "target": "SYSTEM", "is_task": True}
        asked: list[dict] = []

        async def core(name, args):
            if name == "list_workflows":  # like core: a call without a filter is not taken
                asked.append(args)
                return [task] if args["is_task"] else FakeCore.WORKFLOWS
            return await FakeCore()(name, args)

        state = SearchState(user_input="validate the products")
        await handoff(state, "task_validate_products", core=core)
        assert state.form_fill.workflow_key == "task_validate_products"
        assert asked == [{"is_task": False}, {"is_task": True}]  # the whole catalogue, in the two halves core takes

    async def test_a_new_handoff_replaces_an_open_form(self):
        core, state = FakeCore(), SearchState()
        await open_form(make_skill(), core, state, "create_demo_lightpath", FULL)
        await handoff(state, "modify_demo_lightpath", subscription_id=FakeCore.SUB, core=core)
        assert state.form_fill.workflow_key == "modify_demo_lightpath" and state.form_fill.values == {
            "subscription_id": FakeCore.SUB
        }


class TestReviewRegressions:
    """Scenarios from the reviews; each one used to misbehave."""

    async def test_a_nested_object_is_that_fields_value_not_top_level_answers(self):
        other = "11111111-2222-4333-8444-555555555555"
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "modify_demo_lightpath", subscription_id=FakeCore.SUB)
        await turn(skill, core, state, {"note": {"subscription_id": other}})
        assert state.form_fill.page_inputs[0] == {"subscription_id": FakeCore.SUB}  # not clobbered

    async def test_a_failure_mid_walk_leaves_the_last_good_pages(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", FULL)
        assert state.form_fill.status == "confirming"
        pages_before = list(state.form_fill.page_inputs)

        class DownError(Exception):
            pass

        async def failing(name, args):
            if name == "get_workflow_form" and len(args["page_inputs"]) == 1:
                raise DownError("core unreachable")
            return await core(name, args)

        with pytest.raises(DownError):
            await turn(skill, failing, state, {"speed": "1000"})
        assert state.form_fill.page_inputs == pages_before and state.form_fill.status == "confirming"

    async def test_a_stale_choice_is_reported_by_core_when_the_page_regenerates(self):
        class DependentCore(FakeCore):
            async def __call__(self, name, args):
                if name == "get_workflow_form" and args["workflow_key"] == "create_demo_lightpath":
                    inputs = args["page_inputs"]
                    node = {"properties": {"node": {"enum": ["A", "B"], "type": "string"}}, "required": ["node"]}
                    if not inputs:
                        return {"page": 0, "complete": False, "schema": node}
                    self.require(node, inputs[0])
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
        await open_form(skill, core, state, "create_demo_lightpath", {"node": "A", "port": "A1"})
        assert state.form_fill.status == "confirming"
        reply = await turn(skill, core, state, {"node": "B"})
        # The stale port goes to core as it was; core's own message names the new options, and they are the chips.
        assert reply.rejected == [{"loc": ("port",), "msg": "Input should be 'B1' or 'B2'", "type": "literal_error"}]
        assert reply.asked == ["port"] and reply.question("port").choices == ("B1", "B2")
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
        reply = await open_form(skill, core, state, "create_demo_lightpath")
        assert reply.title == "Step 0" and rejected(reply) == ["confirm"]
        reply = await turn(skill, core, state, {"confirm": "ACCEPTED"})
        assert reply.title == "Step 1" and rejected(reply) == ["confirm"]  # the second consent is asked, not assumed
        reply = await turn(skill, core, state, {"confirm": "ACCEPTED"})
        assert reply.status == "confirming"
        # Both consents hold through a re-walk: one name on two pages is two consents, not one overwritten.
        reply = await turn(skill, core, state, {})
        assert reply.status == "confirming" and state.form_fill.page_inputs == [{"confirm": "ACCEPTED"}] * 2

    async def test_consent_is_void_once_what_it_was_given_for_changes(self):
        core, state = FakeCore(accept_page=True), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", FULL)
        reply = await turn(skill, core, state, {"confirm": "ACCEPTED"})
        assert reply.status == "confirming"
        reply = await turn(skill, core, state, {})  # nothing changed: the consent holds through the re-walk
        assert reply.status == "confirming"
        reply = await turn(skill, core, state, {"speed": "100000"})  # a value on a page before the consent changes
        assert reply.status == "gathering" and reply.asked == ["confirm"] and reply.values["speed"] == "100000"
        reply = await turn(skill, core, state, {"confirm": "ACCEPTED"})
        assert reply.status == "confirming" and not core.created

    async def test_a_start_that_fails_without_an_answer_closes_the_form(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", FULL)

        async def timing_out(name, args):
            if name == "create_workflow":
                raise TimeoutError("no answer")
            return await core(name, args)

        reply = await turn(skill, timing_out, state, APPROVE)
        assert reply.status == "failed" and reply.reason == "no answer" and state.form_fill is None  # never retried

    async def test_a_start_core_refuses_reopens_the_form_with_what_it_rejected(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", FULL)

        async def refusing(name, args):
            if name == "create_workflow":
                raise ModelRetry(rejection({"customer_name": "Customer no longer exists"}))
            return await core(name, args)

        reply = await turn(skill, refusing, state, APPROVE)
        assert reply.status == "gathering" and reply.page is None and rejected(reply) == ["customer_name"]
        assert reply.asked == ["customer_name"] and reply.values["speed"] == "10000"  # only what core rejected
        assert reply.question("customer_name").question.endswith("— Customer no longer exists")
        assert state.form_fill.status == "gathering"
        reply = await turn(skill, core, state, {"customer_name": "ACE"})  # answered: the start to approve again
        assert reply.status == "confirming" and reply.values["customer_name"] == "ACE"

    async def test_a_start_core_refuses_without_naming_a_field_closes_the_form(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", FULL)
        error = "HTTP error 400: Bad Request - {'validation_errors': [{'loc': (), 'msg': 'no capacity', 'type': 'value_error'}]}"

        async def refusing(name, args):
            if name == "create_workflow":
                raise ModelRetry(error)
            return await core(name, args)

        reply = await turn(skill, refusing, state, APPROVE)
        assert reply.status == "failed" and "no capacity" in reply.reason and state.form_fill is None

    async def test_a_start_refused_without_cores_verdict_on_the_pages_closes_the_form(self):
        # A tool error that is not core's validation of the pages says nothing about whether the process
        # started: reopening the form would let the next approval start it twice.
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", FULL)

        async def erroring(name, args):
            if name == "create_workflow":
                raise ModelRetry("Error calling tool 'create_workflow': HTTP error 500: Internal Server Error")
            return await core(name, args)

        reply = await turn(skill, erroring, state, APPROVE)
        assert reply.status == "failed" and "500" in reply.reason and state.form_fill is None

    async def test_a_start_core_answered_is_never_sent_again_whatever_the_answer_looks_like(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", FULL)

        async def odd(name, args):
            if name == "create_workflow":
                core.created.append(args)
                return {"unexpected": "shape"}
            return await core(name, args)

        reply = await turn(skill, odd, state, APPROVE)
        assert reply.status == "started" and reply.process_id is None and "unexpected" in reply.reason
        assert state.form_fill is None and len(core.created) == 1  # closed: a second approval has no form to start

    async def test_the_one_option_of_a_required_choice_is_taken_each_walk_and_never_kept(self):
        class RegeneratingCore(FakeCore):
            async def __call__(self, name, args):
                if name == "get_workflow_form" and args["workflow_key"] == "create_demo_lightpath":
                    inputs = args["page_inputs"]
                    node = {"properties": {"node": {"enum": ["A", "B"], "type": "string"}}, "required": ["node"]}
                    if not inputs:
                        return {"page": 0, "complete": False, "schema": node}
                    self.require(node, inputs[0])
                    port = {
                        "properties": {
                            "port": {"enum": [f"{inputs[0]['node']}1"], "type": "string"},
                            "spare": {"enum": ["S1"], "type": "string", "default": None},
                        },
                        "required": ["port"],
                    }
                    if len(inputs) == 1:
                        return {"page": 1, "complete": False, "schema": port}
                    if inputs[1].get("port") not in port["properties"]["port"]["enum"]:
                        raise ModelRetry(rejection({"port": "Input should be the one port of the node"}))
                    return {"page": 2, "complete": True, "schema": {}}
                return await super().__call__(name, args)

        core, state = RegeneratingCore(), SearchState()
        skill = make_skill()
        reply = await open_form(skill, core, state, "create_demo_lightpath", {"node": "A"})
        assert reply.status == "confirming" and reply.values == {"node": "A", "port": "A1"}  # optional: not picked
        assert "port" not in state.form_fill.values
        reply = await turn(skill, core, state, {"node": "B"})  # the page regenerates with another single option
        assert reply.status == "confirming" and reply.values == {"node": "B", "port": "B1"}
