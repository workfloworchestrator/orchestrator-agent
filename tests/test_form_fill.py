"""Tests for the deterministic form-fill skill: its stops are questions or an approval, its input the human's response."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from orchestrator.core.schemas.mcp_tools import FormField, FormFieldError, FormFieldOption, WorkflowFormPage
from pydantic_ai import ModelRetry

from orchestrator_agent.form_fill.model import askable, page_model
from orchestrator_agent.form_fill.skill import FormFillSkill, questions
from orchestrator_agent.state import AskField, Decision, FormFillSession, FormInput, FormReply, Reply, SearchState

PRODUCT = "a19904e6-fa9f-4dd3-90c1-08dbd6d0821e"


# --- pages as core's form tool describes them -------------------------------------------------------------


def field(name, kind="string", *, title=None, required=True, default=None, options=None, **extra) -> FormField:
    """A field of a page; ``options`` as ``{value: label}``, the rest as core's ``FormField`` has it."""
    listed = [FormFieldOption(value=v, label=label) for v, label in options.items()] if options is not None else None
    title = title or name.title().replace("_", " ")
    return FormField(name=name, title=title, kind=kind, required=required, default=default, options=listed, **extra)


def choice_list(name, options, **extra) -> FormField:
    """A multi-select of ``options`` (core puts the options on the list's item)."""
    return field(name, "list", item=field("", required=False, options=options), **extra)


def errors(problems: dict[str, str], type_="value_error") -> list[FormFieldError]:
    """Core's verdict on a page: one error per field, in core's words."""
    return [FormFieldError(loc=[name], msg=msg, type=type_) for name, msg in problems.items()]


def page(index: int, title, fields, verdict=()) -> dict:
    """A page as the tool returns it: the next one to fill, or with ``verdict`` the page core rejected."""
    status: Literal["next", "rejected"] = "rejected" if verdict else "next"
    result = WorkflowFormPage(
        page=index, complete=False, status=status, title=title, fields=fields, errors=list(verdict)
    )
    return result.model_dump(mode="json", by_alias=True)


def complete(index: int) -> dict:
    return WorkflowFormPage(page=index, complete=True, status="complete").model_dump(mode="json", by_alias=True)


def rejection(problems: dict[str, str]) -> str:
    """The tool error text fastmcp builds from core's 400 on ``create_workflow``."""
    listed = [{"loc": (name,), "msg": msg, "type": "value_error"} for name, msg in problems.items()]
    return f"HTTP error 400: Bad Request - {{'type': 'FormValidationError', 'validation_errors': {listed!r}, 'status': 400}}"


SPEED = {"1000": "1 Gbit/s", "10000": "10 Gbit/s", "100000": "100 Gbit/s"}
REDUNDANCY = {"protected": "Protected", "unprotected": "Unprotected"}
PRODUCT_PAGE = (None, [field("product", options={PRODUCT: "Demo Lightpath"}, format="productId")])
LIGHTPATH_FIELDS = [
    field("header", required=False, default="Details", format="label", display_only=True),
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
ACCEPT = field("confirm", format="accept", options={"ACCEPTED": "ACCEPTED"})
ACCEPT_PAGE = ("Dry run", [ACCEPT])
# The last page of a form that ends in core's summary form (``base_summary``): tables to show, nothing to fill.
SUMMARY_TABLE = {
    "columns": [["SURF", "10 Gbit/s", "Protected"]],
    "headers": [],
    "labels": ["customer_name", "speed", "redundancy"],
}
SUMMARY_PAGE = (
    "Demo Lightpath Summary",
    [
        field("divider_1", required=False, nullable=True, format="divider", display_only=True),
        field("product_summary", "any", required=False, format="summary", display_only=True, data=SUMMARY_TABLE),
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
        lacking = {f.name: "Field required" for f in fields if askable(f) and f.required and f.name not in submitted}
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
    return [str(error.loc[0]) for error in reply.rejected if error.loc]


def verdict(reply: Stop) -> list[tuple]:
    """Core's errors on the reply as ``(field, message, type)``."""
    return [(str(error.loc[0]) if error.loc else "", error.msg, error.type) for error in reply.rejected]


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


FIELDS = [
    field("speed", options=SPEED),
    field("speed_policer", "boolean"),
    field("customer_name"),
    field("vlan", "integer", required=False, nullable=True),
    choice_list("nodes", {"a": "Node A", "b": "Node B"}, required=False),
    ACCEPT,
    field("header", required=False, format="label", display_only=True),
]


class TestQuestions:
    """A stop is questions a person can answer: one per field, options as chips, core's message where it rejected."""

    def test_structured_fields_are_asked_as_one_question_each(self):
        port = field(
            "",
            "object",
            required=False,
            fields=[
                field("subscription_id", options={"p-1": "ACE SP DT010A", "p-2": "ACE SP ASD002A"}),
                field("vlan", required=False, default="0"),
            ],
        )
        fields = [
            field("service_ports", "list", item=port, constraints={"min_length": 2, "max_length": 2}),
            field("subscription_id", format="uuid"),
            field("note", required=False, nullable=True, format="long"),
        ]
        # One question per field: its title and whether it is required; a missing value needs no message.
        verdicts = errors({"service_ports": "Field required"}, "missing") + errors({"note": "Too long"})
        assert [q.question for q in questions(fields, {}, verdicts)] == [
            "Service Ports (`service_ports`, required)",
            "Note (`note`, optional — leave empty to keep the default) — Too long",
            "Subscription Id (`subscription_id`, required)",
        ]
        ports = questions(fields, {}, [])[0]
        assert ports.multiple and ports.choices == ()  # a list of objects is typed, and interpreted

    def test_options_are_chips_shown_by_label_and_a_boolean_is_two_chips(self):
        speed, policer, name, _vlan, nodes, confirm = questions(FIELDS, {}, [])  # the label is no question
        assert speed.choices == ("1 Gbit/s", "10 Gbit/s", "100 Gbit/s") and speed.values == ("1000", "10000", "100000")
        assert policer.choices == ("True", "False") and policer.values == (True, False)  # the value is a boolean
        assert name.choices == () and not name.multiple  # free: a person types
        assert nodes.multiple and nodes.choices == ("Node A", "Node B") and nodes.values == ("a", "b")
        assert confirm.choices == ("ACCEPTED",) and confirm.values == ("ACCEPTED",)

    def test_options_that_share_a_label_are_told_apart_by_their_value(self):
        # Labels are descriptions and descriptions coincide. A chip's text is all a transport gets back from
        # a pick, so each option must read differently to map back to its id: core takes ids, never labels.
        (nodes,) = questions([choice_list("nodes", {"id-1": "Node X", "id-2": "Node X", "id-3": "Node Y"})], {}, [])
        assert nodes.choices == ("Node X (id-1)", "Node X (id-2)", "Node Y")
        assert nodes.values == ("id-1", "id-2", "id-3")

    def test_only_what_is_rejected_or_unanswered_is_asked(self):
        verdicts = errors({"speed": "Input should be '1000', '10000' or '100000'"})
        asked = questions(LIGHTPATH_FIELDS, {"customer_name": "UT"}, verdicts)
        assert [q.name for q in asked] == ["speed", "speed_policer"]  # the name was accepted: not asked again
        assert asked[0].question.endswith("— Input should be '1000', '10000' or '100000'")

    def test_a_rejection_core_pins_on_no_field_asks_the_page_again_with_what_core_said(self):
        verdicts = [FormFieldError(loc=["__root__"], msg="ports must differ", type="value_error")]
        asked = questions(LIGHTPATH_FIELDS, {"customer_name": "UT", "speed": "1000"}, verdicts)
        assert [q.name for q in asked] == ["customer_name", "speed", "speed_policer"]
        assert asked[0].question.endswith("— ports must differ") and "—" not in asked[1].question
        # A refusal that is no form verdict at all is asked the same way, with the refusal as it came.
        asked = questions(LIGHTPATH_FIELDS, {"customer_name": "UT", "speed": "1000"}, [], "HTTP error 502: Bad Gateway")
        assert [q.name for q in asked] == ["customer_name", "speed", "speed_policer"]
        assert asked[0].question.endswith("— HTTP error 502: Bad Gateway")


class TestRecordedCorePage:
    """Pages recorded from core's form tool (``create_sn8_service_port`` on a production clone): the contract, live."""

    PAGES = json.loads((Path(__file__).parent / "core_pages.json").read_text())

    def test_a_recorded_page_is_asked_as_core_describes_it(self):
        page = WorkflowFormPage.model_validate(self.PAGES["page_2"])
        asked = questions(page.fields, {}, [])
        assert [q.name for q in asked] == [
            "port",
            "port_mode",
            "admin_state",
            "lldp",
            "ignore_l3_incompletes",
            "native_vlan",
            "contact_persons",
            "ticket_id",
        ]
        port, mode, state, lldp = asked[:4]
        assert port.values[0] == "AH001A" and port.choices[0] == "AH001A (400 Gbit/s)"  # ids behind labels
        assert mode.choices == ("Tagged", "Untagged", "Link member") and mode.values == (
            "tagged",
            "untagged",
            "link_member",
        )
        assert not state.required and lldp.values == (True, False)
        contacts = asked[6]
        assert contacts.multiple and contacts.choices == ()  # a list of objects: typed, then interpreted
        assert list(page_model(page.fields).model_fields) == [q.name for q in asked]

    def test_cores_verdict_on_a_recorded_page_is_asked_in_its_words(self):
        page = WorkflowFormPage.model_validate(self.PAGES["page_2_rejected"])
        assert page.status == "rejected" and page.page == 2
        asked = questions(page.fields, {"port_mode": "sideways"}, page.errors)
        assert [q.name for q in asked][:2] == ["port", "port_mode"]
        assert "—" not in asked[0].question  # missing: the question already says it is required
        assert asked[1].question.endswith("— Input should be 'tagged', 'untagged' or 'link_member'")


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
        page = FormFillSkill._see_page(session, [f for f in FIELDS if askable(f)], [])
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
        assert rejected(reply) == ["customer_name", "speed"] and reply.rejected[0].msg == "Field required"
        assert reply.values == {"product": PRODUCT} and reply.labels == {"product": "Demo Lightpath"}
        # 2. The human answers two of them; core still misses the name, and only that is asked again.
        reply = await turn(skill, core, state, self.VALUES)
        assert reply.asked == ["customer_name"] and reply.question("customer_name").choices == ()
        assert reply.question("customer_name").question == "Customer Name (`customer_name`, required)"
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
            calls.append((name, args))
            return await core(name, args)

        skill = make_skill()
        assert await skill.workflows(counting) == await skill.workflows(counting)
        assert calls == [("list_workflows", {})]  # the whole catalogue, tasks included, in one call, once

    async def test_a_form_that_never_completes_fails_and_closes_instead_of_looping(self):
        class EndlessCore(FakeCore):
            async def __call__(self, name, args):
                if name == "get_workflow_form":
                    return page(len(args["page_inputs"]), None, [])
                return await super().__call__(name, args)

        state = SearchState()
        reply = await open_form(make_skill(), EndlessCore(), state, "create_demo_lightpath")
        assert reply.status == "failed" and "did not complete" in reply.reason and state.form_fill is None

    async def test_core_rejecting_a_page_is_asked_again_with_its_message_and_the_session_stays_open(self):
        message = "Input should be '1000', '10000' or '100000'"
        core, state = FakeCore(reject=(2, {"speed": message})), SearchState()
        skill = make_skill()
        reply = await open_form(skill, core, state, "create_demo_lightpath", {**self.VALUES, "customer_name": "UT"})
        assert reply.status == "gathering" and reply.title == "Demo Lightpath" and reply.page == 1
        assert verdict(reply) == [("speed", message, "value_error")] and reply.reason is None
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
        assert reply.rejected[0].msg == FakeCore.NOT_IN_SYNC and reply.values == {}
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
            if name == "list_workflows":  # core lists everything, tasks included, when nothing narrows it
                asked.append(args)
                return [*FakeCore.WORKFLOWS, task]
            return await FakeCore()(name, args)

        state = SearchState(user_input="validate the products")
        await handoff(state, "task_validate_products", core=core)
        assert state.form_fill.workflow_key == "task_validate_products"
        assert asked == [{}]

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
                    node = [field("node", options={"A": "A", "B": "B"})]
                    if not inputs:
                        return page(0, None, node)
                    if lacking := self.missing(node, inputs[0]):
                        return page(0, None, node, lacking)
                    ports = ["A1", "A2"] if inputs[0]["node"] == "A" else ["B1", "B2"]
                    port = [field("port", options={p: p for p in ports})]
                    if len(inputs) >= 2 and inputs[1].get("port") not in ports:  # core validates the value
                        said = f"Input should be {' or '.join(repr(p) for p in ports)}"
                        return page(1, None, port, errors({"port": said}, "literal_error"))
                    return page(1, None, port) if len(inputs) == 1 else complete(2)
                return await super().__call__(name, args)

        core, state = DependentCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", {"node": "A", "port": "A1"})
        assert state.form_fill.status == "confirming"
        reply = await turn(skill, core, state, {"node": "B"})
        # The stale port goes to core as it was; core's own message names the new options, and they are the chips.
        assert verdict(reply) == [("port", "Input should be 'B1' or 'B2'", "literal_error")]
        assert reply.asked == ["port"] and reply.question("port").choices == ("B1", "B2")
        assert state.form_fill is not None and state.form_fill.status == "gathering"

    async def test_consent_is_given_per_page(self):
        class TwoConsents(FakeCore):
            async def __call__(self, name, args):
                if name == "get_workflow_form" and args["workflow_key"] == "create_demo_lightpath":
                    n = len(args["page_inputs"])
                    if n and (lacking := self.missing([ACCEPT], args["page_inputs"][-1])):
                        return page(n - 1, f"Step {n - 1}", [ACCEPT], lacking)
                    return complete(2) if n == 2 else page(n, f"Step {n}", [ACCEPT])
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

    async def test_a_start_core_refuses_reopens_the_form_at_the_page_it_rejects(self):
        # The customer went away between the walk and the approval: the start is refused, and the form tool,
        # asked again about the same pages, rejects the page that names the customer.
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", FULL)

        async def refusing(name, args):
            if name == "create_workflow":
                raise ModelRetry(rejection({"customer_name": "Customer no longer exists"}))
            if name == "get_workflow_form" and len(args["page_inputs"]) == 3:
                return page(
                    1, "Demo Lightpath", LIGHTPATH_FIELDS, errors({"customer_name": "Customer no longer exists"})
                )
            return await core(name, args)

        reply = await turn(skill, refusing, state, APPROVE)
        assert reply.status == "gathering" and reply.page == 1 and rejected(reply) == ["customer_name"]
        # The page is asked again: what core rejected, and what was left to its default along with it.
        assert reply.asked == ["customer_name", "speed_policer"] and reply.values["speed"] == "10000"
        assert reply.question("customer_name").question.endswith("— Customer no longer exists")
        assert state.form_fill.status == "gathering" and not core.created
        reply = await turn(skill, core, state, {"customer_name": "ACE"})  # answered: the start to approve again
        assert reply.status == "confirming" and reply.values["customer_name"] == "ACE"

    async def test_a_start_core_refuses_without_naming_a_field_asks_that_page_again_with_its_words(self):
        core, state = FakeCore(), SearchState()
        skill = make_skill()
        await open_form(skill, core, state, "create_demo_lightpath", FULL)
        no_capacity = FormFieldError(loc=["__root__"], msg="no capacity", type="value_error")

        async def refusing(name, args):
            if name == "create_workflow":
                raise ModelRetry(
                    "HTTP error 400: Bad Request - {'validation_errors': [{'loc': (), 'msg': 'no capacity'}]}"
                )
            if name == "get_workflow_form" and len(args["page_inputs"]) == 3:
                return page(2, "Redundancy and ticket", REDUNDANCY_PAGE[1], [no_capacity])
            return await core(name, args)

        reply = await turn(skill, refusing, state, APPROVE)
        assert (
            reply.status == "gathering"
            and reply.page == 2
            and verdict(reply) == [("__root__", "no capacity", "value_error")]
        )
        assert reply.asked == ["redundancy", "ticket_id"] and reply.question("redundancy").question.endswith(
            "— no capacity"
        )
        assert state.form_fill.status == "gathering" and not core.created  # the person corrects, or rejects

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
                    node = [field("node", options={"A": "A", "B": "B"})]
                    if not inputs:
                        return page(0, None, node)
                    if lacking := self.missing(node, inputs[0]):
                        return page(0, None, node, lacking)
                    only = f"{inputs[0]['node']}1"
                    port = [
                        field("port", options={only: only}),
                        field("spare", required=False, nullable=True, options={"S1": "S1"}),
                    ]
                    if len(inputs) == 1:
                        return page(1, None, port)
                    if inputs[1].get("port") != only:
                        return page(1, None, port, errors({"port": "Input should be the one port of the node"}))
                    return complete(2)
                return await super().__call__(name, args)

        core, state = RegeneratingCore(), SearchState()
        skill = make_skill()
        reply = await open_form(skill, core, state, "create_demo_lightpath", {"node": "A"})
        assert reply.status == "confirming" and reply.values == {"node": "A", "port": "A1"}  # optional: not picked
        assert "port" not in state.form_fill.values
        reply = await turn(skill, core, state, {"node": "B"})  # the page regenerates with another single option
        assert reply.status == "confirming" and reply.values == {"node": "B", "port": "B1"}
