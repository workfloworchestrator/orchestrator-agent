"""The skill walking a page with widget fields: options as chips, typed words resolved, words never sent to core."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic_ai import ModelRetry

from orchestrator_agent.adapters.a2a.kagent import ask_request
from orchestrator_agent.adapters.chat.librechat import ask_card
from orchestrator_agent.adapters.chat.librechat import pause as librechat_pause
from orchestrator_agent.form_fill import build_form_fill_skill
from orchestrator_agent.form_fill.interpret import ModelInterpreter
from orchestrator_agent.form_fill.pending import pending_of
from orchestrator_agent.form_fill.skill import NOTHING_FIT, SEVERAL_FIT, TOO_MANY_FIT, FormFillSkill
from orchestrator_agent.form_fill.widgets import CoreGraphQL, FieldWidget, Option, Widget, WidgetContext
from orchestrator_agent.state import AskField, FormFillSession, Reply, SearchState

from .test_form_fill import FakeCore, open_form, rejection, turn
from .test_form_widgets import FormatWidget

CUSTOMER: dict[str, Any] = {"format": "customerId", "title": "Customer", "type": "string"}
PAGE = {
    "properties": {
        "customer_id": CUSTOMER,
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "Note"},
    },
    "required": ["customer_id"],
    "title": "Customer",
    "type": "object",
}
UT = Option("c-ut", "Universiteit Twente (UT)", aliases=("Universiteit Twente", "UT"))
UU = Option("c-uu", "Universiteit Utrecht (UU)", aliases=("Universiteit Utrecht", "UU"))
TA = Option("c-ta", "Testaccount (TA)", aliases=("Testaccount", "TA"))
MANY = (TA, UT, UU, *(Option(f"c-{n}", f"Customer {n:02d}") for n in range(20)))
KEY = "widget_demo"


class WidgetCore:
    """Core's form tools for a task asking a customer (one page unless told otherwise); like core's ``CustomerId`` it takes any string."""

    def __init__(self, refuse: str | None = None, pages: Sequence[dict[str, Any]] = (PAGE,)) -> None:
        self.refuse = refuse  # a customer id core's validator rejects
        self.pages = pages
        self.submitted: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []

    async def __call__(self, name: str, args: dict[str, Any]) -> Any:
        if name == "list_workflows":
            return [{**FakeCore.WORKFLOWS[0], "name": KEY}]
        if name == "create_workflow":
            self.created.append(args)
            return {"id": FakeCore.PROCESS_ID}
        assert name == "get_workflow_form"
        inputs = args["page_inputs"]
        self.submitted.extend(inputs)
        if inputs:
            FakeCore.require(self.pages[len(inputs) - 1], inputs[-1])
        if self.refuse is not None and inputs and inputs[0].get("customer_id") == self.refuse:
            raise ModelRetry(rejection({"customer_id": "Customer not allowed"}))
        if len(inputs) >= len(self.pages):
            return {"page": len(inputs), "complete": True, "schema": None}
        return {"page": len(inputs), "complete": False, "schema": self.pages[len(inputs)]}


class Chooser:
    def __init__(self, picks: list[Any]) -> None:
        self.picks = picks
        self.calls: list[str] = []

    async def choose(self, title, options, words):
        self.calls.append(words)
        return self.picks


class Refuses:
    async def choose(self, title, options, words):
        raise AssertionError("no model call expected")

    async def answers(self, form, words):
        raise AssertionError("no interpretation expected")


def make(
    options: Sequence[Option] | None = MANY, choose=None, interpret=None, also: Sequence[FieldWidget] = ()
) -> FormFillSkill:
    return FormFillSkill(
        widgets=[FormatWidget("customerId", "customerId", options), *also], choose=choose, interpret=interpret
    )


async def test_a_long_list_is_asked_as_typed_text_with_its_size():
    reply = await open_form(make(), WidgetCore(), SearchState(), KEY)
    customer = reply.question("customer_id")
    assert customer.choices == () and customer.hint == "Type a name or part of it — 23 options."


async def test_an_exact_name_is_its_option_without_a_model():
    core, state = WidgetCore(), SearchState()
    reply = await open_form(make(choose=Refuses()), core, state, KEY, {"customer_id": "TESTACCOUNT"})
    assert reply.status == "confirming" and reply.values["customer_id"] == "c-ta"
    assert reply.labels == {"customer_id": "Testaccount (TA)"}
    assert reply.approval.args["labels"] == {"customer_id": "Testaccount (TA)"}
    assert "TESTACCOUNT" not in str(core.submitted)


async def test_words_that_fit_one_option_are_that_option():
    core, chooser = WidgetCore(), Chooser(["c-ut"])
    reply = await open_form(make(choose=chooser), core, SearchState(), KEY, {"customer_id": "twente"})
    assert reply.status == "confirming" and reply.values["customer_id"] == "c-ut" and chooser.calls == ["twente"]


@pytest.mark.parametrize(
    "picks,choices,hint",
    [
        pytest.param(["c-ut", "c-uu"], (UT.label, UU.label), SEVERAL_FIT, id="several-fit"),
        pytest.param([o.value for o in MANY[3:14]], (), TOO_MANY_FIT, id="too-many-fit"),
        pytest.param([], (), NOTHING_FIT.format(words="'uni'"), id="nothing-fits"),
    ],
)
async def test_unresolved_words_are_never_submitted(picks, choices, hint):
    core, state = WidgetCore(), SearchState()
    reply = await open_form(make(choose=Chooser(picks)), core, state, KEY, {"customer_id": "uni"})
    customer = reply.question("customer_id")
    assert reply.status == "gathering" and customer.choices == choices and customer.hint == hint
    assert "uni" not in str(core.submitted) and "customer_id" not in state.form_fill.values


async def test_a_picked_candidate_continues_and_is_not_read_again():
    core, state, chooser = WidgetCore(), SearchState(), Chooser(["c-ut", "c-uu"])
    skill = make(choose=chooser)
    await open_form(skill, core, state, KEY, {"customer_id": "uni"})
    reply = await turn(skill, core, state, {"customer_id": "c-uu"})  # the chip's value, as the transport maps it
    assert reply.status == "confirming" and reply.labels == {"customer_id": "Universiteit Utrecht (UU)"}
    await turn(skill, core, state, {"note": "hello"})  # another walk: the picked value stands
    assert chooser.calls == ["uni"]


async def test_a_short_list_is_chips_and_a_single_option_is_taken():
    reply = await open_form(make(options=(UT, UU)), WidgetCore(), SearchState(), KEY)
    customer = reply.question("customer_id")
    assert customer.choices == (UT.label, UU.label) and customer.values == ("c-ut", "c-uu") and customer.hint == ""
    only = await open_form(make(options=(TA,)), WidgetCore(), SearchState(), KEY)  # plain core: one default customer
    assert only.status == "confirming" and only.values["customer_id"] == "c-ta"


@pytest.mark.parametrize("picks", [pytest.param(None, id="no-model"), pytest.param([], id="the-model-finds-no-fit")])
async def test_typed_words_for_a_short_list_are_never_submitted(picks):
    core, state = WidgetCore(), SearchState()
    skill = make(options=(UT, UU), choose=None if picks is None else Chooser(picks))
    reply = await open_form(skill, core, state, KEY, {"customer_id": "nobody"})
    customer = reply.question("customer_id")
    assert reply.status == "gathering" and customer.hint == NOTHING_FIT.format(words="'nobody'")
    assert customer.choices == (UT.label, UU.label)  # the chips are still offered
    assert not any("customer_id" in page for page in core.submitted) and "customer_id" not in state.form_fill.values


@pytest.mark.parametrize(
    "answer,value,label",
    [
        pytest.param("UT", "c-ut", UT.label, id="a-typed-exact-name"),
        pytest.param("c-uu", "c-uu", UU.label, id="a-picked-chip"),
    ],
)
async def test_a_short_list_answer_is_one_of_its_options_without_a_model(answer, value, label):
    core = WidgetCore()
    reply = await open_form(make(options=(UT, UU), choose=Refuses()), core, SearchState(), KEY, {"customer_id": answer})
    assert (
        reply.status == "confirming" and reply.values["customer_id"] == value and reply.labels == {"customer_id": label}
    )
    assert [page["customer_id"] for page in core.submitted if "customer_id" in page] == [value]


async def test_a_field_waiting_for_another_value_is_not_asked():
    reply = await open_form(make(options=None), WidgetCore(), SearchState(), KEY)
    assert "customer_id" not in reply.asked


async def test_core_rejecting_a_picked_option_is_not_reinterpreted():
    skill = make(choose=Refuses(), interpret=Refuses())
    reply = await open_form(skill, WidgetCore(refuse="c-ut"), SearchState(), KEY, {"customer_id": "UT"})
    assert reply.question("customer_id").problem == "Customer not allowed"


def test_a_session_persisted_before_widgets_still_loads():
    question = {"name": "a", "question": "A?", "choices": [], "values": [], "multiple": False, "required": True}
    old = {"workflow_key": KEY, "status": "gathering", "pending": {"id": "p", "kind": "ask", "questions": [question]}}
    session = FormFillSession.model_validate(old)
    pending = pending_of(session)
    assert session.resolved == {} and pending is not None and pending.questions[0].hint == ""
    assert session.waiting == []


def test_the_waiting_fields_are_kept_with_the_session():
    session = FormFillSession(workflow_key=KEY, waiting=["contact"])
    assert FormFillSession.model_validate_json(session.model_dump_json()).waiting == ["contact"]


# --- a widget field that depends on another field (``WidgetContext.values``) -----------------------------

JAN, PIET = Option("p-jan", "Jan de Vries"), Option("p-piet", "Piet Jansen")
CONTACTS = {"c-ut": (JAN, PIET), "c-uu": (Option("p-anna", "Anna Smit"), Option("p-bob", "Bob Visser"))}
REQUIRED_CONTACT = {"format": "contactPerson", "title": "Contact", "type": "string"}
OPTIONAL_CONTACTS = {
    "default": [],
    "items": {"format": "contactPerson", "type": "string"},
    "title": "Contacts",
    "type": "array",
}


class Contacts(Widget):
    """The contact persons of the form's customer: none (the field waits) until that is one of the customers."""

    id = "contactPerson"

    def matches(self, field: Mapping[str, Any]) -> bool:
        return field.get("format") == "contactPerson"

    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None:
        return CONTACTS.get(ctx.values.get("customer_id", ""))


def page_of(fields: dict[str, Any], required: Sequence[str], title: str = "Customer") -> dict[str, Any]:
    return {"properties": fields, "required": list(required), "title": title, "type": "object"}


def with_contact(contact: dict[str, Any], *, required: bool) -> dict[str, Any]:
    """A page asking a customer and a contact person of that customer."""
    fields = {"customer_id": CUSTOMER, "contact": contact}
    return page_of(fields, ["customer_id", "contact"] if required else ["customer_id"])


@pytest.mark.parametrize("customer", [pytest.param("UT", id="typed-name"), pytest.param("c-ut", id="picked-value")])
@pytest.mark.parametrize(
    "contact,required,answer",
    [
        pytest.param(REQUIRED_CONTACT, True, "p-jan", id="required"),
        pytest.param(OPTIONAL_CONTACTS, False, ["p-jan"], id="optional"),
    ],
)
async def test_a_field_waiting_for_the_customer_is_asked_with_its_options_once_it_is_known(
    customer, contact, required, answer
):
    core, state = WidgetCore(pages=[with_contact(contact, required=required)]), SearchState()
    skill = make(also=[Contacts()])
    first = await open_form(skill, core, state, KEY)
    assert first.asked == ["customer_id"]  # the contact waits for its customer
    reply = await turn(skill, core, state, {"customer_id": customer})
    assert reply.status == "gathering" and reply.asked == ["contact"]
    asked = reply.question("contact")
    assert asked.values == ("p-jan", "p-piet") and asked.choices == (JAN.label, PIET.label)
    done = await turn(skill, core, state, {"contact": answer})
    assert done.status == "confirming" and done.values == {"customer_id": "c-ut", "contact": answer}
    assert all(page.get("contact") in (None, answer) for page in core.submitted)


async def test_a_stop_of_waiting_fields_only_asks_them_as_typed_text_and_never_submits_the_words():
    customer = {**CUSTOMER, "default": None}
    note = {"title": "Note", "type": "string"}
    first = page_of({"customer_id": customer, "note": note}, ["note"])
    core, state = WidgetCore(pages=[first, page_of({"contact": OPTIONAL_CONTACTS}, [], "Contacts")]), SearchState()
    skill = make(also=[Contacts()])
    reply = await open_form(skill, core, state, KEY, {"note": "hello"})  # no customer: the contacts wait for one
    contacts = reply.question("contact")
    assert reply.status == "gathering" and reply.asked == ["contact"]
    assert contacts.choices == () and contacts.hint == ""  # asked as typed text, as a field without its widget
    done = await turn(skill, core, state, {"contact": ["Jan"]})
    assert done.status == "confirming" and "contact" not in done.values  # still waiting: the words are kept back
    assert "Jan" not in str(core.submitted)


async def test_a_value_of_a_field_that_waits_again_is_never_submitted():
    core, state = WidgetCore(pages=[with_contact(REQUIRED_CONTACT, required=True)]), SearchState()
    skill = make(also=[Contacts()])
    await open_form(skill, core, state, KEY, {"customer_id": "c-ut"})
    assert (await turn(skill, core, state, {"contact": "p-jan"})).status == "confirming"
    again = await turn(skill, core, state, {"customer_id": "c-ta"})  # a customer without contact persons
    assert again.status == "gathering" and again.asked == ["contact"]
    assert not any(page.get("customer_id") == "c-ta" and "contact" in page for page in core.submitted)


async def test_words_for_a_one_option_field_core_rejected_are_asked_again_with_what_matched():
    only_customer = page_of({"customer_id": CUSTOMER}, ["customer_id"])
    core, state, skill = WidgetCore(refuse="c-ta", pages=[only_customer]), SearchState(), make(options=(TA,))
    refused = await open_form(skill, core, state, KEY)  # the one option is taken, and core rejects it
    assert refused.question("customer_id").problem == "Customer not allowed"
    reply = await turn(skill, core, state, {"customer_id": "someone else"})
    customer = reply.question("customer_id")
    assert reply.status == "gathering" and customer.hint == NOTHING_FIT.format(words="'someone else'")
    assert "someone else" not in str(core.submitted)


HINTED = AskField(
    name="customer_id",
    question="Customer (`customer_id`, required)",
    title="Customer",
    hint="Type a name — 23 options.",
)


def test_librechat_shows_the_hint_in_the_description():
    session = FormFillSession(workflow_key=KEY, pages=[{"title": "Customer"}])
    pending = librechat_pause(Reply("{}", ask=[HINTED]), session)
    (question,) = ask_card(pending, session, 0)["questions"]
    assert question["description"] == "Type a name — 23 options."


def test_kagent_shows_the_hint_after_the_question():
    payload, _ = ask_request("req-1", [HINTED])
    assert payload.questions[0].question == "Customer (`customer_id`, required) — Type a name — 23 options."


def test_the_skill_is_built_with_the_builtins_as_the_extender_arranges_them(monkeypatch):
    own = FormatWidget("surf-customer", "customerId")
    monkeypatch.setattr("orchestrator_agent.form_fill.agent_settings.FORM_WIDGET_EXTENDER", "fake_widgets:extend")
    monkeypatch.setattr("orchestrator_agent.form_fill.load_extender", lambda path: lambda widgets: [own, *widgets])
    skill = build_form_fill_skill("test")
    assert [widget.id for widget in skill.widgets] == ["surf-customer", "customerId", "productId"]
    assert isinstance(skill.interpret, ModelInterpreter) and skill.choose is skill.interpret
    assert isinstance(skill.graphql, CoreGraphQL) and skill.graphql.url.endswith("/api/graphql")


def test_without_a_model_nothing_reads_typed_words():
    skill = build_form_fill_skill()
    assert (
        skill.interpret is None
        and skill.choose is None
        and [w.id for w in skill.widgets] == ["customerId", "productId"]
    )
