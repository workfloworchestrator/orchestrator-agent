"""The skill walking a page with widget fields: options as chips, typed words resolved, words never sent to core."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai import ModelRetry

from orchestrator_agent.form_fill.pending import pending_of
from orchestrator_agent.form_fill.skill import NOTHING_FIT, SEVERAL_FIT, TOO_MANY_FIT, FormFillSkill
from orchestrator_agent.form_fill.widgets import Option
from orchestrator_agent.state import FormFillSession, SearchState

from .test_form_fill import FakeCore, open_form, rejection, turn
from .test_form_widgets import FormatWidget

PAGE = {
    "properties": {
        "customer_id": {"format": "customerId", "title": "Customer", "type": "string"},
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
    """Core's form tools for a one-page task asking a customer; like core's ``CustomerId`` it takes any string."""

    def __init__(self, refuse: str | None = None) -> None:
        self.refuse = refuse  # a customer id core's validator rejects
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
        if not inputs:
            return {"page": 0, "complete": False, "schema": PAGE}
        FakeCore.require(PAGE, inputs[0])
        if inputs[0]["customer_id"] == self.refuse:
            raise ModelRetry(rejection({"customer_id": "Customer not allowed"}))
        return {"page": 1, "complete": True, "schema": None}


class Chooser:
    def __init__(self, picks: list[Any]) -> None:
        self.picks, self.calls = picks, []

    async def choose(self, title, options, words):
        self.calls.append(words)
        return self.picks


class Refuses:
    async def choose(self, title, options, words):
        raise AssertionError("no model call expected")

    async def answers(self, form, words):
        raise AssertionError("no interpretation expected")


def make(options: Sequence[Option] | None = MANY, choose=None, interpret=None) -> FormFillSkill:
    return FormFillSkill(
        widgets=[FormatWidget("customerId", "customerId", options)], choose=choose, interpret=interpret
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
