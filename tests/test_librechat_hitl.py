"""LibreChat's ask-user tool as a human-in-the-loop transport: a stop as cards, tool messages as the response."""

from __future__ import annotations

import json
from typing import Any

from openai.types.chat import ChatCompletionMessage

from orchestrator_agent.adapters.chat.librechat import (
    APPROVE_LABEL,
    KEEP_LABEL,
    MAX_LABEL,
    MAX_QUESTION,
    NOT_SHOWN,
    REJECT_LABEL,
    SKIPPED,
    LibreChatHitl,
    approval_card,
    ask_card,
    call_id,
    pause,
    read,
    shown,
)
from orchestrator_agent.adapters.chat.request import ChatRequest
from orchestrator_agent.form_fill.pending import PendingAsk, pending_of
from orchestrator_agent.state import Approval, AskField, Decision, FormFillSession, FormInput, Reply

GATHERING = '{"workflow_key":"create_port","status":"gathering","page":1,"title":"Port details"}'

# A page of five fields: more than one call takes, with every kind of answer.
PAGE = [
    AskField(
        name="product",
        question="Product (`product`, required)",
        choices=["SN8 10G", "SN8 1G"],
        values=["uuid-10g", "uuid-1g"],
        title="Product",
    ),
    AskField(name="ticket_id", question="Ticket (`ticket_id`, optional)", required=False, title="Ticket"),
    AskField(name="policer", question="Policer (`policer`, required)", choices=["True", "False"], values=[True, False]),
    AskField(
        name="ports",
        question="Ports (`ports`, required)",
        choices=["Port A", "Port B", "Port C"],
        values=["a", "b", "c"],
        multiple=True,
        title="Ports",
        problem="List should have at least 2 items",
    ),
    AskField(name="vlan", question="Vlan (`vlan`, required)", title="Vlan"),
]

APPROVAL = Reply(
    '{"workflow_key":"create_port","status":"confirming","values":{"product":"uuid-10g","vlan":20},'
    '"labels":{"product":"SN8 10G"},"defaults":{"policer":true}}',
    approval=Approval(hint="Start workflow `create_port` with these values?", tool_name="create_workflow", args={}),
)


def _paused(
    ask: list[AskField] | None = None, reply: Reply | None = None, title: str = "Port details"
) -> tuple[FormFillSession, PendingAsk]:
    """A session paused at a stop, as the skill leaves it: the page being asked is the last one walked."""
    session = FormFillSession(workflow_key="create_port", status="gathering", unseen=True, pages=[{"title": title}])
    pending = pause(reply or Reply(GATHERING, ask=PAGE if ask is None else ask), session)
    assert pending is not None
    return session, pending


def _question(session: FormFillSession, index: int) -> dict[str, Any]:
    """One question of the page as LibreChat is shown it, from the card it is on."""
    pending = pending_of(session)
    assert pending is not None
    return next(q for q in ask_card(pending, session, index // 4)["questions"] if q["id"] == f"q{index}")


def _value(session: FormFillSession, index: int, label: str) -> str:
    return next(option["value"] for option in _question(session, index)["options"] if option["label"] == label)


def _decision_value(pending: PendingAsk, label: str) -> str:
    assert APPROVAL.approval is not None
    (question,) = approval_card(pending, APPROVAL.approval)["questions"]
    return next(option["value"] for option in question["options"] if option["label"] == label)


def _answer(pending: PendingAsk, card: int, answers: dict[str, str]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id(pending, card), "content": json.dumps({"answers": answers})}


def _read(session: FormFillSession, messages: list[dict[str, Any]]) -> FormInput | ChatCompletionMessage | None:
    """``read`` over the messages of a request, as the endpoint hands them on: the tool results by call id."""
    return read(session, {m["tool_call_id"]: m["content"] for m in messages if m.get("role") == "tool"})


class TestCards:
    def test_a_chat_that_offers_a_tool_can_show_a_card(self):
        offered = ChatRequest(tools=[{"type": "function", "function": {"name": "ask_the_user"}}])
        assert LibreChatHitl().shows_stops(offered)  # whatever the tool is called
        # The endpoint used without the model spec: LibreChat offers no tool, and cannot show a card.
        assert not LibreChatHitl().shows_stops(ChatRequest())

    def test_a_page_becomes_cards_of_at_most_four_questions(self):
        session, pending = _paused()
        cards = [ask_card(pending, session, card) for card in range(2)]
        assert [[q["id"] for q in card["questions"]] for card in cards] == [["q0", "q1", "q2", "q3"], ["q4"]]
        assert session.unseen is False  # a card is shown as soon as it is sent
        # What is remembered is what the skill asked, nothing of LibreChat's: it survives the session's JSON
        # round trip, and the same cards come out of it again.
        assert session.pending == pending.model_dump(mode="json")
        restored = pending_of(FormFillSession.model_validate(session.model_dump(mode="json")))
        assert restored is not None and [q.name for q in restored.questions] == [f.name for f in PAGE]
        assert [ask_card(restored, session, card) for card in range(2)] == cards

    def test_a_question_is_plain_text_with_the_page_as_its_header(self):
        session, _ = _paused()
        product = _question(session, 0)
        assert product["question"] == "Product (product, required)"
        assert product["header"] == "Port details"
        assert "description" not in product
        # Without a title the skill's own wording is used, less its markdown.
        assert _question(session, 2)["question"] == "Policer (policer, required)"
        # Core's message on a rejected field is the description.
        assert _question(session, 3)["description"] == "List should have at least 2 items"
        # A page the form gave no title is headed by its workflow.
        untitled, _ = _paused(title="unknown")
        assert _question(untitled, 0)["header"] == "create_port"

    def test_options_carry_values_only_this_stop_knows(self):
        session, pending = _paused()
        options = _question(session, 0)["options"]
        assert [option["label"] for option in options] == ["SN8 10G", "SN8 1G"]
        assert all(option["value"].startswith(f"{pending.id}:") for option in options)
        other, _ = _paused()  # the same page asked again is another stop, with other values
        assert _value(other, 0, "SN8 10G") != _value(session, 0, "SN8 10G")
        assert _question(session, 3)["multiSelect"] is True and "multiSelect" not in _question(session, 0)

    def test_an_optional_field_can_be_left_as_it_is(self):
        session, _ = _paused()
        ticket = _question(session, 1)
        assert ticket["question"] == "Ticket (ticket_id, optional)"
        assert [option["label"] for option in ticket["options"]] == [KEEP_LABEL]
        assert KEEP_LABEL in ticket["description"]
        assert "options" not in _question(session, 4)  # a required free field is just typed

    def test_more_options_than_a_question_takes_are_typed_instead(self):
        labels = [f"Customer {n}" for n in range(13)]
        session, _ = _paused([AskField(name="customer", question="Customer", choices=labels, values=list(range(13)))])
        question = _question(session, 0)
        assert "options" not in question
        assert "Customer 0; Customer 1" in question["description"] and "13 options" in question["description"]
        # Twelve fit; twelve and "Keep default" do not.
        twelve, _ = _paused([AskField(name="c", question="C", choices=labels[:12], values=list(range(12)))])
        assert len(_question(twelve, 0)["options"]) == 12
        optional, _ = _paused(
            [AskField(name="c", question="C", choices=labels[:12], values=list(range(12)), required=False)]
        )
        assert [option["label"] for option in _question(optional, 0)["options"]] == [KEEP_LABEL]

    def test_texts_stay_within_the_tools_limits(self):
        long = AskField(name="f", question="Q" * 3000, choices=["L" * 400, "short"], values=[1, 2], title="T" * 3000)
        session, pending = _paused([long])
        question = _question(session, 0)
        assert len(question["question"]) == MAX_QUESTION
        assert len(question["options"][0]["label"]) == MAX_LABEL
        # The choice itself is not cut: picking the shortened option is the whole choice's value.
        picked = _answer(pending, 0, {"q0": question["options"][0]["value"]})
        assert _read(session, [picked]) == FormInput(values={"f": 1})

    def test_the_approval_is_one_card_with_two_options(self):
        session, pending = _paused(reply=APPROVAL)
        (question,) = approval_card(pending, APPROVAL.approval)["questions"]
        assert pending.kind == "approval" and session.pending["kind"] == "approval"
        assert question["question"] == "Start workflow create_port with these values?"
        assert [option["label"] for option in question["options"]] == [APPROVE_LABEL, REJECT_LABEL]

    def test_a_reply_that_ends_the_form_is_no_stop(self):
        session = FormFillSession(workflow_key="w", status="done")
        assert pause(Reply('{"workflow_key":"w","status":"started"}'), session) is None
        assert pause(Reply(GATHERING, ask=PAGE), None) is None
        assert session.pending is None


class TestRead:
    def _first(self, session: FormFillSession, pending: PendingAsk, **overrides: str) -> dict[str, Any]:
        answers = {
            "q0": _value(session, 0, "SN8 10G"),
            "q1": _value(session, 1, KEEP_LABEL),
            "q2": _value(session, 2, "False"),
            "q3": f"{_value(session, 3, 'Port C')}, {_value(session, 3, 'Port A')}",
        }
        return _answer(pending, 0, {**answers, **overrides})

    def test_the_next_card_is_shown_until_all_are_answered(self):
        session, pending = _paused()
        following = _read(session, [self._first(session, pending)])
        # The request is answered with the second card; the skill has not seen an answer yet.
        assert isinstance(following, ChatCompletionMessage) and following.content is None
        (call,) = following.tool_calls or []
        assert call.id == call_id(pending, 1)
        assert json.loads(call.function.arguments) == ask_card(pending, session, 1)

    def test_the_answers_are_the_fields_values(self):
        session, pending = _paused()
        messages = [
            {"role": "user", "content": "create a port"},
            {"role": "assistant", "content": "", "tool_calls": []},
            self._first(session, pending),
            _answer(pending, 1, {"q4": "twenty"}),
        ]
        assert _read(session, messages) == FormInput(
            # A picked option is the value behind it, typed as the form has it; picks of a list field are a
            # list in the order picked; what was typed travels as typed; "Keep default" sends nothing.
            values={"product": "uuid-10g", "policer": False, "ports": ["c", "a"], "vlan": "twenty"}
        )

    def test_typed_text_follows_the_picks_of_a_list_field(self):
        session, pending = _paused()
        typed = self._first(session, pending, q3=f"{_value(session, 3, 'Port B')}, the spare one, in Zwolle")
        values = _read(session, [typed, _answer(pending, 1, {"q4": "20"})])
        assert isinstance(values, FormInput) and values.values is not None
        assert values.values["ports"] == ["b", "the spare one, in Zwolle"]

    def test_a_typed_answer_that_names_one_choice_is_that_choice(self):
        session, pending = _paused()
        typed = self._first(session, pending, q0="sn8 1g", q1="T-1")
        values = _read(session, [typed, _answer(pending, 1, {"q4": "20"})])
        assert isinstance(values, FormInput) and values.values is not None
        assert values.values["product"] == "uuid-1g" and values.values["ticket_id"] == "T-1"

    def test_a_skipped_card_ends_the_form_when_a_field_is_required(self):
        session, pending = _paused()
        skipped = _answer(pending, 0, dict.fromkeys(["q0", "q1", "q2", "q3"], SKIPPED))
        assert _read(session, [skipped]) == FormInput(decision=Decision.CANCEL)

    def test_a_skipped_card_of_optional_fields_keeps_their_defaults(self):
        session, pending = _paused([AskField(name="note", question="Note", required=False, title="Note")])
        assert _read(session, [_answer(pending, 0, {"q0": SKIPPED})]) == FormInput(values={})

    def test_an_answer_for_another_stop_shows_this_one_again(self):
        session, _ = _paused()
        stale = {"role": "tool", "tool_call_id": "ask_0000_0", "content": '{"answers":{"q0":"x"}}'}
        assert _read(session, [stale]) == FormInput()

    def test_an_unanswered_or_unreadable_tool_message_shows_the_stop_again(self):
        session, pending = _paused()
        for content in ("", "not json", '{"answer":"legacy"}', "[]"):
            message = {"role": "tool", "tool_call_id": call_id(pending, 0), "content": content}
            assert _read(session, [message]) == FormInput()

    def test_a_call_librechat_did_not_accept_is_reported_not_repeated(self):
        session, pending = _paused()
        error = "Error: Received tool input did not match expected schema\n Please fix your mistakes."
        refused = _read(session, [{"role": "tool", "tool_call_id": call_id(pending, 0), "content": error}])
        assert isinstance(refused, ChatCompletionMessage) and not refused.tool_calls
        assert refused.content == f"{NOT_SHOWN}: {error}"

    def test_without_a_pending_stop_there_is_nothing_to_read(self):
        answered = {"ask_0000_0": '{"answers":{"q0":"x"}}'}
        assert read(None, answered) is None
        assert read(FormFillSession(workflow_key="w"), answered) is None


class TestApproval:
    def _decide(self, answer_of: Any) -> FormInput | ChatCompletionMessage | None:
        session, pending = _paused(reply=APPROVAL)
        return _read(session, [_answer(pending, 0, {"q0": answer_of(pending)})])

    def test_approve_starts_and_reject_cancels(self):
        assert self._decide(lambda p: _decision_value(p, APPROVE_LABEL)) == FormInput(decision=Decision.START)
        assert self._decide(lambda p: _decision_value(p, REJECT_LABEL)) == FormInput(decision=Decision.CANCEL)

    def test_skipping_the_approval_cancels(self):
        assert self._decide(lambda p: SKIPPED) == FormInput(decision=Decision.CANCEL)

    def test_typed_words_are_no_decision(self):
        for words in ("Approve", "yes", "approve, but make it 1G"):
            assert self._decide(lambda p, words=words: words) == FormInput()

    def test_the_value_of_another_stops_approve_is_no_decision(self):
        _, earlier = _paused(reply=APPROVAL)
        assert self._decide(lambda p: _decision_value(earlier, APPROVE_LABEL)) == FormInput()


class TestShown:
    def test_a_page_stop_is_a_line_above_its_card(self):
        assert shown(Reply(GATHERING, ask=PAGE)) == "**Workflow form `create_port`** — Port details"
        untitled = Reply('{"workflow_key":"create_port","status":"gathering","title":"unknown"}', ask=PAGE[:1])
        assert shown(untitled) == "**Workflow form `create_port`**"

    def test_the_approval_shows_what_will_be_submitted(self):
        assert shown(APPROVAL).splitlines() == [
            "**Start workflow `create_port` with these values?**",
            "",
            "| Field | Value |",
            "|---|---|",
            "| product | SN8 10G |",  # an id is shown as the label the form has for it
            "| vlan | 20 |",
            "",
            "Defaults that apply: policer = true",
        ]

    def test_the_approval_adds_the_workflows_own_summary_when_its_form_has_one(self):
        summary = {
            "headers": ["before", "after"],
            "labels": ["customer", "port_mode"],
            "columns": [["ACE", "untagged"], ["ACE", "tagged"]],
        }
        confirming = {"workflow_key": "modify_port", "status": "confirming", "values": {"port_mode": "tagged"}}
        reply = Reply(json.dumps({**confirming, "summary": [summary]}), approval=APPROVAL.approval)
        assert shown(reply).splitlines() == [
            "**Start workflow `modify_port` with these values?**",
            "",
            "| Field | Value |",  # what is approved is what is sent: always shown, whatever the summary leaves out
            "|---|---|",
            "| port_mode | tagged |",
            "",
            "The workflow's own summary:",
            "",
            "| | before | after |",
            "|---|---|---|",
            "| customer | ACE | ACE |",
            "| port_mode | untagged | tagged |",
        ]

    def test_the_outcome_is_a_sentence(self):
        started = Reply('{"workflow_key":"w","status":"started","process_id":"p-1"}')
        assert shown(started) == "Workflow `w` started. Process id: `p-1`"
        assert shown(Reply('{"workflow_key":"w","status":"cancelled"}')) == (
            "The `w` form was cancelled; nothing was started."
        )
        assert shown(Reply('{"workflow_key":"w","status":"failed","reason":"core is down"}')) == (
            "The `w` form failed: core is down"
        )
