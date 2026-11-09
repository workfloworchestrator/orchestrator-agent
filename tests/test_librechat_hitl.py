"""LibreChat's ask-user tool as a human-in-the-loop transport: the approval as a card, the tool message as the decision."""

from __future__ import annotations

import json
from typing import Any

import pytest
from openai.types.chat import ChatCompletionMessage

from orchestrator_agent.adapters.chat.librechat import (
    APPROVE_LABEL,
    NOT_SHOWN,
    REJECT_LABEL,
    SKIPPED,
    LibreChatHitl,
    approval_card,
    call_id,
    pause,
    read,
    shown,
)
from orchestrator_agent.adapters.chat.request import ChatRequest
from orchestrator_agent.form_fill.pending import PendingStop
from orchestrator_agent.state import Approval, Decision, FormFillSession, FormInput, Reply

APPROVAL = Reply(
    '{"workflow_key":"create_port","status":"confirming","values":{"product":"uuid-10g","vlan":20},'
    '"labels":{"product":"SN8 10G"},"defaults":{"policer":true}}',
    approval=Approval(hint="Start workflow `create_port` with these values?", tool_name="create_workflow", args={}),
)


def _paused() -> tuple[FormFillSession, PendingStop]:
    """A session paused at the approval, as the skill leaves it."""
    session = FormFillSession(workflow_key="create_port", status="confirming", unseen=True)
    pending = pause(APPROVAL, session)
    assert pending is not None
    return session, pending


def _decision_value(pending: PendingStop, label: str) -> str:
    assert APPROVAL.approval is not None
    (question,) = approval_card(pending, APPROVAL.approval)["questions"]
    return next(option["value"] for option in question["options"] if option["label"] == label)


def _answer(pending: PendingStop, answer: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id(pending), "content": json.dumps({"answers": {"q0": answer}})}


def _read(session: FormFillSession, messages: list[dict[str, Any]]) -> FormInput | ChatCompletionMessage | None:
    """``read`` over the messages of a request, as the endpoint hands them on: the tool results by call id."""
    return read(session, {m["tool_call_id"]: m["content"] for m in messages if m.get("role") == "tool"})


class TestCard:
    def test_a_chat_that_offers_a_tool_can_show_a_card(self):
        transport = LibreChatHitl()
        with_tool = ChatRequest.model_validate(
            {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function", "function": {"name": "t"}}]}
        )
        without = ChatRequest.model_validate({"messages": [{"role": "user", "content": "x"}]})
        assert transport.shows_stops(with_tool) is True and transport.shows_stops(without) is False

    def test_the_approval_is_one_card_with_two_options(self):
        session, pending = _paused()
        assert session.pending == {"id": pending.id} and session.unseen is False  # a card is shown at once
        (question,) = approval_card(pending, APPROVAL.approval)["questions"]
        assert question["question"] == "Start workflow create_port with these values?"
        assert [option["label"] for option in question["options"]] == [APPROVE_LABEL, REJECT_LABEL]
        assert "type it here" in question["description"]

    def test_the_card_goes_out_as_a_call_under_the_persons_words(self):
        session = FormFillSession(workflow_key="create_port", status="confirming")
        message = LibreChatHitl().pause(APPROVAL, session, ChatRequest.model_validate({"messages": []}))
        assert message is not None and message.content == shown(APPROVAL)
        (call,) = message.tool_calls
        assert call.function.name == "ask_user_question" and call.id == call_id(
            PendingStop.model_validate(session.pending)
        )

    def test_a_reply_that_is_no_stop_is_no_card(self):
        session = FormFillSession(workflow_key="w", status="done")
        assert pause(Reply('{"workflow_key":"w","status":"started"}'), session) is None
        assert pause(APPROVAL, None) is None
        assert session.pending is None


class TestRead:
    def _decide(self, answer_of: Any) -> FormInput | ChatCompletionMessage | None:
        session, pending = _paused()
        return _read(session, [_answer(pending, answer_of(pending))])

    def test_approve_starts_and_reject_cancels(self):
        assert self._decide(lambda p: _decision_value(p, APPROVE_LABEL)) == FormInput(decision=Decision.START)
        assert self._decide(lambda p: _decision_value(p, REJECT_LABEL)) == FormInput(decision=Decision.CANCEL)

    def test_skipping_the_approval_cancels(self):
        assert self._decide(lambda p: SKIPPED) == FormInput(decision=Decision.CANCEL)

    @pytest.mark.parametrize("words", ["make it 1G", "Approve"])
    def test_typed_words_are_a_rejection_the_model_reads(self, words):
        # "Approve" typed is not the Approve option: an option's value is a token only this stop knows.
        assert self._decide(lambda p: words) == FormInput(decision=Decision.CANCEL, reason=words)

    def test_words_typed_next_to_a_pick_are_what_counts(self):
        rejected = self._decide(lambda p: f"{_decision_value(p, REJECT_LABEL)}, make it 1G")
        assert rejected == FormInput(decision=Decision.CANCEL, reason="make it 1G")
        approved = self._decide(lambda p: f"{_decision_value(p, APPROVE_LABEL)}, but note the ticket")
        assert approved == FormInput(decision=Decision.CANCEL, reason="but note the ticket")  # not a start

    def test_the_value_of_another_stops_approve_is_typed_words(self):
        _, earlier = _paused()
        decided = self._decide(lambda p: _decision_value(earlier, APPROVE_LABEL))
        assert decided == FormInput(decision=Decision.CANCEL, reason=_decision_value(earlier, APPROVE_LABEL))

    def test_an_unanswered_or_unreadable_tool_message_shows_the_stop_again(self):
        session, pending = _paused()
        assert _read(session, []) == FormInput()
        unreadable = {"role": "tool", "tool_call_id": call_id(pending), "content": "not json"}
        assert _read(session, [unreadable]) == FormInput()
        empty = {"role": "tool", "tool_call_id": call_id(pending), "content": ""}
        assert _read(session, [empty]) == FormInput()

    def test_a_call_librechat_did_not_accept_is_reported_not_repeated(self):
        session, pending = _paused()
        error = {"role": "tool", "tool_call_id": call_id(pending), "content": "Error: schema mismatch"}
        answer = _read(session, [error])
        assert isinstance(answer, ChatCompletionMessage) and answer.content == f"{NOT_SHOWN}: Error: schema mismatch"

    def test_without_a_pending_stop_there_is_nothing_to_read(self):
        _, pending = _paused()
        assert _read(FormFillSession(workflow_key="w"), [_answer(pending, "x")]) is None
        assert _read(None, [_answer(pending, "x")]) is None


class TestShown:
    def test_the_approval_shows_what_will_be_submitted(self):
        assert shown(APPROVAL).splitlines() == [
            "**Start workflow `create_port` with these values?**",
            "",
            "| Field | Value |",
            "|---|---|",
            "| product | SN8 10G |",  # an id is shown as the label the form has for it
            "| vlan | 20 |",
            "| policer | true *(default)* |",  # what the form fills in by itself, in the same table
        ]

    def test_an_empty_default_reads_as_none(self):
        reply = Reply(
            '{"workflow_key":"create_port","status":"confirming","values":{"vlan":20},'
            '"defaults":{"native_vlan":null,"contacts":[],"ticket_id":""}}'
        )
        assert shown(reply).splitlines()[-3:] == [
            "| native_vlan | none *(default)* |",
            "| contacts | none *(default)* |",
            "| ticket_id | none *(default)* |",
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

    def test_a_start_core_refused_says_what_it_rejected(self):
        refused = Reply(
            '{"workflow_key":"w","status":"gathering","page":1,"title":"Port",'
            '"rejected":[{"loc":["port"],"msg":"Port is taken","type":"value_error"}]}'
        )
        assert (
            shown(refused) == "**Workflow form `w`** (page 1, Port): the orchestrator rejected it. port: Port is taken"
        )

    def test_the_outcome_is_a_sentence(self):
        started = Reply('{"workflow_key":"w","status":"started","process_id":"p-1"}')
        assert shown(started) == "Workflow `w` started. Process id: `p-1`"
        assert shown(Reply('{"workflow_key":"w","status":"cancelled"}')) == (
            "The `w` form was cancelled; nothing was started."
        )
        assert shown(Reply('{"workflow_key":"w","status":"failed","reason":"core is down"}')) == (
            "The `w` form failed: core is down"
        )
        prepared = Reply(
            '{"workflow_key":"w","status":"prepared","reason":"this client cannot approve a start, so nothing was started","values":{"a":1}}'
        )
        assert shown(prepared).splitlines()[0] == (
            "**The `w` form is complete, but this client cannot approve a start, so nothing was started.**"
        )
        assert shown(Reply("plain words")) == "plain words"
