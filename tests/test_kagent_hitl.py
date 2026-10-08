"""Tests for the kagent human-in-the-loop extension payloads: skill stops out, human answers in."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest

from orchestrator_agent.adapters.a2a.kagent import (
    HITL_EXTENSION_URI,
    AskUserAnswer,
    AskUserResponse,
    ToolApproval,
    ToolApprovalResponse,
    answers_as_values,
    approval_decision,
    approval_request,
    ask_request,
    human_input,
    parse_response,
    pause,
    payload_metadata,
)
from orchestrator_agent.form_fill.pending import PendingAsk
from orchestrator_agent.state import Approval, AskField, Decision, FormFillSession, FormInput, Reply

ASK = [
    AskField(
        name="redundancy",
        question="Redundancy (`redundancy`, required)",
        choices=["Protected", "Unprotected"],
        values=["protected", "unprotected"],
    ),
    AskField(name="ticket_id", question="Ticket Id (`ticket_id`, optional — leave empty to keep the default)"),
    AskField(name="policer", question="Policer (`policer`, required)", choices=["True", "False"], values=[True, False]),
]


def test_ask_request_matches_kagent_wire_shape():
    payload, pending = ask_request("req-1", ASK)
    wire = payload_metadata(payload)[HITL_EXTENSION_URI]
    assert wire == {
        "type": "ask_user_request",
        "id": "req-1",
        "questions": [
            {"question": ASK[0].question, "choices": ["Protected", "Unprotected"], "multiple": False},
            {"question": ASK[1].question, "choices": [], "multiple": False},
            {"question": ASK[2].question, "choices": ["True", "False"], "multiple": False},
        ],
    }
    # What was asked is remembered as it was asked (the value behind each chip with it), and survives the
    # session's JSON round trip.
    assert pending == PendingAsk(id="req-1", kind="ask", questions=ASK)
    restored = PendingAsk.model_validate(pending.model_dump(mode="json"))
    assert [(q.name, list(q.choices), list(q.values)) for q in restored.questions] == [
        ("redundancy", ["Protected", "Unprotected"], ["protected", "unprotected"]),
        ("ticket_id", [], []),
        ("policer", ["True", "False"], [True, False]),
    ]


def test_approval_request_matches_kagent_wire_shape():
    approval = Approval(
        hint="All pages ...", tool_name="create_workflow", args={"workflow_key": "w", "json_data": [{"a": 1}]}
    )
    payload, pending = approval_request("req-2", approval)
    wire = payload_metadata(payload)
    assert wire[HITL_EXTENSION_URI] == {
        "type": "tool_approval_request",
        "hint": "All pages ...",
        "tools": [
            {
                "id": "req-2",
                "call_id": "req-2",
                "name": "create_workflow",
                "args": {"workflow_key": "w", "json_data": [{"a": 1}]},
            }
        ],
    }
    assert pending == PendingAsk(id="req-2", kind="approval")


@pytest.mark.parametrize(
    "metadata, expected",
    [
        pytest.param(None, None, id="no-metadata"),
        pytest.param({"x": 1}, None, id="no-extension-key"),
        pytest.param(
            {HITL_EXTENSION_URI: {"type": "ask_user_request", "id": "r"}}, None, id="a-request-is-not-a-response"
        ),
        pytest.param(
            {HITL_EXTENSION_URI: {"type": "ask_user_response", "answers": "nope"}}, None, id="malformed-is-ignored"
        ),
        pytest.param(
            {
                HITL_EXTENSION_URI: {
                    "type": "ask_user_response",
                    "id": "r",
                    "answers": [{"answer": ["protected (Protected)"]}],
                }
            },
            AskUserResponse(id="r", answers=[AskUserAnswer(answer=["protected (Protected)"])]),
            id="ask-user-response",
        ),
        pytest.param(
            {
                HITL_EXTENSION_URI: {
                    "type": "tool_approval_response",
                    "approvals": [{"id": "r", "approved": False, "rejection_reason": "no"}],
                }
            },
            ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=False, rejection_reason="no")]),
            id="tool-approval-response",
        ),
    ],
)
def test_parse_response(metadata, expected):
    assert parse_response(metadata) == expected


def test_answers_become_the_values_of_the_fields_asked():
    _, pending = ask_request("r", ASK)
    response = AskUserResponse(
        id="r",
        answers=[AskUserAnswer(answer=["Protected"]), AskUserAnswer(answer=[]), AskUserAnswer(answer=["False"])],
    )
    # A picked chip is a label and travels as its value (typed as the form has it); an empty answer sends nothing.
    assert answers_as_values(pending, response) == {"redundancy": "protected", "policer": False}
    typed = AskUserResponse(
        id="r",
        answers=[AskUserAnswer(answer=["the safe one"]), AskUserAnswer(answer=["T-1"]), AskUserAnswer(answer=[" "])],
    )
    assert answers_as_values(pending, typed) == {"redundancy": "the safe one", "ticket_id": "T-1"}  # as typed


def test_multiple_answers_stay_a_list_and_survive_commas_in_labels():
    pending = PendingAsk(
        id="r",
        kind="ask",
        questions=[
            AskField(
                name="nodes",
                question="Nodes?",
                multiple=True,
                choices=["Amsterdam, Science Park", "Utrecht"],
                values=["ams", "utr"],
            )
        ],
    )
    response = AskUserResponse(id="r", answers=[AskUserAnswer(answer=["Amsterdam, Science Park", "Utrecht"])])
    assert answers_as_values(pending, response) == {"nodes": ["ams", "utr"]}
    assert answers_as_values(pending, AskUserResponse(id="other", answers=[AskUserAnswer(answer=["a"])])) is None
    assert answers_as_values(pending, AskUserResponse(id="r", answers=[])) is None


def test_chips_that_share_a_label_are_told_apart_and_map_to_their_ids():
    # kagent answers with chip text, and core takes ids, never labels. Where two options carry the same
    # label (descriptions are not unique) the skill's chips carry the value too (``unique_choices``), so a
    # pick always names one option; only text that is no chip travels as typed, for core to judge.
    twins = AskField(
        name="nodes",
        question="Nodes?",
        multiple=True,
        choices=["Node X (id-1)", "Node X (id-2)", "Node Y"],
        values=["id-1", "id-2", "id-3"],
    )
    pending = PendingAsk(id="r", kind="ask", questions=[twins])
    picked = AskUserResponse(id="r", answers=[AskUserAnswer(answer=["Node X (id-2)", "Node Y"])])
    assert answers_as_values(pending, picked) == {"nodes": ["id-2", "id-3"]}
    typed = AskUserResponse(id="r", answers=[AskUserAnswer(answer=["Node X"])])
    assert answers_as_values(pending, typed) == {"nodes": ["Node X"]}


def test_approval_decision_is_matched_by_id():
    pending = PendingAsk(id="r", kind="approval")
    response = ToolApprovalResponse(
        approvals=[ToolApproval(id="x", approved=True), ToolApproval(id="r", approved=False, rejection_reason="no")]
    )
    assert approval_decision(pending, response) == ToolApproval(id="r", approved=False, rejection_reason="no")
    assert approval_decision(PendingAsk(id="none", kind="approval"), response) is None


def test_a_single_answer_question_takes_the_first_answer_and_kinds_do_not_cross():
    pending = PendingAsk(
        id="r",
        kind="ask",
        questions=[AskField(name="ticket_id", question="Ticket?"), AskField(name="note", question="Note?")],
    )
    response = AskUserResponse(id="r", answers=[AskUserAnswer(answer=["A", "B"]), AskUserAnswer(answer=[])])
    assert answers_as_values(pending, response) == {"ticket_id": "A"}  # first answer; none: nothing
    approvals = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=True)])
    assert approval_decision(pending, approvals) is None  # an ask is not answered by an approval
    assert answers_as_values(PendingAsk(id="r", kind="approval"), response) is None


class TestHumanInput:
    """The human's response as the skill's input: answers are values, an approval is a decision, words are nothing."""

    def test_an_approval_is_a_start_and_a_rejection_a_cancel_whatever_its_reason(self):
        session = FormFillSession(workflow_key="w", status="confirming", pending={"id": "r", "kind": "approval"})
        yes = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=True)])
        no = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=False, rejection_reason="not now")])
        fix = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=False, rejection_reason='{"a": "1"}')])
        assert human_input(session, yes) == FormInput(decision=Decision.START)
        assert human_input(session, no) == FormInput(decision=Decision.CANCEL)
        assert human_input(session, fix) == FormInput(decision=Decision.CANCEL)  # a reason is never read for values

    def test_a_response_that_is_not_about_the_pending_stop_shows_it_again(self):
        session = FormFillSession(workflow_key="w", status="confirming", pending={"id": "r", "kind": "approval"})
        other = ToolApprovalResponse(approvals=[ToolApproval(id="other", approved=True)])
        answers = AskUserResponse(id="r", answers=[AskUserAnswer(answer=["x"])])
        assert human_input(session, other) == FormInput()  # never a start
        assert human_input(session, answers) == FormInput()  # answers to an approval: neither values nor a start

    def test_a_response_that_cannot_be_read_still_shows_the_stop_again(self):
        # A message sent as a response whose payload is malformed is not a chat message: it must not end the form.
        session = FormFillSession(workflow_key="w", status="confirming", pending={"id": "r", "kind": "approval"})
        assert human_input(session, None, attempted=True) == FormInput()
        assert human_input(FormFillSession(workflow_key="w"), None, attempted=True) is None  # nothing was asked

    def test_no_response_or_no_paused_form_is_no_input(self):
        session = FormFillSession(workflow_key="w", status="confirming", pending={"id": "r", "kind": "approval"})
        yes = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=True)])
        assert human_input(session, None) is None
        assert human_input(None, yes) is None
        assert human_input(FormFillSession(workflow_key="w"), yes) is None  # nothing was asked

    def test_a_pause_is_remembered_and_marked_unseen_when_it_answers_a_response(self):
        session = FormFillSession(workflow_key="w")
        payload = pause(Reply("{}", ask=ASK), session, answered=False)
        assert payload.type == "ask_user_request" and session.pending["id"] == payload.id and not session.unseen
        payload = pause(Reply("{}", approval=Approval(hint="h", tool_name="t", args={})), session, answered=True)
        assert payload.type == "tool_approval_request" and session.pending["kind"] == "approval" and session.unseen
        assert pause(Reply('{"status":"started"}'), None, answered=True) is None  # a final reply pauses nothing
