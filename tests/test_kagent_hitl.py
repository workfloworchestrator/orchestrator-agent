"""Tests for the kagent human-in-the-loop extension payloads: skill stops out, human answers in."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest

from orchestrator_agent.adapters.kagent_hitl import (
    HITL_EXTENSION_URI,
    Answers,
    AskUserAnswer,
    AskUserResponse,
    PendingAsk,
    PendingQuestion,
    ToolApproval,
    ToolApprovalResponse,
    answers_as_values,
    approval_decision,
    approval_request,
    ask_request,
    parse_response,
    payload_metadata,
)
from orchestrator_agent.state import Approval, AskField

ASK = [
    AskField(
        name="redundancy",
        question="Redundancy (`redundancy`, required): one of ...",
        choices=["protected (Protected)", "unprotected (Unprotected)"],
    ),
    AskField(name="ticket_id", question="Ticket Id (`ticket_id`, optional): free text"),
    AskField(name=None, question="Which workflow should be started?", choices=["create_x — Create X"]),
]


def test_ask_request_matches_kagent_wire_shape():
    payload, pending = ask_request("req-1", ASK)
    wire = payload_metadata(payload)[HITL_EXTENSION_URI]
    assert wire == {
        "type": "ask_user_request",
        "id": "req-1",
        "questions": [
            {
                "question": ASK[0].question,
                "choices": ["protected (Protected)", "unprotected (Unprotected)"],
                "multiple": False,
            },
            {"question": ASK[1].question, "choices": [], "multiple": False},
            {"question": ASK[2].question, "choices": ["create_x — Create X"], "multiple": False},
        ],
    }
    assert pending == PendingAsk(
        id="req-1",
        kind="ask",
        questions=[PendingQuestion(field="redundancy"), PendingQuestion(field="ticket_id"), PendingQuestion()],
    )


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


def test_answers_become_the_text_contract():
    pending = PendingAsk(
        id="r",
        kind="ask",
        questions=[
            PendingQuestion(field="redundancy", options={"Protected": "protected", "Unprotected": "unprotected"}),
            PendingQuestion(field="ticket_id"),
            PendingQuestion(),
        ],
    )
    response = AskUserResponse(
        id="r",
        answers=[
            AskUserAnswer(answer=["Protected"]),
            AskUserAnswer(answer=[]),
            AskUserAnswer(answer=["create_x — Create X"]),
        ],
    )
    # A picked chip is a label and travels as its value; an empty answer sends nothing; free text follows verbatim.
    assert answers_as_values(pending, response) == Answers({"redundancy": "protected"}, "create_x — Create X")


def test_multiple_answers_travel_as_a_json_list_and_survive_commas_in_labels():
    pending = PendingAsk(
        id="r",
        kind="ask",
        questions=[
            PendingQuestion(field="nodes", multiple=True, options={"Amsterdam, Science Park": "ams", "Utrecht": "utr"})
        ],
    )
    response = AskUserResponse(id="r", answers=[AskUserAnswer(answer=["Amsterdam, Science Park", "Utrecht"])])
    answers = answers_as_values(pending, response)
    assert answers == Answers({"nodes": ["ams", "utr"]}, "")
    assert answers_as_values(pending, AskUserResponse(id="other", answers=[AskUserAnswer(answer=["a"])])) is None
    assert answers_as_values(pending, AskUserResponse(id="r", answers=[])) is None


def test_approval_decision_is_matched_by_id():
    pending = PendingAsk(id="r", kind="approval")
    response = ToolApprovalResponse(
        approvals=[
            ToolApproval(id="x", approved=True),
            ToolApproval(id="r", approved=False, rejection_reason='{"speed": "1000"}'),
        ]
    )
    assert approval_decision(pending, response) == ToolApproval(
        id="r", approved=False, rejection_reason='{"speed": "1000"}'
    )
    assert approval_decision(PendingAsk(id="none", kind="approval"), response) is None


def test_a_single_answer_question_takes_the_first_answer_and_kinds_do_not_cross():
    pending = PendingAsk(
        id="r", kind="ask", questions=[PendingQuestion(field="ticket_id"), PendingQuestion(field="note")]
    )
    response = AskUserResponse(id="r", answers=[AskUserAnswer(answer=["A", "B"]), AskUserAnswer(answer=[])])
    assert answers_as_values(pending, response) == Answers({"ticket_id": "A"}, "")  # first answer; none: nothing
    approvals = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=True)])
    assert approval_decision(pending, approvals) is None  # an ask is not answered by an approval
    assert answers_as_values(PendingAsk(id="r", kind="approval"), response) is None


def test_an_approval_resumes_as_a_decision_and_a_json_rejection_as_a_correction():
    from orchestrator_agent.adapters.a2a_hitl import resume
    from orchestrator_agent.state import Decision, FormFillSession

    session = FormFillSession(workflow_key="w", status="confirming", hitl_request={"id": "r", "kind": "approval"})
    yes = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=True)])
    no = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=False, rejection_reason="not now")])
    fix = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=False, rejection_reason='{"speed": "1000"}')])
    assert resume(session, yes, "Human input supplied") == ("Human input supplied", Decision.START, None)
    assert resume(session, no, "Human input supplied") == ("Human input supplied", Decision.CANCEL, None)
    assert resume(session, fix, "Human input supplied") == (
        "Human input supplied",
        None,
        {"speed": "1000"},
    )  # a correction
    other = ToolApprovalResponse(approvals=[ToolApproval(id="other", approved=True)])
    assert resume(session, other, "Human input supplied") == ("Human input supplied", None, None)  # not ours: re-asked
