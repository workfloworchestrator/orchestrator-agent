"""Tests for the kagent human-in-the-loop extension payloads: skill stops out, human answers in."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest

from orchestrator_agent.form_fill.hitl import (
    HITL_EXTENSION_URI,
    AskUserAnswer,
    AskUserResponse,
    PendingAsk,
    ToolApproval,
    ToolApprovalResponse,
    answers_as_text,
    approval_as_text,
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
        id="req-1", kind="ask", fields=["redundancy", "ticket_id", None], multiple=[False] * 3, options=[None] * 3
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
    assert pending == PendingAsk(id="req-2", kind="approval", fields=[])


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
        fields=["redundancy", "ticket_id", None],
        options=[{"Protected": "protected", "Unprotected": "unprotected"}, None, None],
    )
    response = AskUserResponse(
        id="r",
        answers=[
            AskUserAnswer(answer=["Protected"]),
            AskUserAnswer(answer=[]),
            AskUserAnswer(answer=["create_x — Create X"]),
        ],
    )
    # A picked chip is a label and travels as its value; an empty answer keeps the default; free text follows verbatim.
    assert answers_as_text(pending, response) == '{"redundancy": "protected", "ticket_id": ""}\ncreate_x — Create X'


def test_multiple_answers_travel_as_a_json_list_and_survive_commas_in_labels():
    from orchestrator_agent.form_fill.contract import raw_pairs

    pending = PendingAsk(
        id="r",
        kind="ask",
        fields=["nodes"],
        multiple=[True],
        options=[{"Amsterdam, Science Park": "ams", "Utrecht": "utr"}],
    )
    response = AskUserResponse(id="r", answers=[AskUserAnswer(answer=["Amsterdam, Science Park", "Utrecht"])])
    text = answers_as_text(pending, response)
    assert text == '{"nodes": ["ams", "utr"]}'
    assert raw_pairs(text)["nodes"] == ["ams", "utr"]
    assert answers_as_text(pending, AskUserResponse(id="other", answers=[AskUserAnswer(answer=["a"])])) is None
    assert answers_as_text(pending, AskUserResponse(id="r", answers=[])) is None


def test_approval_decision_is_matched_by_id():
    pending = PendingAsk(id="r", kind="approval")
    response = ToolApprovalResponse(
        approvals=[
            ToolApproval(id="x", approved=True),
            ToolApproval(id="r", approved=False, rejection_reason='{"speed": "1000"}'),
        ]
    )
    assert approval_decision(pending, response) == (False, '{"speed": "1000"}')
    assert approval_decision(PendingAsk(id="none", kind="approval"), response) is None


def test_a_single_answer_question_takes_the_first_answer_and_kinds_do_not_cross():
    pending = PendingAsk(id="r", kind="ask", fields=["ticket_id", "note"], multiple=[False, False])
    response = AskUserResponse(id="r", answers=[AskUserAnswer(answer=["A", "B"]), AskUserAnswer(answer=[])])
    assert answers_as_text(pending, response) == '{"ticket_id": "A", "note": ""}'  # first answer; none is empty
    approvals = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=True)])
    assert approval_decision(pending, approvals) is None  # an ask is not answered by an approval
    assert answers_as_text(PendingAsk(id="r", kind="approval"), response) is None


def test_an_approval_becomes_the_contract_words():
    pending = PendingAsk(id="r", kind="approval")
    yes = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=True)])
    no = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=False, rejection_reason="not now")])
    fix = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=False, rejection_reason='{"speed": "1000"}')])
    assert approval_as_text(pending, yes) == "yes"
    assert approval_as_text(pending, no) == "no"
    assert approval_as_text(pending, fix) == '{"speed": "1000"}'  # a rejection carrying a JSON object is a correction
    assert approval_as_text(PendingAsk(id="other", kind="approval"), yes) is None
