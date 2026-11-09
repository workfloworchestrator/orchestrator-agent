"""Tests for the kagent human-in-the-loop extension payloads: the approval out, the human's decision in."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest

from orchestrator_agent.adapters.a2a.kagent import (
    HITL_EXTENSION_URI,
    ToolApproval,
    ToolApprovalResponse,
    approval_decision,
    approval_request,
    human_input,
    parse_response,
    pause,
    payload_metadata,
)
from orchestrator_agent.form_fill.pending import PendingStop
from orchestrator_agent.state import Approval, Decision, FormFillSession, FormInput, Reply

APPROVAL = Approval(
    hint="All pages ...", tool_name="create_workflow", args={"workflow_key": "w", "json_data": [{"a": 1}]}
)


def test_approval_request_matches_kagent_wire_shape():
    payload, pending = approval_request("req-2", APPROVAL)
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
    assert pending == PendingStop(id="req-2")


@pytest.mark.parametrize(
    "metadata, expected",
    [
        pytest.param(None, None, id="no-metadata"),
        pytest.param({"x": 1}, None, id="no-extension-key"),
        pytest.param(
            {HITL_EXTENSION_URI: {"type": "tool_approval_request", "tools": []}}, None, id="a-request-is-not-a-response"
        ),
        pytest.param(
            {HITL_EXTENSION_URI: {"type": "ask_user_response", "id": "r", "answers": []}},
            None,
            id="answers-are-not-a-decision",
        ),
        pytest.param(
            {HITL_EXTENSION_URI: {"type": "tool_approval_response", "approvals": "nope"}},
            None,
            id="malformed-is-ignored",
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


def test_approval_decision_is_matched_by_id():
    pending = PendingStop(id="r")
    response = ToolApprovalResponse(
        approvals=[ToolApproval(id="x", approved=True), ToolApproval(id="r", approved=False, rejection_reason="no")]
    )
    assert approval_decision(pending, response) == ToolApproval(id="r", approved=False, rejection_reason="no")
    assert approval_decision(PendingStop(id="none"), response) is None


class TestHumanInput:
    """The human's response as the skill's input: an approval is a start, a rejection a cancel with its reason."""

    SESSION = {"workflow_key": "w", "status": "confirming", "pending": {"id": "r"}}

    def test_an_approval_is_a_start_and_a_rejection_a_cancel_with_what_the_human_said(self):
        session = FormFillSession(**self.SESSION)
        yes = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=True)])
        no = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=False)])
        why = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=False, rejection_reason=" make it 1G ")])
        assert human_input(session, yes) == FormInput(decision=Decision.START)
        assert human_input(session, no) == FormInput(decision=Decision.CANCEL)
        assert human_input(session, why) == FormInput(decision=Decision.CANCEL, reason="make it 1G")

    def test_a_response_that_is_not_about_the_pending_stop_shows_it_again(self):
        session = FormFillSession(**self.SESSION)
        other = ToolApprovalResponse(approvals=[ToolApproval(id="other", approved=True)])
        assert human_input(session, other) == FormInput()  # never a start

    def test_a_response_that_cannot_be_read_still_shows_the_stop_again(self):
        # A message sent as a response whose payload is malformed is not a chat message: it must not end the form.
        session = FormFillSession(**self.SESSION)
        assert human_input(session, None, attempted=True) == FormInput()
        assert human_input(FormFillSession(workflow_key="w"), None, attempted=True) is None  # nothing was asked

    def test_no_response_or_no_paused_form_is_no_input(self):
        session = FormFillSession(**self.SESSION)
        yes = ToolApprovalResponse(approvals=[ToolApproval(id="r", approved=True)])
        assert human_input(session, None) is None
        assert human_input(None, yes) is None
        assert human_input(FormFillSession(workflow_key="w"), yes) is None  # nothing was asked

    def test_a_pause_is_remembered_and_marked_unseen_when_it_answers_a_response(self):
        session = FormFillSession(workflow_key="w", status="confirming")
        payload = pause(Reply("{}", approval=APPROVAL), session, answered=False)
        assert payload is not None and payload.type == "tool_approval_request"
        assert session.pending == {"id": payload.tools[0].id} and not session.unseen
        payload = pause(Reply("{}", approval=APPROVAL), session, answered=True)
        assert payload is not None and session.unseen
        assert pause(Reply('{"status":"started"}'), session, answered=True) is None  # a final reply pauses nothing
        assert pause(Reply("{}", approval=APPROVAL), None, answered=True) is None
