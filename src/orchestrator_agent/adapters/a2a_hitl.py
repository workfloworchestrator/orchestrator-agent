# Copyright 2019-2026 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""kagent's human-in-the-loop extension on the A2A executor's side.

The form-fill skill itself runs inside the agent (``form_fill.capability``); what is left for the
transport is reading the response payload, keeping a paused form with its task, turning the human's
response into the contract the skill reads (a JSON object keyed by field name, ``yes`` / ``no``), and
turning a stop into the pause payload.
"""

from __future__ import annotations

import uuid

import structlog
from a2a.server.agent_execution import RequestContext
from google.protobuf import json_format

from orchestrator_agent.form_fill.hitl import (
    AskUserRequest,
    HITLResponse,
    PendingAsk,
    ToolApprovalRequest,
    ToolApprovalResponse,
    answers_as_text,
    approval_as_text,
    approval_request,
    ask_request,
    parse_response,
)
from orchestrator_agent.state import FormFillSession, Reply, SearchState

logger = structlog.get_logger(__name__)


def response_in(context: RequestContext) -> HITLResponse | None:
    """The human's response carried by this message's metadata, if it is one (the SDK hands it over as a struct)."""
    if context.message is None:
        return None
    return parse_response(json_format.MessageToDict(context.message.metadata))


def prior_session(
    prior_state: SearchState | None, hitl: bool, task_id: str
) -> tuple[FormFillSession | None, FormFillSession | None]:
    """(the form session this turn continues, the one it must leave alone).

    A human-in-the-loop form lives in one task that pauses and resumes, so a session of another task is
    not continued — but kept, for that task.
    """
    session = prior_state.form_fill if prior_state is not None else None
    if hitl and session is not None and session.task_id not in (None, task_id):
        return None, session
    return session, None


def resume_text(session: FormFillSession | None, response: HITLResponse, user_input: str) -> str:
    """The text the skill reads for the human's response to the pending stop; the message itself if none is pending."""
    if session is None or not session.hitl_request:
        return user_input
    pending = PendingAsk.model_validate(session.hitl_request)
    if isinstance(response, ToolApprovalResponse):
        mapped = approval_as_text(pending, response)
    else:
        mapped = answers_as_text(pending, response)
    if mapped is None:
        logger.warning("HITL response does not match the pending request; re-asking", pending_id=pending.id)
        return user_input
    return mapped


def stop_payload(reply: Reply, state: SearchState) -> AskUserRequest | ToolApprovalRequest | None:
    """The pause payload for a stop (ask or approval), remembered on the session; None when the reply is final.

    It records the pending request on the session, so it must run before the state is persisted.
    """
    session = state.form_fill
    if session is None:
        return None
    if reply.ask is None and reply.approval is None:
        session.hitl_request = None
        return None
    request_id = uuid.uuid4().hex
    payload: AskUserRequest | ToolApprovalRequest
    if reply.approval is not None:
        payload, pending = approval_request(request_id, reply.approval)
    else:
        payload, pending = ask_request(request_id, reply.ask or [])
    session.hitl_request = pending.model_dump(mode="json")
    return payload


__all__ = ["prior_session", "response_in", "resume_text", "stop_payload"]
