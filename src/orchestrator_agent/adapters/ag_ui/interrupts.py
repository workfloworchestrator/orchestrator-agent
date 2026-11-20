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

"""The start of a workflow over AG-UI: the framework's tool approval, shown by the client as an interrupt.

AG-UI (protocol 0.1.19 and later; the JS client from 1.0) lets a run end waiting for a person: ``RUN_FINISHED``
carries an interrupt per thing to answer, and the client answers by starting the next run with a ``resume``
entry for it. pydantic-ai maps its tool approvals onto exactly that: a run that stops on ``start_workflow``
ends with a ``tool_call`` interrupt, and the entry that answers it becomes the call's result, an approval
or a denial with what the person said. This module only adds what that mapping does not know: which call
is waiting when a chat is reopened or a person writes instead of deciding.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ag_ui.core import Interrupt, ResumeEntry
from pydantic import BaseModel, StrictBool
from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.ui.ag_ui._interrupt import approval_to_interrupt
from pydantic_ai.ui.ag_ui._utils import INTERRUPT_ID_PREFIX

from orchestrator_agent.form_fill.pages import approval
from orchestrator_agent.form_fill.start import pending_start
from orchestrator_agent.state import FormPhase, SearchState

INTERRUPTS_VERSION = (0, 1, 19)  # the AG-UI release that added the interrupt lifecycle
_VERSION = re.compile(r"^\s*v?(\d+(?:\.\d+)*)")


class ApprovalAnswer(BaseModel):
    """The payload that resolves the interrupt: the decision on the start, as AG-UI shapes an approval.

    A rejection may say why, or what to change: the model reads it and acts on it.
    """

    approved: StrictBool  # a real JSON boolean; "yes" is no decision
    reason: str | None = None


def supports_interrupts(version: str | None) -> bool:
    """Whether a client that says it speaks this AG-UI version gets interrupts (and so can be shown an approval)."""
    match = _VERSION.match(version or "")
    return match is not None and tuple(int(part) for part in match.group(1).split(".")) >= INTERRUPTS_VERSION


def pending_interrupt(state: SearchState, history: Sequence[ModelMessage]) -> Interrupt | None:
    """The approval the thread waits on, as the client was shown it; None when it waits on none."""
    call = pending_start(history)
    session = state.form_fill
    if call is None or session is None or session.status != FormPhase.CONFIRMING:
        return None
    return approval_to_interrupt(call, {call.tool_call_id: approval(session).model_dump(mode="json")})


def answers(call: ToolCallPart | None, resume: Sequence[ResumeEntry] | None) -> list[ResumeEntry]:
    """The resume entries a run carries on to the framework: the answer to the pending start, and no other.

    An entry about anything else is stale (its form is over).
    """
    if call is None:
        return []
    interrupt_id = f"{INTERRUPT_ID_PREFIX}{call.tool_call_id}"
    return [entry for entry in resume or [] if entry.interrupt_id == interrupt_id]


def words(entry: ResumeEntry) -> str | None:
    """What the person wrote when they did not approve the start; None when they approved or wrote nothing."""
    payload = entry.payload if isinstance(entry.payload, dict) else {}
    reason = payload.get("reason")
    return reason if payload.get("approved") is False and isinstance(reason, str) and reason.strip() else None


def denial(call: ToolCallPart, reason: str) -> ResumeEntry:
    """The entry that declines the waiting start, with what the model is told about it."""
    interrupt_id = f"{INTERRUPT_ID_PREFIX}{call.tool_call_id}"
    return ResumeEntry(interrupt_id=interrupt_id, status="resolved", payload={"approved": False, "reason": reason})


def rejected(entry: ResumeEntry) -> bool:
    """Whether the person rejected the start outright: no approval, and not a word about what to change."""
    payload = entry.payload if isinstance(entry.payload, dict) else {}
    return entry.status == "cancelled" or (payload.get("approved") is False and not payload.get("reason"))


__all__ = [
    "ApprovalAnswer",
    "answers",
    "denial",
    "pending_interrupt",
    "rejected",
    "supports_interrupts",
    "words",
]
