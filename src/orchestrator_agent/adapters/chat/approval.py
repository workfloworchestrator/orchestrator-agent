# Copyright 2019-2026 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The start of a workflow over chat completions: the framework's tool approval, shown through a client tool.

A chat client that can ask its user a question offers a tool for it (LibreChat from v0.8.8, on a model spec
with ``askUserQuestion: true``: ``ask_user_question``). A reply that calls the tool pauses the client's run
and shows a card; the person's answer comes back as the tool message of the next request,
``{"answers": {<question id>: <string>}}``, a picked option as its ``value`` verbatim. A run that stops on
``start_workflow`` goes out as such a call, under the id of the call that waits, and the answer becomes that
call's result: an approval, a rejection, or what the person typed instead.
"""

from __future__ import annotations

import json
from typing import Any

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call import Function
from pydantic import BaseModel, ValidationError
from pydantic_ai import DeferredToolResults, ToolDenied
from pydantic_ai.messages import ToolCallPart

from orchestrator_agent.adapters.chat.request import ChatRequest
from orchestrator_agent.form_fill.start import NOT_APPROVED, REJECTED
from orchestrator_agent.state import ApprovalStop

ASK_TOOL = "ask_user_question"
SKIPPED = "The user chose not to answer this question."  # LibreChat's answer to a skipped card
QUESTION = "q0"
NOTE = "Nothing is started until you approve. Type an answer instead to say what should change."


class _Answers(BaseModel):
    answers: dict[str, str]


def can_ask(request: ChatRequest) -> bool:
    """Whether the chat offers the tool an approval is shown through."""
    return any((tool.get("function") or {}).get("name") == ASK_TOOL for tool in request.tools or [])


def card(call: ToolCallPart, stop: ApprovalStop) -> ChatCompletionMessage:
    """The assistant message that shows the start to approve: the values as text, the decision as a card."""
    values = "\n".join(f"- {name}: {_shown(value)}" for name, value in stop.values.items())
    question: dict[str, Any] = {
        "id": QUESTION,
        "header": "Approval",
        "question": f"Start workflow {stop.workflow_key} with these values?",
        "description": NOTE,
        "options": [
            {"label": "Approve", "value": f"{call.tool_call_id}:approve"},
            {"label": "Reject", "value": f"{call.tool_call_id}:reject"},
        ],
    }
    ask = ChatCompletionMessageFunctionToolCall(
        id=call.tool_call_id,
        type="function",
        function=Function(name=ASK_TOOL, arguments=json.dumps({"questions": [question]})),
    )
    return ChatCompletionMessage(role="assistant", content=values or None, tool_calls=[ask])


def decision(call: ToolCallPart, request: ChatRequest) -> tuple[DeferredToolResults | None, bool]:
    """The person's answer to the waiting start as the call's result, and whether it rejects the start outright.

    A picked option is the decision. Anything else the person wrote, on the card or as a message instead of
    an answer, declines the start for now with their words. No answer at all is no decision.
    """
    said = request.user_text.strip()
    if request.answers_a_call:
        try:
            answers = _Answers.model_validate_json(request.tool_results.get(call.tool_call_id, "")).answers
        except ValidationError:
            return None, False
        said = answers.get(QUESTION, "").strip()
    if said == f"{call.tool_call_id}:approve":
        return DeferredToolResults(approvals={call.tool_call_id: True}), False
    if said in (f"{call.tool_call_id}:reject", SKIPPED):
        return DeferredToolResults(approvals={call.tool_call_id: ToolDenied(REJECTED)}), True
    if not said:
        return None, False
    return DeferredToolResults(approvals={call.tool_call_id: ToolDenied(f"{NOT_APPROVED} {said!r}")}), False


def _shown(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


__all__ = ["ASK_TOOL", "SKIPPED", "can_ask", "card", "decision"]
