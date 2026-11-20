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

"""LibreChat's ask-user tool: how a chat in LibreChat is shown the start of a workflow to approve.

LibreChat (from v0.8.8) gives a chat on a model spec with ``askUserQuestion: true`` one function tool,
``ask_user_question``. A reply that calls it pauses LibreChat's run and shows a card; the person's answer
comes back as the tool message of the next request,
``{"answers": {<question id>: <string>}}``, a picked option as its ``value`` verbatim. A run that stops on
``start_workflow`` goes out as such a call, under the id of the call that waits, and the answer becomes that
call's result: an approval, a rejection, or what the person typed instead. Everything here is LibreChat's
contract; what a decision means for the form is shared (``form_fill.start``).
"""

from __future__ import annotations

import json
from typing import Any

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call import Function
from pydantic import BaseModel, ValidationError
from pydantic_ai import DeferredToolResults
from pydantic_ai.messages import ToolCallPart

from orchestrator_agent.adapters.chat.request import ChatRequest
from orchestrator_agent.form_fill.start import answer, listed, question
from orchestrator_agent.state import ApprovalStop

ASK_TOOL = "ask_user_question"
SKIPPED = "The user chose not to answer this question."  # LibreChat's answer to a skipped card
QUESTION = "q0"
NOTE = "Nothing is started until you approve. Type an answer instead to say what should change."


class _Answers(BaseModel):
    answers: dict[str, str]


def offers_ask_tool(request: ChatRequest) -> bool:
    """Whether the chat offers LibreChat's ask-user tool: only then can it be shown an approval."""
    return any((tool.get("function") or {}).get("name") == ASK_TOOL for tool in request.tools or [])


def card(call: ToolCallPart, stop: ApprovalStop) -> ChatCompletionMessage:
    """The assistant message that shows the start to approve: the values as text, the decision as a card."""
    asked: dict[str, Any] = {
        "id": QUESTION,
        "header": "Approval",
        "question": question(stop),
        "description": NOTE,
        "options": [
            {"label": "Approve", "value": f"{call.tool_call_id}:approve"},
            {"label": "Reject", "value": f"{call.tool_call_id}:reject"},
        ],
    }
    ask = ChatCompletionMessageFunctionToolCall(
        id=call.tool_call_id,
        type="function",
        function=Function(name=ASK_TOOL, arguments=json.dumps({"questions": [asked]})),
    )
    return ChatCompletionMessage(role="assistant", content=listed(stop) or None, tool_calls=[ask])


def decision(call: ToolCallPart, request: ChatRequest) -> tuple[DeferredToolResults | None, bool]:
    """The answer on LibreChat's card as the result of the waiting start, and whether it rejects it outright.

    A picked option is the decision; anything else typed on the card declines the start for now with those
    words. A request that carries no answer to this card is no decision.
    """
    try:
        answers = _Answers.model_validate_json(request.tool_results.get(call.tool_call_id, "")).answers
    except ValidationError:
        return None, False
    said = answers.get(QUESTION, "").strip()
    if not request.answers_a_call or not said:
        return None, False
    if said in (f"{call.tool_call_id}:reject", SKIPPED):
        said = ""
    return answer(call, said == f"{call.tool_call_id}:approve", said)


__all__ = ["ASK_TOOL", "SKIPPED", "card", "decision", "offers_ask_tool"]
