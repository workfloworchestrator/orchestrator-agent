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

"""The start that waits for the person: which call it is, and what the model is told when they did not approve.

The start is the framework's tool approval (``skill.start``). Every route that can show an approval needs the
same two things around it: the ``start_workflow`` call a thread waits on, and the words the call's result
carries when the person rejected the start or wrote something instead of deciding.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from orchestrator_agent.tool_names import START_WORKFLOW_TOOL

# What the model is told, as the result of the start that was waiting, when the person did not approve it.
REJECTED = "The person rejected the start. The form is closed and nothing was started."
NOT_APPROVED = "Not approved: nothing was started. The person wants to talk about it first and wrote:"


def pending_start(history: Sequence[ModelMessage]) -> ToolCallPart | None:
    """The ``start_workflow`` call the thread waits on: the last one, while it has no result yet."""
    answered = {
        part.tool_call_id
        for message in history
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    calls = [
        part
        for message in history
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == START_WORKFLOW_TOOL
    ]
    return calls[-1] if calls and calls[-1].tool_call_id not in answered else None


__all__ = ["NOT_APPROVED", "REJECTED", "pending_start"]
