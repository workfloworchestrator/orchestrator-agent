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

"""A chat-completions request, as far as this agent reads it.

The agent keeps its own memory of a conversation, so of a request it reads little: the latest user
message, the results of the run's tool calls, and whether the chat offers a tool. What goes back is built
from the OpenAI SDK's response models (``completions``); the request is read with these,
which take any OpenAI-compatible client's body and ignore the rest of it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

CHAT_MODEL = "wfo"  # the model id listed at /v1/models; a request may name any


class ContentPart(BaseModel):
    """One part of a message's content; only its text is read."""

    model_config = ConfigDict(extra="ignore")

    type: str = "text"
    text: str | None = None


class ChatMessage(BaseModel):
    """One message of a request."""

    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | list[ContentPart] | None = None
    tool_call_id: str | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.text for part in self.content or [] if part.type == "text" and part.text)


class ChatRequest(BaseModel):
    """The body of a chat-completions request."""

    model_config = ConfigDict(extra="ignore")

    model: str = CHAT_MODEL
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None

    @property
    def user_text(self) -> str:
        """The latest user message; the agent keeps its own memory of the turns before it.

        A client may put a message of its own before it (LibreChat: the answers given to earlier
        questions); the last one is what the person wrote.
        """
        return next((message.text for message in reversed(self.messages) if message.role == "user"), "")

    @property
    def answers_a_call(self) -> bool:
        """Whether the request continues a run after a tool call: its last message is the tool's result."""
        return bool(self.messages) and self.messages[-1].role == "tool"

    @property
    def tool_results(self) -> dict[str, str]:
        """The results of the run's tool calls, by the id of the call each answers."""
        return {m.tool_call_id: m.text for m in self.messages if m.role == "tool" and m.tool_call_id}


__all__ = ["CHAT_MODEL", "ChatMessage", "ChatRequest", "ContentPart"]
