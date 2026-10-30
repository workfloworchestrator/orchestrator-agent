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

"""A stored conversation as the chat popup reads it back: its title, its transcript, its pending stop.

The agent keeps a thread as the model's memory (pydantic-ai messages). A client that reopens the thread
needs it as AG-UI messages, the shape it renders from the stream: pydantic-ai's adapter converts them, and
two things are put the way the stream put them. A tool result that carried an artifact is the artifact's
reference again, not the raw payload the model saw; and the form-fill skill's replies, one JSON object
each, are the words the person read.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any

from ag_ui.core import Message
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.ui.ag_ui import AGUIAdapter

from orchestrator_agent.adapters.words import shown
from orchestrator_agent.artifacts import DataArtifact, ExportArtifact, QueryArtifact
from orchestrator_agent.persistence import load_messages
from orchestrator_agent.state import Reply, SearchState

UNTITLED = "New chat"
TITLE_LENGTH = 80


class ChatSummary(BaseModel):
    """A chat as the list shows it (``GET /chats/``)."""

    thread_id: str
    title: str
    updated_at: datetime
    form_open: bool  # a workflow form waits on the person


class Chat(ChatSummary):
    """A chat as it is opened (``GET /chats/{id}``): its last messages, and the form stop to show again."""

    messages: list[dict[str, Any]]  # AG-UI messages, the last ``last`` of the thread
    pending: dict[str, Any] | None  # the form stop to show again, as an AG-UI interrupt


_ARTIFACT: TypeAdapter[QueryArtifact | ExportArtifact | DataArtifact] = TypeAdapter(
    QueryArtifact | ExportArtifact | DataArtifact
)


def title_of(state: SearchState) -> str:
    """What the chat is called in a list: the first thing the person asked, cut short."""
    for message in load_messages(state.message_history) or []:
        if isinstance(message, ModelRequest):
            for part in _the_persons(message):
                if isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content.strip():
                    text = " ".join(part.content.split())
                    return text if len(text) <= TITLE_LENGTH else text[: TITLE_LENGTH - 1] + "…"
    return UNTITLED


def transcript(state: SearchState, *, last: int | None = None) -> list[Message]:
    """The thread's messages as AG-UI messages, the last ``last`` of them (all when None)."""
    messages = [_for_the_person(message) for message in load_messages(state.message_history) or []]
    agui = AGUIAdapter.dump_messages(messages)
    return agui[-last:] if last else agui


def pending_stop(state: SearchState) -> dict[str, Any] | None:
    """The form stop the thread waits on, as the client was shown it; None when no form is open."""
    session = state.form_fill
    if session is None or session.status == "done" or not session.pending:
        return None
    shown_as = session.pending.get("shown")
    return dict(shown_as) if isinstance(shown_as, dict) else None


def _for_the_person(message: ModelMessage) -> ModelMessage:
    """The message as the stream showed it: artifact references for tool results, words for the skill's replies."""
    if isinstance(message, ModelResponse):
        said = [
            dataclasses.replace(part, content=shown(Reply(part.content))) if isinstance(part, TextPart) else part
            for part in message.parts
        ]
        return dataclasses.replace(message, parts=said)
    asked = [_reference(part) if isinstance(part, ToolReturnPart) else part for part in _the_persons(message)]
    return dataclasses.replace(message, parts=asked)


def _the_persons(message: ModelRequest) -> list[ModelRequestPart]:
    """The request's parts without a tool's note to the model.

    A tool result can carry extra words for the model (``ToolReturn.content``: "a chart is shown below your
    reply, summarise it"); pydantic-ai files them as a user prompt part next to the tool return. Nobody
    typed them, so a request that carries tool returns keeps none of its user prompt parts.
    """
    if any(isinstance(part, ToolReturnPart) for part in message.parts):
        return [part for part in message.parts if not isinstance(part, UserPromptPart)]
    return list(message.parts)


def _reference(part: ToolReturnPart) -> ToolReturnPart:
    """A tool result that carried an artifact, as its lightweight reference (what the stream sent)."""
    try:
        artifact = _ARTIFACT.validate_python(part.metadata)
    except ValidationError:
        return part
    return dataclasses.replace(part, content=artifact.model_dump_json(), metadata=None)


__all__ = ["UNTITLED", "Chat", "ChatSummary", "pending_stop", "title_of", "transcript"]
