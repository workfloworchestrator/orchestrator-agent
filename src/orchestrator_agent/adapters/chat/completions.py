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

"""Chat-completions adapter — the agent as the "model" of an OpenAI-compatible chat client (LibreChat).

A chat client has no A2A client; it calls ``POST /v1/chat/completions`` on a custom endpoint and shows
what comes back. This adapter drives the same capabilities-based agent as the A2A one, one turn per
request: the latest user message is the turn's input, the conversation's memory and open form are loaded
from the state persisted under the ``X-Conversation-Id`` header (LibreChat sends its conversation id
there when the endpoint is configured to), and the answer is one assistant message.

A workflow form is filled by the model in the conversation. Its start needs a person's approval, which
this route cannot show yet: a form can be prepared here, not started (the AG-UI route approves starts).

The reply is computed before the response starts, so a streamed response is the finished message in a few
chunks; the client shows it at once. The request is read with ``request``'s models; what goes back
is built from the OpenAI SDK's own response models, so the wire shape is theirs.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated, Literal

import structlog
from fastapi import Depends, FastAPI, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.shared import ErrorObject
from starlette.responses import JSONResponse, Response, StreamingResponse

from orchestrator_agent.adapters.chat.request import CHAT_MODEL, ChatRequest
from orchestrator_agent.persistence import PostgresStatePersistence
from orchestrator_agent.turn import NO_RESULTS, ConversationLocks, run_turn

if TYPE_CHECKING:
    from orchestrator_agent.agent import WFOAgent

logger = structlog.get_logger(__name__)

CONVERSATION_HEADER = "x-conversation-id"
# A chat client says who it is in a header its endpoint is configured to send (LibreChat: ``headers:`` of
# the custom endpoint in librechat.yaml). On its own it only sends the user agent of its HTTP library.
CLIENT_HEADER = "x-agent-client"
_bearer = HTTPBearer(auto_error=False)


class ChatCompletionsAdapter:
    """Serves the agent as an OpenAI-compatible chat model and adds its routes to a FastAPI app.

    Usage::

        ChatCompletionsAdapter(agent).add_routes(app)
    """

    def __init__(self, agent: "WFOAgent") -> None:
        self.agent = agent
        self._locks = ConversationLocks()

    def add_routes(self, app: FastAPI) -> None:
        app.add_api_route("/v1/chat/completions", self.chat, methods=["POST"])
        app.add_api_route("/v1/models", self.models, methods=["GET"])

    async def models(self) -> JSONResponse:
        return JSONResponse(
            {"object": "list", "data": [{"id": CHAT_MODEL, "object": "model", "owned_by": "orchestrator-agent"}]}
        )

    async def chat(
        self,
        body: ChatRequest,
        conversation: Annotated[str | None, Header(alias=CONVERSATION_HEADER)] = None,
        client: Annotated[str | None, Header(alias=CLIENT_HEADER)] = None,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    ) -> Response:
        try:
            message = await self.complete(
                body,
                conversation=conversation,
                client=client,
                auth_token=credentials.credentials if credentials else None,
            )
        except Exception:
            logger.exception("Chat completion failed")
            error = ErrorObject(message="The agent could not answer this request", type="server_error")
            return JSONResponse({"error": error.model_dump()}, status_code=500)
        if body.stream:
            return StreamingResponse(_chunks(body.model, message), media_type="text/event-stream")
        completion = _completion(body.model, message)
        return Response(completion.model_dump_json(exclude_unset=True), media_type="application/json")

    async def complete(
        self, body: ChatRequest, *, conversation: str | None, client: str | None = None, auth_token: str | None = None
    ) -> ChatCompletionMessage:
        """One turn of a conversation, as the assistant message that answers it.

        Without a conversation id the request is a conversation of its own. ``client`` is who the caller
        says it is (the ``X-Agent-Client`` header): it picks the transport a form is shown through.
        """
        thread = conversation or uuid.uuid4().hex
        async with self._locks.turn(thread):
            return await self._complete(body, thread, client, auth_token)

    async def _complete(
        self, body: ChatRequest, thread: str, client: str | None, auth_token: str | None
    ) -> ChatCompletionMessage:
        from orchestrator.core.db import db

        run_id = uuid.uuid4()
        try:
            persistence = PostgresStatePersistence(thread_id=thread, run_id=run_id, session=db.session)
            prior_state = await persistence.load_state()
            user_input = body.user_text

            state, output = await run_turn(
                self.agent,
                db.session,
                thread=thread,
                run_id=run_id,
                agent_type="chat",
                prior_state=prior_state,
                user_input=user_input,
                hitl=False,
                auth_token=auth_token,
            )
            message = _says(output or NO_RESULTS)
            await persistence.snapshot(state)
            db.session.commit()
            return message
        except Exception:
            db.session.rollback()
            raise


def _says(text: str) -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=text)


# --- the message on the wire: the OpenAI SDK's own response models -----------------------------------------


def _finish_reason(message: ChatCompletionMessage) -> Literal["stop", "tool_calls"]:
    return "tool_calls" if message.tool_calls else "stop"


def _completion(model: str, message: ChatCompletionMessage) -> ChatCompletion:
    return ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[Choice(index=0, message=message, finish_reason=_finish_reason(message))],
    )


def _chunks(model: str, message: ChatCompletionMessage) -> Iterator[str]:
    """The message as server-sent ``chat.completion.chunk`` events: its role, its text, its call, the end."""
    identifier, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

    def event(delta: ChoiceDelta, finish_reason: Literal["stop", "tool_calls"] | None = None) -> str:
        chunk = ChatCompletionChunk(
            id=identifier,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
        )
        return f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

    yield event(ChoiceDelta(role="assistant", content=""))
    if message.content:
        yield event(ChoiceDelta(content=message.content))
    for index, call in enumerate(message.tool_calls or []):
        if isinstance(call, ChatCompletionMessageFunctionToolCall):
            function = ChoiceDeltaToolCallFunction(name=call.function.name, arguments=call.function.arguments)
            delta_call = ChoiceDeltaToolCall(index=index, id=call.id, type="function", function=function)
            yield event(ChoiceDelta(tool_calls=[delta_call]))
    yield event(ChoiceDelta(), _finish_reason(message))
    yield "data: [DONE]\n\n"


__all__ = ["CLIENT_HEADER", "CONVERSATION_HEADER", "ChatCompletionsAdapter"]
