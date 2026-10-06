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

The workflow form-fill skill is answered by a person through the calling client's own way of asking
them: a human-in-the-loop transport (``adapters.hitl.HumanInTheLoop``), picked per request as the one
whose client is calling — LibreChat, through its ask-user tool (``librechat``). A stop of the skill goes out as what makes that client
show it, and the response it sends back is the skill's input. A request no transport claims is a caller
that cannot show a stop: no form is opened. A form turn runs no model, so nothing between the person's
click and the skill is a model's reading.

The reply is computed before the response starts, so a streamed response is the finished message in a few
chunks; the client shows it at once. The request is read with ``request``'s models; what goes back
is built from the OpenAI SDK's own response models, so the wire shape is theirs.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator, Mapping
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

from orchestrator_agent.adapters.chat.librechat import CLIENT, LibreChatHitl
from orchestrator_agent.adapters.chat.request import CHAT_MODEL, ChatRequest
from orchestrator_agent.adapters.hitl import HumanInTheLoop
from orchestrator_agent.persistence import PostgresStatePersistence
from orchestrator_agent.state import FormInput
from orchestrator_agent.turn import NO_RESULTS, ConversationLocks, run_turn

if TYPE_CHECKING:
    from orchestrator_agent.agent import WFOAgent

logger = structlog.get_logger(__name__)

CONVERSATION_HEADER = "x-conversation-id"
# A chat client says who it is in a header its endpoint is configured to send (LibreChat: ``headers:`` of
# the custom endpoint in librechat.yaml). On its own it only sends the user agent of its HTTP library.
CLIENT_HEADER = "x-agent-client"
FORM_RESPONSE = "[form response]"  # the input of a turn that is the person's response to a stop
NOT_OPEN = "That form is no longer open. Ask again to start it."

# A client's way of showing a stop, over this protocol: the request in, an assistant message out.
ChatHitl = HumanInTheLoop[ChatRequest, ChatCompletionMessage]

_bearer = HTTPBearer(auto_error=False)


class ChatCompletionsAdapter:
    """Serves the agent as an OpenAI-compatible chat model and adds its routes to a FastAPI app.

    ``transports`` are the clients a form can be shown through, by the name each gives itself in the
    ``X-Agent-Client`` header: the transport of the client that is calling is used.

    Usage::

        ChatCompletionsAdapter(agent).add_routes(app)
    """

    def __init__(self, agent: "WFOAgent", transports: Mapping[str, ChatHitl] | None = None) -> None:
        self.agent = agent
        self.transports: Mapping[str, ChatHitl] = {CLIENT: LibreChatHitl()} if transports is None else transports
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

        # The transport of the client that is calling, when this chat can show a person a stop (a form
        # needs one).
        transport = self.transports.get((client or "").strip().lower())
        if transport is not None and not transport.shows_stops(body):
            transport = None
        run_id = uuid.uuid4()
        try:
            persistence = PostgresStatePersistence(thread_id=thread, run_id=run_id, session=db.session)
            prior_state = await persistence.load_state()
            session = prior_state.form_fill if prior_state is not None else None

            user_input = body.user_text
            form_input: FormInput | None = None
            if body.answers_a_call:
                # The person answered a stop. The transport reads the answer as the skill's input for this
                # turn — or answers the request itself, while its client is not done showing the stop.
                response = transport.read(session, body) if transport is not None else None
                if isinstance(response, ChatCompletionMessage):
                    return response
                if response is None:
                    return _says(NOT_OPEN)
                user_input, form_input = FORM_RESPONSE, response

            state, output = await run_turn(
                self.agent,
                db.session,
                thread=thread,
                run_id=run_id,
                agent_type="chat",
                prior_state=prior_state,
                user_input=user_input,
                hitl=transport is not None,
                form_input=form_input,
                auth_token=auth_token,
            )
            # The model's text, or the form-fill skill's reply: as what makes the client show it when it is
            # a stop (recorded on the session, so before the snapshot), in the client's wording otherwise.
            message = _says(output or NO_RESULTS)
            if (reply := state.form_reply) is not None:
                stop = transport.pause(reply, state.form_fill, body) if transport is not None else None
                text = transport.words(reply) if transport is not None else reply.text
                message = stop if stop is not None else _says(text)
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


__all__ = ["CLIENT_HEADER", "CONVERSATION_HEADER", "ChatCompletionsAdapter", "ChatHitl"]
