"""How a protocol adapter picks among its human-in-the-loop transports.

Each client's own transport is covered where it is used (``test_kagent_hitl`` and the A2A executor tests,
``test_librechat_hitl`` and ``test_chat_completions``); this is what only a second transport shows: the
one whose client is calling is used, and a transport may answer a request itself.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueueLegacy
from a2a.types import Message, Part, Role, SendMessageRequest, TaskState, TaskStatusUpdateEvent
from google.protobuf import json_format
from openai.types.chat import ChatCompletionMessage

from orchestrator_agent.adapters.a2a import A2AHitl, WFOAgentExecutor
from orchestrator_agent.adapters.chat import ChatCompletionsAdapter, ChatHitl, ChatRequest, LibreChatHitl
from orchestrator_agent.adapters.chat.librechat import shown
from orchestrator_agent.state import AskField, FormFillSession, FormInput, Reply, SearchState

from .conftest import make_text_result_event, mock_event_stream

ASK = Reply(
    '{"workflow_key":"w","status":"gathering","title":"Page"}',
    ask=[AskField(name="redundancy", question="Redundancy?", choices=["Protected"], values=["protected"])],
)
ASK_TOOL = [{"type": "function", "function": {"name": "ask_user_question"}}]


def _chat_request(*messages: dict[str, Any]) -> ChatRequest:
    body = {"messages": list(messages) or [{"role": "user", "content": "x"}], "tools": ASK_TOOL}
    return ChatRequest.model_validate(body)


def _agent(setup: Any = None) -> MagicMock:
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)

    @asynccontextmanager
    async def _run(*args: Any, **kwargs: Any):
        agent.state = kwargs["deps"].state
        if setup is not None:
            setup(agent.state)
        yield mock_event_stream(make_text_result_event("model text"))

    agent.run_stream_events = MagicMock(side_effect=_run)
    return agent


def _stops(state: SearchState) -> None:
    state.form_fill = state.form_fill or FormFillSession(workflow_key="w")
    state.form_reply = ASK


class _OtherChatClient:
    """A second chat client with its own way of asking a person, for the adapter to pick."""

    def shows_stops(self, request: ChatRequest) -> bool:
        return True

    def read(self, session: FormFillSession | None, request: ChatRequest) -> FormInput | ChatCompletionMessage | None:
        return FormInput(values={"from": "other"}) if request.answers_a_call else None

    def pause(
        self, reply: Reply, session: FormFillSession | None, request: ChatRequest
    ) -> ChatCompletionMessage | None:
        return ChatCompletionMessage(role="assistant", content="shown the other client's way")

    def words(self, reply: Reply) -> str:
        return "the other client's words"


class TestChatAdapterPicksTheTransport:
    TRANSPORTS: dict[str, ChatHitl] = {"librechat": LibreChatHitl(), "other": _OtherChatClient()}

    async def _complete(self, agent: MagicMock, request: ChatRequest, client: str) -> ChatCompletionMessage:
        with (
            patch("orchestrator_agent.adapters.chat.completions.PostgresStatePersistence") as persistence,
            patch("orchestrator.core.db.db"),
        ):
            persistence.return_value.load_state = AsyncMock(return_value=None)
            persistence.return_value.snapshot = AsyncMock()
            adapter = ChatCompletionsAdapter(agent, transports=self.TRANSPORTS)
            return await adapter.complete(request, conversation="conv-1", client=client)

    @pytest.mark.asyncio
    async def test_the_client_that_is_calling_shows_the_stop_its_own_way(self):
        librechat = await self._complete(_agent(_stops), _chat_request(), "librechat")
        assert librechat.tool_calls and librechat.content == shown(ASK)
        other = await self._complete(_agent(_stops), _chat_request(), "other")
        assert other.content == "shown the other client's way" and not other.tool_calls

    @pytest.mark.asyncio
    async def test_its_response_is_read_by_that_client(self):
        agent = _agent()
        result = {"role": "tool", "tool_call_id": "any", "content": "whatever the other client sends"}
        await self._complete(agent, _chat_request(result), "other")
        assert agent.state.form_input == FormInput(values={"from": "other"}) and agent.state.hitl is True


class _InterimA2A:
    """An A2A transport that is not done showing its stop: it answers the message itself."""

    PAUSE: dict[str, Any] = {"urn:test:hitl": {"more": True}}

    def shows_stops(self, request: RequestContext) -> bool:
        return True

    def read(self, session: FormFillSession | None, request: RequestContext) -> FormInput | dict[str, Any] | None:
        return self.PAUSE

    def pause(self, reply: Reply, session: FormFillSession | None, request: RequestContext) -> dict[str, Any] | None:
        return None

    def words(self, reply: Reply) -> str:
        return reply.text


class TestA2AExecutorPicksTheTransport:
    @pytest.mark.asyncio
    async def test_a_transport_may_answer_a_message_itself(self):
        agent = _agent()
        transports: list[A2AHitl] = [_InterimA2A()]
        message = Message(role=Role.ROLE_USER, parts=[Part(text="x")], message_id=str(uuid.uuid4()))
        request = RequestContext(
            call_context=ServerCallContext(),
            request=SendMessageRequest(message=message),
            task_id="task-1",
            context_id="ctx-1",
        )
        with (
            patch("orchestrator_agent.adapters.a2a.adapter.PostgresStatePersistence") as persistence,
            patch("orchestrator.core.db.db"),
        ):
            persistence.return_value.load_state = AsyncMock(return_value=None)
            queue = EventQueueLegacy()
            await WFOAgentExecutor(agent, transports=transports).execute(request, queue)
        events = []
        while True:
            try:
                events.append(queue.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        last = [event for event in events if isinstance(event, TaskStatusUpdateEvent)][-1]
        # The task pauses with the transport's payload, declared under its extension; the skill did not run.
        assert last.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        assert json_format.MessageToDict(last.status.message.metadata) == {"urn:test:hitl": {"more": True}}
        assert list(last.status.message.extensions) == ["urn:test:hitl"]
        assert not agent.run_stream_events.called
