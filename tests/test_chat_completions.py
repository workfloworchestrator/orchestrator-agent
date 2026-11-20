"""The chat-completions adapter: the agent as the model of an OpenAI-compatible chat client.

The agent's run is mocked (what the form-fill capability would do to the state), as in the A2A executor
tests: no LLM, no DB — the adapter's own turn logic and wire shapes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from orchestrator_agent.adapters.chat.completions import (
    CLIENT_HEADER,
    CONVERSATION_HEADER,
    ChatCompletionsAdapter,
)
from orchestrator_agent.state import SearchState

from .conftest import make_text_result_event, mock_event_stream, parse_sse_events

ASK_TOOL = {"type": "function", "function": {"name": "ask_user_question", "parameters": {"type": "object"}}}


def _agent_doing(setup, text: str = "model text") -> MagicMock:
    """An agent mock whose run applies ``setup(state)`` (what the form-fill capability would do) and answers ``text``."""
    agent = MagicMock()

    @asynccontextmanager
    async def _run(*args, **kwargs):
        agent._last_call = (args, kwargs)
        setup(kwargs["deps"].state)
        yield mock_event_stream(make_text_result_event(text))

    agent.run_stream_events = MagicMock(side_effect=_run)
    return agent


class _Turn:
    """One request to the adapter with the persistence stubbed: the response, and what was run and snapshotted."""

    def __init__(self, agent: MagicMock, prior: SearchState | None = None) -> None:
        self.agent = agent
        self.prior = prior
        self.thread: str | None = None
        self.snapshot: SearchState | None = None

    async def post(
        self, body: dict[str, Any], *, conversation: str | None = "conv-1", client: str | None = "librechat"
    ) -> httpx.Response:
        app = FastAPI()
        ChatCompletionsAdapter(self.agent).add_routes(app)
        headers = {CONVERSATION_HEADER: conversation, "authorization": "Bearer user-token"} if conversation else {}
        if client:
            headers[CLIENT_HEADER] = client
        with (
            patch("orchestrator_agent.adapters.chat.completions.PostgresStatePersistence") as persistence,
            patch("orchestrator.core.db.db"),
        ):
            persistence.return_value.load_state = AsyncMock(return_value=self.prior)
            persistence.return_value.snapshot = AsyncMock()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                response = await http.post("/v1/chat/completions", json=body, headers=headers)
            self.thread = persistence.call_args.kwargs["thread_id"] if persistence.call_args else None
            if persistence.return_value.snapshot.await_args:
                (self.snapshot,), _ = persistence.return_value.snapshot.await_args
        return response

    @property
    def state(self) -> SearchState:
        """The state the agent ran with."""
        state: SearchState = self.agent._last_call[1]["deps"].state
        return state

    @property
    def ran(self) -> bool:
        called: bool = self.agent.run_stream_events.called
        return called


def _body(*messages: dict[str, Any], tools: bool = True, stream: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "wfo", "user": "u-1", "stream": stream, "messages": list(messages)}
    if tools:
        body["tools"] = [ASK_TOOL]
    return body


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


class TestChatTurn:
    @pytest.mark.asyncio
    async def test_the_models_answer_is_the_assistant_message(self):
        turn = _Turn(_agent_doing(lambda state: None, text="Three subscriptions."))
        response = await turn.post(_body(_user("how many subscriptions?")))
        assert response.status_code == 200
        (choice,) = response.json()["choices"]
        assert choice["message"] == {"role": "assistant", "content": "Three subscriptions."}
        assert choice["finish_reason"] == "stop" and response.json()["object"] == "chat.completion"
        assert turn.agent._last_call[0] == ("how many subscriptions?",)
        assert turn.thread == "conv-1"  # memory and the open form are keyed on LibreChat's conversation

    @pytest.mark.asyncio
    async def test_a_streamed_answer_is_chunks_ending_in_done(self):
        turn = _Turn(_agent_doing(lambda state: None, text="Three."))
        response = await turn.post(_body(_user("how many?"), stream=True))
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.text.rstrip().endswith("data: [DONE]")
        chunks = parse_sse_events([response.content.replace(b"data: [DONE]", b"")])
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
        assert deltas == [{"role": "assistant", "content": ""}, {"content": "Three."}, {}]
        assert [chunk["choices"][0]["finish_reason"] for chunk in chunks] == [None, None, "stop"]

    @pytest.mark.asyncio
    async def test_only_the_last_user_message_is_the_input(self):
        turn = _Turn(_agent_doing(lambda state: None))
        await turn.post(
            _body(
                _user("first"),
                {"role": "assistant", "content": "answer"},
                _user("# Answers the user gave to questions asked earlier in this conversation\nQ: ... A: ..."),
                {"role": "user", "content": [{"type": "text", "text": "the real question"}]},
            )
        )
        assert turn.agent._last_call[0] == ("the real question",)

    @pytest.mark.asyncio
    async def test_this_route_cannot_show_an_approval(self):
        turn = _Turn(_agent_doing(lambda state: None))
        await turn.post(_body(_user("create a port")))
        assert turn.state.hitl is False  # a form can be prepared here, not started

    @pytest.mark.asyncio
    async def test_the_bearer_token_is_forwarded_to_core(self):
        turn = _Turn(_agent_doing(lambda state: None))
        with patch("orchestrator_agent.turn.bind_outbound_token") as bind:
            await turn.post(_body(_user("hi")))
        bind.assert_called_once_with("user-token")

    @pytest.mark.asyncio
    async def test_a_request_without_a_conversation_id_is_a_conversation_of_its_own(self):
        turn = _Turn(_agent_doing(lambda state: None))
        response = await turn.post(_body(_user("hi")), conversation=None)
        assert response.status_code == 200 and turn.thread and turn.thread != "conv-1"

    @pytest.mark.asyncio
    async def test_a_failing_run_is_a_server_error(self):
        agent = _agent_doing(lambda state: None)
        agent.run_stream_events = MagicMock(side_effect=RuntimeError("model down"))
        response = await _Turn(agent).post(_body(_user("hi")))
        assert response.status_code == 500 and response.json()["error"]["type"] == "server_error"
