"""The chat-completions adapter: the agent as LibreChat's model, forms through its ask-user tool.

The agent's run is mocked (what the form-fill capability would do to the state), as in the A2A executor
tests: no LLM, no DB — the adapter's own turn logic and wire shapes.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from orchestrator_agent.adapters.chat.completions import (
    CLIENT_HEADER,
    CONVERSATION_HEADER,
    FORM_RESPONSE,
    NOT_OPEN,
    ChatCompletionsAdapter,
)
from orchestrator_agent.adapters.chat.librechat import NOT_SHOWN, ask_card, call_id, pause
from orchestrator_agent.form_fill.pending import PendingAsk, pending_of
from orchestrator_agent.state import Approval, AskField, FormFillSession, FormInput, Reply, SearchState

from .conftest import make_text_result_event, mock_event_stream, parse_sse_events

ASK_TOOL = {"type": "function", "function": {"name": "ask_user_question", "parameters": {"type": "object"}}}
ASK = Reply(
    '{"workflow_key":"w","status":"gathering","page":0,"title":"Page"}',
    ask=[AskField(name="redundancy", question="Redundancy?", choices=["Protected"], values=["protected"])],
)
FIVE = Reply(
    '{"workflow_key":"w","status":"gathering"}', ask=[AskField(name=f"f{n}", question=f"F{n}?") for n in range(5)]
)
APPROVAL = Reply(
    '{"workflow_key":"w","status":"confirming"}',
    approval=Approval(hint="Start?", tool_name="create_workflow", args={"workflow_key": "w", "json_data": [{}]}),
)


def _agent_doing(setup, text: str = "model text") -> MagicMock:
    """An agent mock whose run applies ``setup(state)`` (what the form-fill capability would do) and answers ``text``."""
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)

    @asynccontextmanager
    async def _run(*args, **kwargs):
        agent._last_call = (args, kwargs)
        setup(kwargs["deps"].state)
        yield mock_event_stream(make_text_result_event(text))

    agent.run_stream_events = MagicMock(side_effect=_run)
    return agent


def _stopping(reply: Reply) -> MagicMock:
    def stopped(state: SearchState) -> None:
        state.form_fill = state.form_fill or FormFillSession(workflow_key="w", status="gathering")
        state.form_reply = reply

    return _agent_doing(stopped, text=reply.text)


def _open_form(reply: Reply) -> tuple[SearchState, PendingAsk]:
    """The persisted state of a conversation whose form is paused at ``reply``'s stop."""
    session = FormFillSession(workflow_key="w", status="gathering")
    pending = pause(reply, session)
    assert pending is not None
    return SearchState(form_fill=session), pending


def _tool_message(pending: PendingAsk, card: int, answers: dict[str, str] | str) -> dict[str, Any]:
    content = answers if isinstance(answers, str) else json.dumps({"answers": answers})
    return {"role": "tool", "tool_call_id": call_id(pending, card), "name": "ask_user_question", "content": content}


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
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", json=body, headers=headers)
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
    async def test_the_client_header_is_what_makes_the_caller_one_that_can_show_a_stop(self):
        librechat = _Turn(_agent_doing(lambda state: None))
        await librechat.post(_body(_user("create a port")))
        assert librechat.state.hitl is True
        # Another client offering a tool of the same name is not LibreChat: no form for it.
        unnamed = _Turn(_agent_doing(lambda state: None))
        await unnamed.post(_body(_user("create a port")), client=None)
        assert unnamed.state.hitl is False
        other = _Turn(_agent_doing(lambda state: None))
        await other.post(_body(_user("create a port")), client="some-other-client")
        assert other.state.hitl is False

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


class TestFormStops:
    @pytest.mark.asyncio
    async def test_a_stop_is_an_ask_user_call(self):
        turn = _Turn(_stopping(ASK))
        response = await turn.post(_body(_user("create a lightpath")))
        (choice,) = response.json()["choices"]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] == "**Workflow form `w`** — Page"  # not the skill's JSON
        (call,) = choice["message"]["tool_calls"]
        assert call["type"] == "function" and call["function"]["name"] == "ask_user_question"
        (question,) = json.loads(call["function"]["arguments"])["questions"]
        assert question["id"] == "q0" and [option["label"] for option in question["options"]] == ["Protected"]
        # The session remembers the stop, keyed by the call, before the state is persisted.
        assert turn.snapshot is not None and turn.snapshot.form_fill is not None
        pending = pending_of(turn.snapshot.form_fill)
        assert pending is not None and call["id"] == call_id(pending, 0)
        assert [(q.name, list(q.values)) for q in pending.questions] == [("redundancy", ["protected"])]
        assert turn.snapshot.form_fill.unseen is False

    @pytest.mark.asyncio
    async def test_a_streamed_stop_ends_in_the_tool_call(self):
        response = await _Turn(_stopping(APPROVAL)).post(_body(_user("go on"), stream=True))
        chunks = parse_sse_events([response.content.replace(b"data: [DONE]", b"")])
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
        assert deltas[1] == {"content": "**Start workflow `w` with these values?**"}
        (call,) = deltas[2]["tool_calls"]
        assert call["index"] == 0 and call["function"]["name"] == "ask_user_question"
        assert [o["label"] for o in json.loads(call["function"]["arguments"])["questions"][0]["options"]] == [
            "Approve",
            "Reject",
        ]
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_a_reply_that_ends_the_form_is_text_for_the_person(self):
        def started(state: SearchState) -> None:
            state.form_fill = None
            state.form_reply = Reply('{"workflow_key":"w","status":"started","process_id":"p-1"}')

        response = await _Turn(_agent_doing(started)).post(_body(_user("x")))
        (choice,) = response.json()["choices"]
        assert choice["message"] == {"role": "assistant", "content": "Workflow `w` started. Process id: `p-1`"}
        assert choice["finish_reason"] == "stop"


class TestFormResponses:
    @pytest.mark.asyncio
    async def test_the_tool_message_is_the_skills_input(self):
        prior, pending = _open_form(ASK)
        assert prior.form_fill is not None
        token = ask_card(pending, prior.form_fill, 0)["questions"][0]["options"][0]["value"]
        turn = _Turn(_stopping(APPROVAL), prior)
        response = await turn.post(
            _body(
                _user("create a lightpath"),
                {"role": "assistant", "content": "", "tool_calls": []},
                _tool_message(pending, 0, {"q0": token}),
            )
        )
        # The picked option arrives as the form value behind it; no text is the turn's input.
        assert turn.state.form_input == FormInput(values={"redundancy": "protected"})
        assert turn.agent._last_call[0] == (FORM_RESPONSE,)
        assert turn.state.form_fill is not None and turn.state.hitl is True
        # The next stop goes out in the same run.
        assert response.json()["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_a_page_of_more_than_four_fields_is_asked_card_by_card(self):
        prior, pending = _open_form(FIVE)
        first = _tool_message(pending, 0, {f"q{n}": f"v{n}" for n in range(4)})
        turn = _Turn(_stopping(APPROVAL), prior)
        response = await turn.post(_body(_user("x"), first))
        # The second card is shown without a run: the skill gets the response once it is complete.
        (call,) = response.json()["choices"][0]["message"]["tool_calls"]
        assert call["id"] == call_id(pending, 1)
        assert [q["id"] for q in json.loads(call["function"]["arguments"])["questions"]] == ["q4"]
        assert not turn.ran and turn.snapshot is None

        done = _Turn(_stopping(APPROVAL), prior)
        await done.post(_body(_user("x"), first, _tool_message(pending, 1, {"q4": "v4"})))
        assert done.state.form_input == FormInput(values={f"f{n}": f"v{n}" for n in range(5)})

    @pytest.mark.asyncio
    async def test_a_message_instead_of_an_answer_is_no_response(self):
        # LibreChat replaces the paused run: the unanswered call comes back with an empty result, then the
        # new message. The skill then ends the form and the model answers.
        prior, pending = _open_form(ASK)
        turn = _Turn(_agent_doing(lambda state: None), prior)
        await turn.post(_body(_user("create"), _tool_message(pending, 0, ""), _user("never mind, how many ports?")))
        assert turn.state.form_input is None and turn.state.form_fill is not None
        assert turn.agent._last_call[0] == ("never mind, how many ports?",)

    @pytest.mark.asyncio
    async def test_an_answer_without_an_open_form_runs_nothing(self):
        _, pending = _open_form(ASK)
        turn = _Turn(_agent_doing(lambda state: None), prior=None)
        response = await turn.post(_body(_user("x"), _tool_message(pending, 0, {"q0": "v"})))
        assert response.json()["choices"][0]["message"]["content"] == NOT_OPEN
        assert not turn.ran
        # Nor when the request no longer offers the tool.
        prior, pending = _open_form(ASK)
        plain = _Turn(_agent_doing(lambda state: None), prior)
        response = await plain.post(_body(_user("x"), _tool_message(pending, 0, {"q0": "v"}), tools=False))
        assert response.json()["choices"][0]["message"]["content"] == NOT_OPEN and not plain.ran

    @pytest.mark.asyncio
    async def test_a_call_librechat_refused_is_not_sent_again(self):
        prior, pending = _open_form(ASK)
        turn = _Turn(_stopping(ASK), prior)
        error = "Error: Received tool input did not match expected schema"
        response = await turn.post(_body(_user("x"), _tool_message(pending, 0, error)))
        (choice,) = response.json()["choices"]
        assert choice["finish_reason"] == "stop" and choice["message"]["content"] == f"{NOT_SHOWN}: {error}"
        assert not turn.ran
