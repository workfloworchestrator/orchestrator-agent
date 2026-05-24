# Copyright 2019-2026 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

"""OpenAI-compatible chat-completions adapter.

LibreChat (and any other OpenAI client) sees this agent as a regular
model endpoint. One streaming endpoint at ``POST /v1/chat/completions``
plus a non-streaming fallback for ``stream: false`` requests.

Mirrors :class:`A2AAdapter`'s shape: a class with ``add_routes(app)`` and
``__aenter__/__aexit__`` for agent lifecycle. The actual translation is
event-by-event — pydantic-ai's ``PartDeltaEvent`` becomes OpenAI's
``delta.content`` chunks; ``FunctionToolResultEvent`` carrying a
``ToolArtifact`` becomes a fenced markdown block streamed as content so
LibreChat renders it inline. A future iteration can promote artifacts
to native ``tool_calls`` once the LibreChat plugin is ready to dispatch
on them.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, AsyncIterator

import structlog
from a2a.types import Artifact as A2AArtifact
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResultEvent

from orchestrator_agent.agent import AgentAdapter
from orchestrator_agent.artifacts import ToolArtifact
from orchestrator_agent.events import AgentStepActiveEvent, PlanCreatedEvent
from orchestrator_agent.persistence import PostgresStatePersistence
from orchestrator_agent.state import SearchState

logger = structlog.get_logger(__name__)


_CONVERSATION_HEADER = "x-conversation-id"  # LibreChat templates {{LIBRECHAT_BODY_CONVERSATIONID}} into this
_TITLE_MODEL = "orchestrator-agent-title"  # paired with `titleModel` in librechat.yaml


def _thread_id_from_request(headers: dict[str, str], messages: list[dict[str, Any]]) -> str:
    """Derive a stable thread_id for state persistence.

    Prefers an explicit ``X-Conversation-Id`` header (LibreChat can template
    its conversation id into this); falls back to a hash of the first user
    message, which is stable for the lifetime of a conversation even
    without the header.
    """
    if cid := headers.get(_CONVERSATION_HEADER):
        return cid
    first = next((m for m in messages if m.get("role") == "user"), None)
    if first is None:
        return f"openai-{uuid.uuid4()}"
    seed = json.dumps(first.get("content", ""), sort_keys=True, ensure_ascii=False)
    return "openai-" + hashlib.sha256(seed.encode()).hexdigest()[:16]


def _message_text(message: dict[str, Any]) -> str:
    """Flatten an OpenAI message content field (str or parts-list) to a string."""
    content = message.get("content") or ""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def _format_artifact_block(name: str, data: dict[str, Any]) -> str:
    """Render an artifact as a fenced markdown block with a tagged language.

    Wire format::

        ```wfo-artifact:<Name>
        { ...json... }
        ```

    The custom language hint is the dispatch key for downstream renderers.
    Markdown viewers that don't recognise it (vanilla LibreChat, GitHub,
    pasted in a doc) fall back to rendering it as a generic code block —
    still legible, still copy-pasteable. Renderers that know the prefix
    (our LibreChat fork) intercept and mount the matching React component
    keyed on ``<Name>``.
    """
    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    return f"\n\n```wfo-artifact:{name}\n{pretty}\n```\n\n"


def _chunk(completion_id: str, model: str, *, delta: dict[str, Any], finish: str | None = None) -> str:
    """Wrap one delta as an OpenAI streaming chat-completion SSE event."""
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class OpenAIAdapter:
    """Wires an OpenAI-compatible chat endpoint onto a FastAPI app.

    Usage mirrors :class:`A2AAdapter`::

        adapter = OpenAIAdapter(agent)
        adapter.add_routes(app)
        async with adapter:
            yield
    """

    def __init__(self, agent: AgentAdapter) -> None:
        self.agent = agent

    def add_routes(self, app: FastAPI) -> None:
        @app.post("/v1/chat/completions")
        async def chat_completions(req: Request) -> Any:
            body = await req.json()
            messages = body.get("messages") or []
            model = body.get("model") or "orchestrator-agent"
            stream = bool(body.get("stream", False))

            last_user = next(
                (m for m in reversed(messages) if m.get("role") == "user"),
                None,
            )
            user_input = _message_text(last_user) if last_user else ""

            headers = {k.lower(): v for k, v in req.headers.items()}
            thread_id = _thread_id_from_request(headers, messages)
            completion_id = f"chatcmpl-{uuid.uuid4()}"

            # Title-generation short-circuit. LibreChat's titleModel setting
            # lets us route title requests to a distinct model name; we own
            # both sides of that name. When this model is requested, return
            # the user's first message verbatim — no LLM call, no extra cost.
            # The OpenAI `model` field is the canonical discriminator for
            # alternate behaviours on the same endpoint.
            if model == _TITLE_MODEL:
                first_user = next((m for m in messages if m.get("role") == "user"), None)
                title = _message_text(first_user).strip()[:60] if first_user else "Chat"
                return JSONResponse(
                    {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": title},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    }
                )

            if stream:
                return StreamingResponse(
                    self._stream(completion_id, model, thread_id, user_input),
                    media_type="text/event-stream",
                )
            return JSONResponse(await self._collect(completion_id, model, thread_id, user_input))

    async def _run_events(self, thread_id: str, user_input: str) -> AsyncIterator[Any]:
        """Set up persistence + state and drive the agent's event stream."""
        from orchestrator.db import db
        from orchestrator.db.models import AgentRunTable

        run_id = uuid.uuid4()
        agent_run = AgentRunTable(run_id=run_id, thread_id=thread_id, agent_type="openai")
        db.session.add(agent_run)
        db.session.commit()
        try:
            persistence = PostgresStatePersistence(thread_id=thread_id, run_id=run_id, session=db.session)
            self.agent._persistence = persistence

            deps = StateDeps(SearchState(user_input=user_input, run_id=run_id, adapter_metadata=None))
            logger.debug("openai: run start", thread_id=thread_id, user_input=user_input[:100])
            async for event in self.agent.run_stream_events(deps=deps):
                yield event
        except Exception:
            db.session.rollback()
            raise

    @staticmethod
    def _step_active_to_reasoning(event: AgentStepActiveEvent) -> tuple[str, str] | None:
        """``AgentStepActiveEvent`` → reasoning-channel text, or ``None`` to suppress."""
        step = event.value.get("step", "")
        reason = event.value.get("reasoning")
        # "Planner" and "Synthesizer" are phase markers without reasoning;
        # PlanCreatedEvent already enumerates the work that follows the
        # Planner, and the final assistant text already covers what the
        # Synthesizer produced. Suppress these to keep the panel terse.
        if not reason and step in {"Planner", "Synthesizer"}:
            return None
        # LibreChat's Thinking panel renders reasoning_content as plain
        # text, not markdown — keep separators minimal and ASCII.
        text = step + (f" — {reason}" if reason else "") + "\n\n"
        return ("reasoning_content", text)

    @staticmethod
    def _tool_result_to_content(event: FunctionToolResultEvent) -> tuple[str, str] | None:
        """``FunctionToolResultEvent`` → fenced artifact block, or ``None`` if not an artifact."""
        result = event.result
        if not isinstance(result, ToolReturnPart):
            return None
        md = result.metadata
        if isinstance(md, ToolArtifact):
            return (
                "content",
                _format_artifact_block(type(md).__name__, md.model_dump(mode="json", by_alias=True)),
            )
        if isinstance(md, A2AArtifact):
            payload = {
                "name": md.name,
                "metadata": md.metadata,
                "parts": [p.model_dump(mode="json") for p in md.parts],
            }
            return ("content", _format_artifact_block(md.name or "Artifact", payload))
        return None

    @staticmethod
    def _event_to_delta(event: Any) -> tuple[str, str] | None:
        """Translate one agent event to ``(delta_key, text)``.

        ``delta_key`` is either ``"content"`` (visible assistant output, e.g.
        text tokens and our ``wfo-artifact:`` fenced blocks) or
        ``"reasoning_content"`` (LibreChat's collapsible Thinking panel).

        Progress signals reuse the same ``AgentStepActiveEvent`` /
        ``PlanCreatedEvent`` CustomEvents that AG-UI already passes through —
        same event stream, same source of truth.

        We deliberately don't route tool calls to native ``tool_calls`` deltas:
        LibreChat runs a tool-execution harness on response-side ``tool_calls``
        and loops indefinitely when no handler is registered for the name
        (verified — custom endpoints don't pass ``tools`` in the request, so
        no handler ever exists).
        """
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content:
            return ("content", event.part.content)
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
            return ("content", event.delta.content_delta)
        if isinstance(event, AgentStepActiveEvent):
            return OpenAIAdapter._step_active_to_reasoning(event)
        if isinstance(event, PlanCreatedEvent):
            steps = ", ".join(t.get("skill_name", "") for t in event.value)
            return ("reasoning_content", f"Plan: {steps}\n\n")
        if isinstance(event, FunctionToolResultEvent):
            return OpenAIAdapter._tool_result_to_content(event)
        return None

    async def _stream(self, completion_id: str, model: str, thread_id: str, user_input: str) -> AsyncIterator[str]:
        yield _chunk(completion_id, model, delta={"role": "assistant"})
        emitted = False
        try:
            async for event in self._run_events(thread_id, user_input):
                if delta := self._event_to_delta(event):
                    channel, text = delta
                    # Only `content` counts as "user-visible output"; reasoning
                    # is progress noise that doesn't replace a missing answer.
                    if channel == "content":
                        emitted = True
                    yield _chunk(completion_id, model, delta={channel: text})
                if isinstance(event, AgentRunResultEvent):
                    # If nothing was streamed (e.g. skill failed before any LLM
                    # text), the final summary lives only on the result event.
                    if not emitted and (output := str(event.result.output or "").strip()):
                        yield _chunk(completion_id, model, delta={"content": output})
                    break
        except Exception as e:
            logger.exception("openai: stream failed")
            yield _chunk(completion_id, model, delta={"content": f"\n\n:warning: {e}"})
        yield _chunk(completion_id, model, delta={}, finish="stop")
        yield "data: [DONE]\n\n"

    async def _collect(self, completion_id: str, model: str, thread_id: str, user_input: str) -> dict[str, Any]:
        parts: list[str] = []
        try:
            async for event in self._run_events(thread_id, user_input):
                if delta := self._event_to_delta(event):
                    channel, text = delta
                    # Non-streaming responses have no reasoning channel; drop
                    # progress signals here, keep only visible content.
                    if channel == "content":
                        parts.append(text)
                if isinstance(event, AgentRunResultEvent):
                    if not parts and (output := str(event.result.output or "").strip()):
                        parts.append(output)
                    break
        except Exception as e:
            logger.exception("openai: collect failed")
            parts.append(f"\n\n:warning: {e}")
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(parts)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def __aenter__(self) -> "OpenAIAdapter":
        await self.agent.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.agent.__aexit__(*exc)
