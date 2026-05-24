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

"""A2A delegation — call out to domain agents from inside a skill.

Each peripheral domain (IMS, CIM, Jira, telemetry, alarming) is owned by its
own kagent agent. This module provides:

- ``call_a2a()`` — wraps the ``a2a-sdk`` client. Sends one message, returns
  the final text and any artifacts the remote agent emitted.
- ``make_a2a_delegate_handler()`` — factory that builds a direct-dispatch
  async handler for a delegation skill. The planner picked the skill, the
  planner produced the query — the handler just executes the A2A call.
  No second LLM, no toolset, no prompt.

The handler returns a :class:`ToolReturn` whose ``metadata`` is the first
artifact (for the surface adapter to render as Block Kit / AG-UI) and whose
``return_value`` is the textual answer the planner will see in its next-turn
memory. When the remote answered with a structured DataPart only (e.g.
kagent's ``ask_user`` clarification), we JSON-serialize it into
``return_value`` so the planner knows what was asked and can correlate the
user's next reply against it.
"""

from __future__ import annotations

import json
import uuid
from typing import Awaitable, Callable

import httpx
import structlog
from a2a.client import ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TextPart,
)
from pydantic_ai.messages import ToolReturn

from orchestrator_agent.memory import ToolStep
from orchestrator_agent.settings import agent_settings
from orchestrator_agent.state import SearchState

logger = structlog.get_logger(__name__)


async def call_a2a(
    base_url: str,
    *,
    thread_id: str,
    text: str,
    auth_token: str | None = None,
    timeout: float = 120.0,
) -> tuple[str, list[Artifact]]:
    """Call a remote A2A agent. Returns ``(final_text, artifacts)``.

    The agent card is fetched from ``base_url/.well-known/agent-card.json`` (one extra
    request per call; caching could be added). The send is
    a streaming iterator; we drain it and reduce to the latest ``Task`` /
    ``Message`` for final-text and artifact extraction.

    ``artifacts`` is a list of typed :class:`a2a.types.Artifact` objects —
    whatever the remote agent emitted via its A2A artifact channel. The
    current delegation flow doesn't surface them; they're collected so the
    extension point is in place.
    """
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
    base = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base)
        card = await resolver.get_agent_card()

        # Override the card's advertised URL with the base_url we actually reached
        # the agent at. Otherwise the SDK uses the in-cluster URL the card declares
        # (e.g. http://ims-mcp-agent.kagent:8080) — unresolvable from outside the cluster.
        card = card.model_copy(update={"url": base})

        factory = ClientFactory(ClientConfig(streaming=False, httpx_client=httpx_client))
        client = factory.create(card)

        message = Message(
            role=Role.user,
            message_id=str(uuid.uuid4()),
            context_id=thread_id,
            parts=[Part(root=TextPart(text=text))],
            metadata={},  # kagent ADK distinguishes "new session" from "resume" on metadata presence
        )

        latest_task: Task | None = None
        latest_message: Message | None = None
        streamed_artifacts: list[Artifact] = []

        async for event in client.send_message(message):
            if isinstance(event, Message):
                latest_message = event
                continue
            # Otherwise: tuple[Task, TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None]
            task, update = event
            latest_task = task
            if isinstance(update, TaskArtifactUpdateEvent):
                streamed_artifacts.append(update.artifact)
            # TaskStatusUpdateEvent: task itself has the latest state — handled by latest_task

    result: Task | Message | None = latest_task or latest_message
    if result is None:
        return "", []
    final_text = _extract_final_text(result)
    artifacts: list[Artifact] = list(streamed_artifacts)
    if isinstance(result, Task):
        if not artifacts and result.artifacts:
            artifacts.extend(result.artifacts)
        # Non-terminal status (e.g. input-required) carries the agent's response
        # in ``status.message`` rather than ``artifacts``. Forward it as a
        # synthesized Artifact so surface adapters can render it the same way.
        if result.status.state != TaskState.completed and result.status.message is not None:
            artifacts.append(
                Artifact(
                    artifact_id=str(uuid.uuid4()),
                    name=f"a2a_status_{result.status.state.value}",
                    parts=list(result.status.message.parts),
                    metadata={"a2a_state": result.status.state.value},
                )
            )
    return final_text, artifacts


def _part_text(part: Part) -> str | None:
    """Return the text payload of a Part if it's a TextPart, else None."""
    return getattr(part.root, "text", None)


def _message_text(msg: Message | None) -> str:
    """Return the first non-empty text part of a Message, or ``""``."""
    if msg is None:
        return ""
    return next((t for p in msg.parts if (t := _part_text(p))), "")


def _extract_final_text(result: Task | Message) -> str:
    """Pull the user-facing answer out of an A2A response.

    Returns text only. Non-text content (DataParts, e.g. kagent's ``ask_user``)
    is surfaced as an Artifact by :func:`call_a2a` instead, so the surface
    adapter can render it.
    """
    if isinstance(result, Message):
        return _message_text(result)
    if text := _message_text(result.status.message):
        return text
    for msg in reversed(result.history or []):
        if msg.role == Role.agent and (text := _message_text(msg)):
            return text
    return ""


DelegateHandler = Callable[["SearchState", str], Awaitable["ToolReturn"]]


def make_a2a_delegate_handler(
    *,
    short_name: str,
    url_attr: str,
) -> DelegateHandler:
    """Build a direct-dispatch handler for a delegation skill.

    The returned callable takes ``(state, query)`` and performs the A2A
    round-trip to the configured domain agent. ``query`` is whatever the
    planner placed in ``Task.reasoning`` for this task — a focused English
    question phrased for the domain agent.

    No prompt, no toolset, no LLM in the loop. The planner already picked
    this skill and constructed the query; we just execute it.
    """

    async def _delegate(state: SearchState, query: str) -> ToolReturn:
        url = getattr(agent_settings, url_attr, "") or ""
        if not url:
            raise RuntimeError(f"{url_attr} is not configured — the {short_name} domain agent is unreachable.")

        thread_id = str(state.run_id) if state.run_id else f"delegate-{uuid.uuid4()}"
        final_text, artifacts = await call_a2a(url, thread_id=thread_id, text=query)

        # Forward the first artifact (if any) via ToolReturn.metadata so the
        # surface adapter can render it. Multi-artifact responses aren't
        # common from the agents we delegate to; if they become so, ToolReturn
        # would need a list shape or we'd emit multiple events.
        first_artifact = artifacts[0] if artifacts else None
        if len(artifacts) > 1:
            logger.warning(
                "Domain agent returned multiple artifacts; only the first is forwarded",
                short_name=short_name,
                count=len(artifacts),
            )

        # When the remote returned a DataPart-only response (e.g. kagent's
        # ``ask_user`` clarification), ``_extract_final_text`` yields "". The
        # planner needs to see *what* was asked, otherwise it can't correlate
        # the user's next reply against an unanswered question and loops on
        # the original query. Surface a JSON summary of the first artifact as
        # the textual response so the planner records it in memory.
        response_text = final_text or _summarize_artifact(first_artifact)

        state.memory.record_tool_step(
            ToolStep(
                step_type=f"ask_{short_name}",
                description=f"Asked {short_name}: {query[:80]} → {response_text[:300]}",
                context={"short_name": short_name, "artifact_count": len(artifacts)},
            )
        )

        return ToolReturn(
            return_value=response_text or "(no response)",
            metadata=first_artifact,
        )

    return _delegate


def _summarize_artifact(artifact: Artifact | None) -> str:
    """Best-effort text summary of an Artifact for the planner's memory.

    Prefers TextParts; falls back to JSON-dumping the first DataPart so any
    structured payload (typically kagent's ``ask_user`` questions/choices)
    becomes visible to the planner without losing its shape. Surface
    rendering is unaffected — the full artifact is still forwarded via
    ``ToolReturn.metadata``.
    """
    if artifact is None:
        return ""
    for part in artifact.parts:
        text = getattr(part.root, "text", None)
        if text:
            return str(text)
        data = getattr(part.root, "data", None)
        if data is not None:
            try:
                return json.dumps(data, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(data)
    return ""
