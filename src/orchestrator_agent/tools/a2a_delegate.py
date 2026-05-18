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

Artifact forwarding is intentionally not wired yet — current delegates
return plain text. When a domain agent eventually emits structured artifacts
that need to surface to Slack / AG-UI, change the ``return ToolReturn(...)``
line in ``_delegate`` to forward ``artifacts[0]`` as ``metadata``. The client
already parses them.
"""

from __future__ import annotations

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
    # Prefer artifacts collected from the event stream; fall back to whatever's on the Task
    artifacts: list[Artifact] = streamed_artifacts or (
        list(result.artifacts) if isinstance(result, Task) and result.artifacts else []
    )
    return final_text, artifacts


def _part_text(part: Part) -> str | None:
    """Return the text payload of a Part if it's a TextPart, else None."""
    root = part.root
    return getattr(root, "text", None)


def _message_text(msg: Message | None) -> str:
    """Return the first non-empty text part of a Message, or ``""``."""
    if msg is None:
        return ""
    return next((t for p in msg.parts if (t := _part_text(p))), "")


def _extract_final_text(result: Task | Message) -> str:
    """Pull the user-facing answer out of an A2A response.

    Handles both the standard A2A shape (text on ``Task.status.message``) and
    kagent's ADK shape (no status message; the agent's reply lives in
    ``Task.history`` with role=agent).
    """
    if isinstance(result, Message):
        return _message_text(result)
    if text := _message_text(result.status.message):
        return text
    # kagent ADK fallback: walk history backwards, find last agent message with text.
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

        if artifacts:
            # Forwarding not implemented yet — log so we notice when a delegate
            # starts emitting structured output that we should pass through.
            logger.warning(
                "Domain agent returned artifacts; artifact forwarding is not yet wired",
                short_name=short_name,
                count=len(artifacts),
            )

        state.memory.record_tool_step(
            ToolStep(
                step_type=f"ask_{short_name}",
                description=f"Delegated to {short_name}: {query[:80]}",
                context={"short_name": short_name, "artifact_count": len(artifacts)},
            )
        )

        return ToolReturn(return_value=final_text or "(no response)", metadata=None)

    return _delegate
