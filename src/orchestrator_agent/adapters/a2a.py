# Copyright 2019-2025 SURF, GÉANT.
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

"""A2A adapter — exposes the orchestrator agent via the A2A protocol (v1.0).

Uses the ``a2a-sdk`` 1.x server primitives (``AgentExecutor``, ``DefaultRequestHandler`` and the route
factories). The executor drives the plain capabilities-based agent inside ``async with agent:`` (MCP
session) and collects artifact results + final text.

``A2A_SKILLS`` is static A2A protocol metadata (the agent's advertised skills) and is
unrelated to pydantic-ai capabilities — it just describes what the agent can do.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import structlog
from a2a.helpers import new_data_part, new_task, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentExtension, AgentInterface, Message, TaskState
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
from fastapi import FastAPI
from google.protobuf import json_format
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResultEvent

from orchestrator_agent.adapters.a2a_hitl import FormTurn
from orchestrator_agent.adapters.kagent_hitl import HITL_EXTENSION_DESCRIPTION, HITL_EXTENSION_URI, payload_metadata
from orchestrator_agent.agent import new_deps
from orchestrator_agent.artifacts import QueryArtifact, ToolArtifact
from orchestrator_agent.capabilities.spec import skills_from_specs
from orchestrator_agent.mcp_client import bind_outbound_token
from orchestrator_agent.persistence import PostgresStatePersistence, dump_messages, load_messages

if TYPE_CHECKING:
    from orchestrator_agent.agent import WFOAgent

logger = structlog.get_logger(__name__)

NO_RESULTS = "No results"

# Card-level tool description a consuming agent reads. The output-handling note covers only the
# artifact-bearing replies (a result with a pre-rendered table/chart), not every tool call.
AGENT_CARD_DESCRIPTION = (
    "Answers questions about orchestration data and starts workflows by walking you through their "
    "input forms: ask for one (what, and for which subscription or product) and it walks the form with "
    "you, asking for the user's confirmation before starting anything. Every form reply is one JSON "
    "object with a status: gathering (the page's schema, what the orchestrator rejected in its own words, "
    "the values so far), confirming (the values to be submitted and the defaults that apply), started "
    "(the process id), cancelled, failed (why). Answer a form reply with one JSON object keyed by field name (values, not labels) or in "
    "words; to start or cancel, say so outright. When a reply includes a pre-rendered Markdown table or "
    "Mermaid chart, relay that block to the user verbatim, without reformatting or summarising it."
)


A2A_SKILLS = skills_from_specs()


class WFOAgentExecutor(AgentExecutor):
    """AgentExecutor that drives the capabilities-based agent's event stream.

    Consumes the pydantic-ai event stream and publishes A2A events (status updates, artifacts) via the
    ``TaskUpdater`` helper, one turn at a time per context.

    The workflow form-fill skill runs inside the agent as a capability. The executor's part is the
    transport around it: before the run, a human-in-the-loop response (kagent's extension) becomes the
    text the skill reads; after the run, the skill's reply (``FormTurn.finish``) is delivered as an
    ``input-required`` pause on the same task for a caller that activated the extension, or as
    completed text for any other caller.
    """

    def __init__(self, agent: "WFOAgent") -> None:
        self.agent = agent
        # One turn at a time per context: the state is loaded, mutated and snapshotted per request, and a
        # caller may send two messages for one conversation at once.
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._lock_users: Counter[str] = Counter()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        context_id = context.context_id or ""
        self._lock_users[context_id] += 1
        try:
            async with self._locks[context_id]:
                await self._execute(context, event_queue)
        finally:
            self._lock_users[context_id] -= 1
            if not self._lock_users[context_id]:
                del self._lock_users[context_id]
                self._locks.pop(context_id, None)

    async def _execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        updater = TaskUpdater(event_queue, task_id, context_id)

        if context.current_task is None:  # a new task: the SDK expects it announced before its first status
            history = [context.message] if context.message is not None else None
            await event_queue.enqueue_event(
                new_task(task_id, context_id, TaskState.TASK_STATE_SUBMITTED, history=history)
            )
        await updater.start_work()

        user_input = context.get_user_input()
        auth_token = self._parse_auth_token(context.message)
        hitl = HITL_EXTENSION_URI in context.requested_extensions  # the caller asked for native pauses

        deps = new_deps(user_input=user_input)

        from orchestrator.core.db import db
        from orchestrator.core.db.models import AgentRunTable

        try:
            deps.state.run_id = uuid.uuid4()
            agent_run = AgentRunTable(run_id=deps.state.run_id, thread_id=context_id, agent_type="a2a")
            db.session.add(agent_run)
            db.session.commit()

            logger.debug("A2A execute: starting", task_id=task_id, user_input=user_input[:100] if user_input else "")

            # Multi-turn memory: load the prior conversation for this context and replay it as
            # pydantic-ai message_history, so the proxy only needs to send the latest user turn.
            persistence = PostgresStatePersistence(thread_id=context_id, run_id=deps.state.run_id, session=db.session)
            prior_state = await persistence.load_state()
            message_history = load_messages(prior_state.message_history if prior_state else [])
            # A form being filled spans turns (and, with the extension, pauses on one task): its side of this turn.
            turn = FormTurn.begin(prior_state, context, hitl=hitl, task_id=task_id, user_input=user_input)
            user_input = turn.install(deps.state)

            with bind_outbound_token(auth_token):
                async with self.agent:
                    final_output = await self._run_model(user_input, deps, message_history, updater)

            stop = turn.finish(deps.state)  # before the snapshot: a pause is recorded on the session
            if stop is not None:
                final_output = stop.text  # the contract text, without the model's prose and rendered blocks

            await persistence.snapshot(deps.state)
            db.session.commit()

            # The answer is the agent's prose + any deterministically rendered chart/table blocks, or the
            # form-fill skill's reply.
            payload = stop.payload if stop is not None else None
            message = updater.new_agent_message(
                parts=[new_text_part(final_output or NO_RESULTS)],
                metadata=payload_metadata(payload) if payload else None,
            )
            if payload is None:
                await updater.complete(message=message)
                return
            message.extensions.append(HITL_EXTENSION_URI)
            await updater.requires_input(message)

        except Exception:
            db.session.rollback()
            logger.exception("A2A execute: Task failed", task_id=context.task_id)
            await updater.failed(message=updater.new_agent_message(parts=[new_text_part("Task execution failed")]))

    async def _run_model(
        self, user_input: str, deps: Any, message_history: list[ModelMessage] | None, updater: TaskUpdater
    ) -> str:
        """Stream one model run: artifacts go out as they happen; the prose plus rendered blocks is returned."""
        final_output = ""
        injected_blocks: list[str] = []
        async with self.agent.run_stream_events(user_input, deps=deps, message_history=message_history) as events:
            async for event in events:
                if isinstance(event, FunctionToolResultEvent):
                    result = event.part
                    if isinstance(result, ToolReturnPart) and isinstance(result.metadata, ToolArtifact):
                        data = json.loads(result.model_response_str())
                        await updater.add_artifact(parts=[new_data_part(data)])
                        # Collect chart/table blocks to append after the agent's prose. The agent only sees
                        # raw data (no ToolReturn.content relay), so it just summarises — the adapter
                        # guarantees the block appears.
                        if isinstance(result.metadata, QueryArtifact) and result.metadata.rendered_block:
                            injected_blocks.append(result.metadata.rendered_block.to_markdown())
                elif isinstance(event, AgentRunResultEvent):
                    final_output = str(event.result.output)
                    deps.state.message_history = dump_messages(event.result.all_messages())
                elif not isinstance(event, PartDeltaEvent):
                    logger.debug("A2A execute: event", event_type=type(event).__name__)
        if injected_blocks:
            final_output = final_output.rstrip() + "\n\n" + "\n\n".join(injected_blocks)
        return final_output

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        await updater.cancel()

    @staticmethod
    def _parse_auth_token(message: Message | None) -> str | None:
        """Extract a bearer token from message metadata, if any (for MCP forwarding)."""
        metadata = json_format.MessageToDict(message.metadata) if message is not None else {}
        token = metadata.get("auth_token") or metadata.get("authToken")
        return str(token) if token else None


class A2AAdapter:
    """Wires the A2A protocol layer and adds routes to a FastAPI app.

    Usage::

        adapter = A2AAdapter(agent, url="http://localhost:8080/")
        adapter.add_routes(app)
    """

    def __init__(self, agent: "WFOAgent", url: str = "") -> None:
        self.agent = agent
        self.executor = WFOAgentExecutor(agent)
        self.agent_card = AgentCard(
            name="WFO Agent",
            description=AGENT_CARD_DESCRIPTION,
            version="1.0.0",
            supported_interfaces=[
                AgentInterface(
                    url=url, protocol_binding=TransportProtocol.JSONRPC.value, protocol_version=PROTOCOL_VERSION_1_0
                )
            ],
            capabilities=AgentCapabilities(
                streaming=True,
                extensions=[
                    AgentExtension(uri=HITL_EXTENSION_URI, description=HITL_EXTENSION_DESCRIPTION, required=False)
                ],
            ),
            skills=A2A_SKILLS,
            default_input_modes=["application/json"],
            default_output_modes=["text/markdown", "application/json"],
        )
        self.request_handler = DefaultRequestHandler(
            agent_executor=self.executor, task_store=InMemoryTaskStore(), agent_card=self.agent_card
        )

    def add_routes(self, app: FastAPI) -> None:
        """Add the agent card and the JSON-RPC endpoint to a FastAPI app."""
        add_a2a_routes_to_fastapi(
            app,
            agent_card_routes=create_agent_card_routes(self.agent_card),
            jsonrpc_routes=create_jsonrpc_routes(self.request_handler, rpc_url="/"),
        )
