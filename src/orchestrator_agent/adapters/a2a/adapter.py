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

The endpoint serves A2A 1.0 and, through the SDK's compatibility layer, A2A 0.3: a 0.3 caller gets
everything but workflow forms, which need the HITL extension.

The workflow form-fill skill runs inside the agent as a capability and is answered by a human in the
loop: for a caller that activated kagent's HITL extension, a stop of the skill goes out as an
``input-required`` pause on the task with the questions (or the start to approve) as its payload, and
the human's response comes back as a message carrying the matching payload. No form is opened for any
other caller.

``A2A_SKILLS`` is static A2A protocol metadata (the agent's advertised skills) and is
unrelated to pydantic-ai capabilities — it just describes what the agent can do.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import structlog
from a2a.helpers import new_data_part, new_task, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentExtension, AgentInterface, Message, TaskState
from a2a.utils.constants import PROTOCOL_VERSION_0_3, PROTOCOL_VERSION_1_0, TransportProtocol
from fastapi import FastAPI
from google.protobuf import json_format
from pydantic_ai.messages import ToolReturnPart

from orchestrator_agent.adapters.a2a.kagent import HITL_EXTENSION_DESCRIPTION, HITL_EXTENSION_URI, KagentHitl
from orchestrator_agent.adapters.hitl import HumanInTheLoop
from orchestrator_agent.capabilities.spec import skills_from_specs
from orchestrator_agent.persistence import PostgresStatePersistence
from orchestrator_agent.state import FormInput
from orchestrator_agent.turn import NO_RESULTS, ConversationLocks, OnArtifact, run_turn

if TYPE_CHECKING:
    from orchestrator_agent.agent import WFOAgent

logger = structlog.get_logger(__name__)


# A client's way of showing a stop, over this protocol: its request in, and out the metadata of the
# ``input-required`` message that pauses the task (the stop's payload, under its extension's URI).
A2AHitl = HumanInTheLoop[RequestContext, dict[str, Any]]

# kagent stamps the chat conversation a call belongs to on every hop (root: the user-facing chat; parent:
# the agent calling us). Its remote-agent tool's own A2A context is not that conversation: by default it
# is one per tool, shared by every chat that tool serves, and with session isolation a new one per call.
_LINEAGE_HEADERS = ("x-kagent-root-context-id", "x-kagent-parent-context-id")

# Card-level tool description a consuming agent reads. The output-handling note covers only the
# artifact-bearing replies (a result with a pre-rendered table/chart), not every tool call.
AGENT_CARD_DESCRIPTION = (
    "Answers questions about orchestration data and starts workflows by walking the user through their "
    "input forms: ask for one (what, and for which subscription or product). A form is filled by a person "
    "through human-in-the-loop pauses (kagent's HITL extension): the task pauses with the questions of a "
    "form page, and at the end with the start to approve; nothing is started without that approval, and a "
    "caller that does not activate the extension cannot start a workflow. When a paused call returns "
    "before the form is finished, call again to show the next step. When a reply includes a pre-rendered "
    "Markdown table or Mermaid chart, relay that block to the user verbatim, without reformatting or "
    "summarising it."
)


A2A_SKILLS = skills_from_specs()


class WFOAgentExecutor(AgentExecutor):
    """AgentExecutor that drives the capabilities-based agent's event stream.

    Consumes the pydantic-ai event stream and publishes A2A events (status updates, artifacts) via the
    ``TaskUpdater`` helper, one turn at a time per context.

    The workflow form-fill skill runs inside the agent as a capability. The executor's part is the
    transport around it — the human-in-the-loop transport of the client that is calling (``adapters.hitl``;
    kagent's extension in ``kagent``, by default the only one): before the run, the conversation's open form and the
    human's response carried by the message go on the run state; after the run, a stop of the skill is
    delivered as an ``input-required`` pause on the task, and anything else as completed text.
    """

    def __init__(self, agent: "WFOAgent", transports: Sequence[A2AHitl] | None = None) -> None:
        self.agent = agent
        self.transports: Sequence[A2AHitl] = (KagentHitl(),) if transports is None else transports
        self._locks = ConversationLocks()

    @staticmethod
    def _conversation(context: RequestContext) -> str:
        """The conversation this call belongs to: what its memory and its open form are keyed on.

        The chat a parent agent forwards with the call when it does (``_LINEAGE_HEADERS``), otherwise the
        A2A context. Keying on the context alone would let one chat continue, or end, another chat's form
        whenever a parent reuses a context across chats, and lose the form when it opens one per call.
        """
        headers = (context.call_context.state.get("headers") or {}) if context.call_context else {}
        return next((headers[name] for name in _LINEAGE_HEADERS if headers.get(name)), context.context_id or "")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        async with self._locks.turn(self._conversation(context)):
            await self._execute(context, event_queue)

    async def _execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        conversation = self._conversation(context)
        updater = TaskUpdater(event_queue, task_id, context_id)

        if context.current_task is None:  # a new task: the SDK expects it announced before its first status
            history = [context.message] if context.message is not None else None
            await event_queue.enqueue_event(
                new_task(task_id, context_id, TaskState.TASK_STATE_SUBMITTED, history=history)
            )
        await updater.start_work()

        user_input = context.get_user_input()
        auth_token = self._parse_auth_token(context.message)
        # The transport of the client that is calling: it asked for native pauses (a form needs them).
        transport = next((t for t in self.transports if t.shows_stops(context)), None)

        from orchestrator.core.db import db

        try:
            run_id = uuid.uuid4()
            logger.debug("A2A execute: starting", task_id=task_id, user_input=user_input[:100] if user_input else "")

            # The conversation's memory and open form. A form being filled spans turns: it is what the
            # skill continues, with the human's response to its stop when this message carries one.
            persistence = PostgresStatePersistence(thread_id=conversation, run_id=run_id, session=db.session)
            prior_state = await persistence.load_state()
            session = prior_state.form_fill if prior_state is not None else None
            response = transport.read(session, context) if transport is not None else None
            if isinstance(response, dict):  # the transport answers this message itself
                await self._send(updater, NO_RESULTS, response)
                return

            state, final_output = await run_turn(
                self.agent,
                db.session,
                thread=conversation,
                run_id=run_id,
                agent_type="a2a",
                prior_state=prior_state,
                user_input=user_input,
                hitl=transport is not None,
                form_input=response if isinstance(response, FormInput) else None,
                auth_token=auth_token,
                on_artifact=self._publisher(updater),
            )

            reply = state.form_reply
            stop = None
            if reply is not None:
                # The skill's reply, in place of the model's prose and rendered blocks — as a pause when
                # it is a stop (recorded on the session, so before the snapshot).
                final_output = transport.words(reply) if transport is not None else reply.text
                stop = transport.pause(reply, state.form_fill, context) if transport is not None else None

            await persistence.snapshot(state)
            db.session.commit()
            await self._send(updater, final_output or NO_RESULTS, stop)

        except Exception:
            db.session.rollback()
            logger.exception("A2A execute: Task failed", task_id=context.task_id)
            await updater.failed(message=updater.new_agent_message(parts=[new_text_part("Task execution failed")]))

    @staticmethod
    async def _send(updater: TaskUpdater, text: str, stop: dict[str, Any] | None) -> None:
        """The turn's answer: completed text, or — as a stop — an ``input-required`` pause carrying its payload."""
        message = updater.new_agent_message(parts=[new_text_part(text)], metadata=stop)
        if stop is None:
            await updater.complete(message=message)
            return
        message.extensions.extend(stop)  # a payload sits under the URI of the extension that defines it
        await updater.requires_input(message)

    @staticmethod
    def _publisher(updater: TaskUpdater) -> OnArtifact:
        """Tool results that carry an artifact go out as A2A artifacts (a data part), as they happen."""

        async def publish(result: ToolReturnPart) -> None:
            data = json.loads(result.model_response_str())
            await updater.add_artifact(parts=[new_data_part(data)])

        return publish

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
            # One endpoint, both protocol generations: A2A 1.0, and 0.3 for callers that have not moved yet
            # (kagent before 1.0). The SDK adds the legacy card fields for the 0.3 interface.
            supported_interfaces=[
                AgentInterface(url=url, protocol_binding=TransportProtocol.JSONRPC.value, protocol_version=version)
                for version in (PROTOCOL_VERSION_1_0, PROTOCOL_VERSION_0_3)
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
            jsonrpc_routes=create_jsonrpc_routes(self.request_handler, rpc_url="/", enable_v0_3_compat=True),
        )
