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

from __future__ import annotations

import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from types import TracebackType
from typing import Any, AsyncIterator, Sequence, cast

import structlog
from fasta2a.applications import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.schema import (
    Artifact,
    Message,
    Part,
    SendMessageResponse,
    Skill,
    Task,
    TaskArtifactUpdateEvent,
    TaskSendParams,
    TaskStatus,
    TaskStatusUpdateEvent,
    a2a_request_ta,
    a2a_response_ta,
    stream_message_response_ta,
)
from fasta2a.schema import TextPart as A2ATextPart
from fasta2a.storage import InMemoryStorage
from pydantic_ai._a2a import AgentWorker
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.messages import (
    TextPart as AiTextPart,
)
from pydantic_ai.run import AgentRunResultEvent
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from orchestrator_agent.agent import AgentAdapter
from orchestrator_agent.artifacts import ToolArtifact
from orchestrator_agent.memory import FALLBACK_MESSAGE, collect_tool_descriptions
from orchestrator_agent.skills import SKILLS
from orchestrator_agent.state import SearchState, TaskAction

logger = structlog.get_logger(__name__)


def _build_state_fallback(state: SearchState) -> str:
    """Build A2A output from execution state when event stream yielded nothing useful."""
    completed = state.memory.completed_turns[-1:]
    if completed:
        return collect_tool_descriptions(completed[-1].steps)
    return FALLBACK_MESSAGE


A2A_SKILLS = [
    Skill(
        id=action.value,
        name=skill.name,
        description=skill.description,
        tags=skill.tags,
        input_modes=["application/json"],
        output_modes=["application/json"],
    )
    for action, skill in SKILLS.items()
]


class A2AWorker(AgentWorker):
    """AgentWorker subclass that consumes the agent's event stream directly.

    Bypasses ``AgentWorker.run_task()`` (which calls ``agent.run()``) and
    instead drives ``agent.run_stream_events()`` — the same pipeline used by
    the AG-UI adapter.  This keeps ``AgentAdapter`` protocol-agnostic while
    giving the A2A adapter full control over stream consumption and result
    assembly.

    The A2A protocol advertises skills on the Agent Card but has no first-class
    field for targeting a skill on ``message/send``. The convention is to pass
    ``{"skill_id": "<action>"}`` in the message metadata.  This worker extracts
    that hint and passes it as ``target_action`` to skip the planner.
    """

    async def _execute_agent(self, user_input: str, message: Message) -> tuple[list[Artifact], str]:
        """Run the agent and collect artifacts and final output.

        Shared by ``run_task``, ``run_task_inline``, and ``run_task_streaming``.
        Returns ``(artifacts, final_output)``.
        """
        agent = cast(AgentAdapter, self.agent)

        target_action = self._parse_target_action(message)

        deps = StateDeps(SearchState(user_input=user_input))

        from orchestrator.db import db
        from orchestrator.db.models import AgentRunTable

        deps.state.run_id = uuid.uuid4()
        agent_run = AgentRunTable(run_id=deps.state.run_id, thread_id=str(uuid.uuid4()), agent_type="a2a")
        db.session.add(agent_run)
        db.session.commit()

        event_stream = agent.run_stream_events(deps=deps, target_action=target_action)
        artifacts: list[Artifact] = []
        final_output = ""

        async for event in event_stream:
            logger.debug("A2A _execute_agent: event", event_type=type(event).__name__)
            if isinstance(event, FunctionToolResultEvent):
                result = event.result
                if isinstance(result, ToolReturnPart) and isinstance(result.metadata, ToolArtifact):
                    artifact_id = str(uuid.uuid4())
                    artifact = Artifact(
                        artifact_id=artifact_id,
                        parts=[A2ATextPart(kind="text", text=result.model_response_str())],
                    )
                    artifacts.append(artifact)
            if isinstance(event, AgentRunResultEvent):
                final_output = str(event.result.output)

        # Prefer artifact content over LLM text (matches collect_stream_output behavior)
        if artifacts:
            final_output = "\n\n".join(
                part["text"] for a in artifacts for part in a["parts"] if part.get("kind") == "text"
            )

        # State-based fallback: use tool step descriptions when events yielded nothing useful
        if not artifacts and (not final_output or final_output in ("", FALLBACK_MESSAGE)):
            final_output = _build_state_fallback(deps.state)

        logger.debug(
            "A2A _execute_agent: collection complete",
            artifact_count=len(artifacts),
            final_output=final_output[:200] if final_output else "",
            query_id=str(deps.state.query_id) if deps.state.query_id else None,
        )

        return artifacts, final_output

    async def _finalize_task(
        self,
        task_id: str,
        context_id: str,
        user_input: str,
        artifacts: list[Artifact],
        final_output: str,
    ) -> None:
        """Update storage with completed task results and multi-turn context."""
        final_text = final_output or "No results"
        agent_message = Message(
            role="agent",
            parts=[A2ATextPart(kind="text", text=final_text)],
            kind="message",
            message_id=str(uuid.uuid4()),
        )

        await self.storage.update_task(
            task_id, state="completed", new_artifacts=artifacts, new_messages=[agent_message]
        )

        context_messages = await self.storage.load_context(context_id) or []
        context_messages.extend(
            [
                ModelRequest(parts=[UserPromptPart(content=user_input)]),
                ModelResponse(parts=[AiTextPart(content=final_text)]),
            ]
        )
        await self.storage.update_context(context_id, context_messages)

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params["id"])
        if task is None:
            raise ValueError(f'Task {params["id"]} not found')

        if task["status"]["state"] != "submitted":
            raise ValueError(f'Task {params["id"]} has already been processed (state: {task["status"]["state"]})')

        await self.storage.update_task(task["id"], state="working")

        user_input = self._extract_user_input(task.get("history", []))
        logger.debug("A2AWorker: Starting execution", task_id=task["id"])

        try:
            artifacts, final_output = await self._execute_agent(user_input, params["message"])
        except Exception:
            await self.storage.update_task(task["id"], state="failed")
            raise

        await self._finalize_task(task["id"], task["context_id"], user_input, artifacts, final_output)

    async def run_task_inline(self, task_id: str, context_id: str, message: Message) -> Task:
        """Run the agent inline and return the completed task.

        Unlike ``run_task()`` which is used by the broker, this method is called
        directly from the ``message/send`` endpoint so the response contains
        the completed task with artifacts and agent messages.
        """
        await self.storage.update_task(task_id, state="working")

        user_input = self._extract_user_input([message])
        logger.debug("A2AWorker: Starting inline execution", task_id=task_id)

        try:
            artifacts, final_output = await self._execute_agent(user_input, message)
        except Exception:
            logger.exception("A2A inline: Task failed", task_id=task_id)
            await self.storage.update_task(task_id, state="failed")
            raise

        await self._finalize_task(task_id, context_id, user_input, artifacts, final_output)

        task = await self.storage.load_task(task_id)
        if task is None:
            raise RuntimeError(f"Task {task_id} not found after inline execution")
        return task

    async def run_task_streaming(
        self, task_id: str, context_id: str, message: Message, jsonrpc_id: int | str | None
    ) -> AsyncIterator[bytes]:
        """Run the agent inline and yield SSE events for the full task lifecycle.

        Unlike ``run_task()`` which is fire-and-forget via the broker, this method
        streams ``TaskStatusUpdateEvent`` and ``TaskArtifactUpdateEvent`` SSE events
        back to the client as the agent executes.
        """
        agent = cast(AgentAdapter, self.agent)

        def _sse_encode(result: TaskStatusUpdateEvent | TaskArtifactUpdateEvent) -> bytes:
            envelope = {"jsonrpc": "2.0", "id": jsonrpc_id, "result": result}
            return b"data: " + stream_message_response_ta.dump_json(envelope, by_alias=True) + b"\n\n"

        # 1. Mark working
        await self.storage.update_task(task_id, state="working")
        yield _sse_encode(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                kind="status-update",
                status=TaskStatus(state="working"),
                final=False,
            )
        )

        target_action = self._parse_target_action(message)

        user_input = self._extract_user_input([message])
        deps = StateDeps(SearchState(user_input=user_input))

        from orchestrator.db import db
        from orchestrator.db.models import AgentRunTable

        deps.state.run_id = uuid.uuid4()
        agent_run = AgentRunTable(run_id=deps.state.run_id, thread_id=str(uuid.uuid4()), agent_type="a2a")
        db.session.add(agent_run)
        db.session.commit()

        try:
            event_stream = agent.run_stream_events(deps=deps, target_action=target_action)
            artifacts: list[Artifact] = []
            final_output = ""

            async for event in event_stream:
                logger.debug("A2A stream: event", event_type=type(event).__name__)
                if isinstance(event, FunctionToolResultEvent):
                    result = event.result
                    if isinstance(result, ToolReturnPart) and isinstance(result.metadata, ToolArtifact):
                        artifact_id = str(uuid.uuid4())
                        artifact = Artifact(
                            artifact_id=artifact_id,
                            parts=[A2ATextPart(kind="text", text=result.model_response_str())],
                        )
                        artifacts.append(artifact)
                        yield _sse_encode(
                            TaskArtifactUpdateEvent(
                                task_id=task_id,
                                context_id=context_id,
                                kind="artifact-update",
                                artifact=artifact,
                            )
                        )
                if isinstance(event, AgentRunResultEvent):
                    final_output = str(event.result.output)

            # State-based fallback: use tool step descriptions when events yielded nothing useful
            if not artifacts and (not final_output or final_output in ("", FALLBACK_MESSAGE)):
                final_output = _build_state_fallback(deps.state)

            # Build final agent message
            if not final_output and artifacts:
                final_output = "\n\n".join(
                    part["text"] for a in artifacts for part in a["parts"] if part.get("kind") == "text"
                )
            agent_message = Message(
                role="agent",
                parts=[A2ATextPart(kind="text", text=final_output or "No results")],
                kind="message",
                message_id=str(uuid.uuid4()),
            )

            # Update storage
            await self.storage.update_task(
                task_id,
                state="completed",
                new_artifacts=artifacts,
                new_messages=[agent_message],
            )

            # Update context for multi-turn
            context_messages = await self.storage.load_context(context_id) or []
            context_messages.extend(
                [
                    ModelRequest(parts=[UserPromptPart(content=user_input)]),
                    ModelResponse(parts=[AiTextPart(content=final_output or "No results")]),
                ]
            )
            await self.storage.update_context(context_id, context_messages)

            # 2. Final completed event
            yield _sse_encode(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    kind="status-update",
                    status=TaskStatus(state="completed", message=agent_message),
                    final=True,
                )
            )
        except Exception:
            logger.exception("A2A stream: Task failed", task_id=task_id)
            await self.storage.update_task(task_id, state="failed")
            yield _sse_encode(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    kind="status-update",
                    status=TaskStatus(state="failed"),
                    final=True,
                )
            )

    @staticmethod
    def _parse_target_action(message: Message) -> TaskAction | None:
        """Extract target action from message metadata, if any."""
        metadata = message.get("metadata", {}) or {}
        skill_id = metadata.get("skill_id") or metadata.get("skillId")
        if not skill_id:
            return None
        try:
            action = TaskAction(skill_id)
            logger.debug("A2A: Routing to skill directly", target_action=action)
            return action
        except ValueError:
            logger.warning("A2A: Unknown skillId, falling back to planner", skill_id=skill_id)
            return None

    @staticmethod
    def _extract_user_input(history: list[Message] | Sequence[Message]) -> str:
        """Extract user input text from A2A task history messages."""

        def _is_text_part(part: Part) -> bool:
            return part.get("kind") == "text"

        for msg in reversed(history):
            if msg.get("role") == "user":
                for part in msg.get("parts", []):
                    if _is_text_part(part):
                        return cast(A2ATextPart, part)["text"]
        return ""


class StreamingFastA2A(FastA2A):
    """FastA2A subclass that handles ``message/stream`` requests.

    For ``message/stream``, bypasses the broker and runs the agent inline
    within the SSE generator, yielding ``TaskStatusUpdateEvent`` and
    ``TaskArtifactUpdateEvent`` events as the agent executes.
    """

    def __init__(self, *args: Any, worker: A2AWorker | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._worker = worker

    async def _agent_run_endpoint(self, request: Request) -> Response:
        data = await request.body()
        a2a_request = a2a_request_ta.validate_json(data)

        method = a2a_request["method"]
        if method == "message/stream":
            message = a2a_request["params"]["message"]
            context_id = message.get("context_id") or str(uuid.uuid4())
            task = await self.task_manager.storage.submit_task(context_id, message)
            jsonrpc_id = a2a_request.get("id")

            if self._worker is None:
                raise RuntimeError("StreamingFastA2A requires a worker for message/stream")

            return StreamingResponse(
                self._worker.run_task_streaming(task["id"], context_id, message, jsonrpc_id),
                media_type="text/event-stream",
            )
        if method == "message/send":
            if self._worker is None:
                raise RuntimeError("StreamingFastA2A requires a worker for message/send")

            message = a2a_request["params"]["message"]
            context_id = message.get("context_id") or str(uuid.uuid4())
            task = await self.task_manager.storage.submit_task(context_id, message)
            jsonrpc_id = a2a_request.get("id")
            completed_task = await self._worker.run_task_inline(task["id"], context_id, message)
            jsonrpc_response = SendMessageResponse(jsonrpc="2.0", id=jsonrpc_id, result=completed_task)
        elif method == "tasks/get":
            jsonrpc_response = await self.task_manager.get_task(a2a_request)
        elif method == "tasks/cancel":
            jsonrpc_response = await self.task_manager.cancel_task(a2a_request)
        else:
            raise NotImplementedError(f"Method {method} not implemented.")
        return Response(
            content=a2a_response_ta.dump_json(jsonrpc_response, by_alias=True), media_type="application/json"
        )


class A2AApp:
    """A2A adapter app: FastA2A server, worker, and lifecycle.

    Builds the FastA2A app, broker, storage, and worker so that the worker
    lifecycle can be tied to the host application's lifespan when mounted
    as a sub-app.
    """

    def __init__(self, agent: AgentAdapter, url: str = "") -> None:
        self.agent = agent

        storage = InMemoryStorage()
        broker = InMemoryBroker()
        self.worker = A2AWorker(agent=agent, broker=broker, storage=storage)  # type: ignore[arg-type]

        @asynccontextmanager
        async def _noop_lifespan(_app: FastA2A) -> AsyncIterator[None]:
            yield

        self.app = StreamingFastA2A(
            storage=storage,
            broker=broker,
            worker=self.worker,
            name="WFO Search Agent",
            url=url,
            description="Search, filter and aggregate orchestration data",
            skills=A2A_SKILLS,
            lifespan=_noop_lifespan,
        )

    async def __aenter__(self) -> A2AApp:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        await self._stack.enter_async_context(self.app.task_manager)
        await self._stack.enter_async_context(self.agent)
        await self._stack.enter_async_context(self.worker.run())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._stack.__aexit__(exc_type, exc_val, exc_tb)
