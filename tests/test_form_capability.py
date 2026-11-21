"""The form-fill skill as a pydantic-ai capability: the model fills the pages with its tools; the approval is code's."""

from __future__ import annotations

import json
import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from types import SimpleNamespace
from typing import Any

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.ui import StateDeps

from orchestrator_agent.form_fill import FormFillCapability, FormFillSkill
from orchestrator_agent.state import SearchState
from orchestrator_agent.tool_names import (
    CANCEL_FORM_TOOL,
    RESTART_FROM_PAGE_TOOL,
    SEARCH_OPTIONS_TOOL,
    START_WORKFLOW_FORM_TOOL,
    START_WORKFLOW_TOOL,
    SUBMIT_PAGE_TOOL,
)

from .test_form_fill import FILLED, PRODUCT, FakeCore

KEY = "create_demo_lightpath"


class Script:
    """A model that answers with scripted responses, remembering what each request offered it."""

    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.requests: list[list[ModelMessage]] = []
        self.tools: list[list[str]] = []
        self.schemas: list[dict[str, dict]] = []
        self.instructions: list[str] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append(messages)
        self.tools.append(sorted(tool.name for tool in info.function_tools))
        self.schemas.append({tool.name: tool.parameters_json_schema for tool in info.function_tools})
        self.instructions.append(info.instructions or "")
        return self.responses[min(len(self.requests), len(self.responses)) - 1]

    def __len__(self) -> int:
        return len(self.requests)


def _agent(core, script: Script, skill: FormFillSkill | None = None) -> Agent:
    toolset = SimpleNamespace(direct_call_tool=core)
    capability = FormFillCapability(skill or FormFillSkill(), toolset)  # type: ignore[arg-type]

    async def streamed(messages: list[ModelMessage], info: AgentInfo):
        """The scripted response as a streaming model sends it: its text, or its tool calls."""
        for index, part in enumerate(script(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {index: DeltaToolCall(name=part.tool_name, json_args=json.dumps(part.args or {}))}

    model = FunctionModel(script, stream_function=streamed)
    return Agent(
        model,
        deps_type=StateDeps[SearchState],
        capabilities=[capability],  # type: ignore[list-item]  # the capability is typed on these deps
        output_type=[str, DeferredToolRequests],
    )


def _text(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _call(tool: str, **args) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=tool, args=args)])


async def _run(agent: Agent, text: str | None, state: SearchState, *, hitl: bool = True, history=None, answer=None):
    """One run, as an adapter sets it up: the message text, or the person's answer to the start that is waiting."""
    state.user_input, state.hitl = text or "", hitl
    results = DeferredToolResults(approvals=answer) if answer else None
    return await agent.run(text, deps=StateDeps(state), message_history=history, deferred_tool_results=results)


def _results(messages: list[ModelMessage]) -> list[Any]:
    """What the tools answered in the run, in order."""
    return [
        part.content
        for m in messages
        if isinstance(m, ModelRequest)
        for part in m.parts
        if isinstance(part, ToolReturnPart)
    ]


async def _at_the_approval(core) -> tuple[SearchState, list[ModelMessage], str]:
    """A conversation whose form is complete and whose start waits: the state, the messages, the waiting call's id."""
    script = Script(
        _call(START_WORKFLOW_FORM_TOOL, workflow_key=KEY),
        _call(SUBMIT_PAGE_TOOL, product=PRODUCT),
        _call(SUBMIT_PAGE_TOOL, **FILLED),
        _call(SUBMIT_PAGE_TOOL, redundancy="protected"),
        _call(START_WORKFLOW_TOOL),
    )
    state = SearchState()
    result = await _run(_agent(core, script), "create a 10G lightpath for UA, protected", state)
    assert isinstance(result.output, DeferredToolRequests)
    assert state.form_fill is not None and state.form_fill.status == "confirming"
    return state, result.all_messages(), result.output.approvals[0].tool_call_id


class TestTheModelFillsTheForm:
    async def test_without_a_form_the_model_has_the_handoff_only_and_no_form_instructions(self):
        script = Script(_text("719 subscriptions"))
        state = SearchState()
        result = await _run(_agent(FakeCore(), script), "how many subscriptions?", state)
        assert result.output == "719 subscriptions" and state.form_fill is None
        assert script.tools == [[START_WORKFLOW_FORM_TOOL]]
        assert "Filling the workflow form" not in script.instructions[0]

    async def test_the_handoff_opens_the_form_and_the_tools_follow_the_page(self):
        script = Script(
            _call(START_WORKFLOW_FORM_TOOL, workflow_key=KEY),
            _call(SUBMIT_PAGE_TOOL, product=PRODUCT),
            _text("Which speed, and for which customer?"),
        )
        state = SearchState()
        result = await _run(_agent(FakeCore(), script), "create a lightpath", state)
        assert result.output == "Which speed, and for which customer?"  # the model asks, in its own words
        assert state.form_fill.status == "gathering"
        assert state.form_fill.page_inputs == [{"product": PRODUCT}]
        opened, accepted = _results(result.all_messages())
        assert opened["status"] == "opened" and accepted["status"] == "accepted"
        # The tool set is the state: the page tools appear with the form, and submit_page takes the current page.
        assert script.tools[0] == [START_WORKFLOW_FORM_TOOL]
        assert script.tools[1] == sorted(
            [CANCEL_FORM_TOOL, SEARCH_OPTIONS_TOOL, START_WORKFLOW_FORM_TOOL, SUBMIT_PAGE_TOOL]
        )
        assert list(script.schemas[1][SUBMIT_PAGE_TOOL]["properties"]) == ["product"]
        assert RESTART_FROM_PAGE_TOOL in script.tools[2]
        assert set(script.schemas[2][SUBMIT_PAGE_TOOL]["properties"]) == {"customer_name", "speed", "speed_policer"}
        # And the instructions say where the form stands, per request.
        assert "Filling the workflow form `create_demo_lightpath`" in script.instructions[1]
        assert (
            "Current page: 0" in script.instructions[1] and "Current page: 1 (Demo Lightpath)" in script.instructions[2]
        )
        assert f'product = "{PRODUCT}" ("Demo Lightpath")' in script.instructions[2]

    async def test_the_next_message_continues_the_form_with_the_model(self):
        core = FakeCore()
        state = SearchState()
        first = Script(
            _call(START_WORKFLOW_FORM_TOOL, workflow_key=KEY), _call(SUBMIT_PAGE_TOOL, product=PRODUCT), _text("Speed?")
        )
        await _run(_agent(core, first), "create a lightpath for UA", state)
        second = Script(_call(SUBMIT_PAGE_TOOL, customer_name="UA"), _text("Which speed: 1, 10 or 100 Gbit/s?"))
        result = await _run(_agent(core, second), "it is for UA", state)
        assert result.output == "Which speed: 1, 10 or 100 Gbit/s?"
        (rejected,) = _results(result.new_messages())
        assert rejected["status"] == "rejected"  # core judged it; the model asks again with core's words in hand
        assert "The orchestrator rejected that" in second.instructions[1]
        assert "speed: Field required" in second.instructions[1]
        assert state.form_fill.draft == {"customer_name": "UA"}

    async def test_the_start_of_a_complete_form_stops_the_run_for_the_persons_approval(self):
        script = Script(
            _call(START_WORKFLOW_FORM_TOOL, workflow_key=KEY),
            _call(SUBMIT_PAGE_TOOL, product=PRODUCT),
            _call(SUBMIT_PAGE_TOOL, **FILLED),
            _call(SUBMIT_PAGE_TOOL, redundancy="protected"),
            _call(START_WORKFLOW_TOOL),
            _text("I have started it!"),  # never asked for: the run stops on the call
        )
        core, state = FakeCore(), SearchState()
        result = await _run(_agent(core, script), "create a 10G lightpath for UA, protected", state)
        assert isinstance(result.output, DeferredToolRequests) and len(script) == 5
        assert START_WORKFLOW_TOOL not in script.tools[3] and START_WORKFLOW_TOOL in script.tools[4]  # once complete
        (call,) = result.output.approvals  # it takes no values: what is sent are the accepted pages
        assert (call.tool_name, call.args) == (START_WORKFLOW_TOOL, {})
        shown = result.output.metadata[call.tool_call_id]
        assert shown["workflow_key"] == KEY and shown["values"]["speed"] == "10 Gbit/s"
        assert state.form_fill.status == "confirming" and core.created == []

    async def test_a_caller_that_cannot_approve_gets_the_form_prepared(self):
        core = FakeCore()
        script = Script(
            _call(START_WORKFLOW_FORM_TOOL, workflow_key="modify_demo_lightpath", subscription_id=FakeCore.SUB),
            _call(SUBMIT_PAGE_TOOL, note="maintenance Friday"),
            _call(START_WORKFLOW_TOOL),
            _text("The form is complete, but nothing was started."),
        )
        state = SearchState()
        result = await _run(_agent(core, script), "set the note", state, hitl=False)
        assert result.output == "The form is complete, but nothing was started."
        prepared = _results(result.all_messages())[-1]
        assert prepared["status"] == "prepared" and prepared["values"]["note"] == "maintenance Friday"
        assert state.form_fill is None and core.created == []
        assert "cannot approve a start" in script.instructions[1]

    async def test_the_model_can_end_the_form(self):
        core = FakeCore()
        state = SearchState()
        await _run(
            _agent(core, Script(_call(START_WORKFLOW_FORM_TOOL, workflow_key=KEY), _text("Which product?"))),
            "create",
            state,
        )
        script = Script(_call(CANCEL_FORM_TOOL), _text("Cancelled. There are 719 subscriptions."))
        result = await _run(_agent(core, script), "never mind, how many subscriptions?", state)
        assert result.output == "Cancelled. There are 719 subscriptions."
        assert state.form_fill is None
        assert script.tools[1] == [START_WORKFLOW_FORM_TOOL]  # the page tools are gone with the form

    async def test_a_handoff_while_a_form_is_open_is_refused_and_the_form_stays(self):
        core, state = FakeCore(), SearchState()
        script = Script(
            _call(START_WORKFLOW_FORM_TOOL, workflow_key=KEY),
            _call(START_WORKFLOW_FORM_TOOL, workflow_key="create_nsi_light_path"),
            _text("One form at a time."),
        )
        await _run(_agent(core, script), "create a lightpath", state)
        assert state.form_fill.workflow_key == KEY and "is open" in str(script.requests[-1][-1].parts[0].content)

    async def test_a_handoff_to_an_unknown_key_is_a_retry_and_opens_nothing(self):
        script = Script(_call(START_WORKFLOW_FORM_TOOL, workflow_key="create_unicorn"), _text("No such workflow."))
        state = SearchState()
        result = await _run(_agent(FakeCore(), script), "make me a unicorn", state)
        assert result.output == "No such workflow." and len(script) == 2
        assert state.form_fill is None


class TestThePersonDecides:
    """The person's decision is the result of the ``start_workflow`` call: the model reads it like any tool result."""

    async def test_an_approval_starts_the_workflow_and_the_model_says_so(self):
        core = FakeCore()
        state, history, call = await _at_the_approval(core)
        script = Script(_text("Started: process p-1."))
        result = await _run(_agent(core, script), None, state, history=history, answer={call: True})
        assert result.output == "Started: process p-1." and len(core.created) == 1 and state.form_fill is None
        assert _results(result.new_messages()) == [{"status": "started", "process_id": FakeCore.PROCESS_ID}]

    async def test_a_rejection_with_words_goes_to_the_model_which_corrects_and_the_start_waits_again(self):
        core = FakeCore()
        state, history, call = await _at_the_approval(core)
        script = Script(
            _call(RESTART_FROM_PAGE_TOOL, page=1),
            _call(SUBMIT_PAGE_TOOL, **{**FILLED, "speed": "1000"}),
            _call(SUBMIT_PAGE_TOOL),  # the ticket page: nothing to add
            _call(START_WORKFLOW_TOOL),
        )
        result = await _run(_agent(core, script), None, state, history=history, answer={call: ToolDenied("make it 1G")})
        assert "make it 1G" in str(script.requests[0][-1].parts[0].content)  # the denial, as the call's result
        assert script.tools[0] == sorted(
            [CANCEL_FORM_TOOL, RESTART_FROM_PAGE_TOOL, START_WORKFLOW_FORM_TOOL, START_WORKFLOW_TOOL]
        )
        assert isinstance(result.output, DeferredToolRequests) and core.created == []
        (again,) = result.output.approvals
        assert result.output.metadata[again.tool_call_id]["values"]["speed"] == "1 Gbit/s"

    async def test_a_question_at_the_approval_goes_to_the_model_and_the_form_stays_complete(self):
        core = FakeCore()
        state, history, call = await _at_the_approval(core)
        script = Script(_text("Protected means two paths."))
        asked = ToolDenied("what does protected mean?")
        result = await _run(_agent(core, script), None, state, history=history, answer={call: asked})
        assert result.output == "Protected means two paths." and core.created == []
        assert state.form_fill.status == "confirming" and START_WORKFLOW_TOOL in script.tools[0]  # to ask again


class TestStreamedRuns:
    """The routes run the agent as a stream of events: the same seams must hold there."""

    async def test_the_start_to_approve_and_the_approval_work_on_a_streamed_run(self):
        from pydantic_ai.run import AgentRunResultEvent

        async def streamed(agent: Agent, text: str | None, state: SearchState, **more):
            state.hitl = True
            async with agent.run_stream_events(text, deps=StateDeps(state), **more) as events:
                return [event async for event in events if isinstance(event, AgentRunResultEvent)][-1].result

        core = FakeCore()
        script = Script(
            _call(START_WORKFLOW_FORM_TOOL, workflow_key="modify_demo_lightpath", subscription_id=FakeCore.SUB),
            _call(SUBMIT_PAGE_TOOL, note="maintenance Friday"),
            _call(START_WORKFLOW_TOOL),
        )
        state = SearchState()
        waiting = await streamed(_agent(core, script), "set the note to maintenance Friday", state)
        assert isinstance(waiting.output, DeferredToolRequests)
        approved = DeferredToolResults(approvals={waiting.output.approvals[0].tool_call_id: True})
        done = await streamed(
            _agent(core, Script(_text("Started."))),
            None,
            state,
            message_history=waiting.all_messages(),
            deferred_tool_results=approved,
        )
        assert done.output == "Started." and state.form_fill is None and len(core.created) == 1
