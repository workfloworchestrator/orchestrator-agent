"""The form-fill skill as a pydantic-ai capability: the model fills the pages with its tools; the approval is code's."""

from __future__ import annotations

import json
import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from types import SimpleNamespace

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.ui import StateDeps

from orchestrator_agent.form_fill import FormFillCapability, FormFillSkill
from orchestrator_agent.form_fill import session as fs
from orchestrator_agent.form_fill.pending import pending_approval
from orchestrator_agent.state import Decision, FormFillSession, FormInput, FormReply, Reply, SearchState
from orchestrator_agent.tool_names import (
    CANCEL_FORM_TOOL,
    RESTART_FROM_PAGE_TOOL,
    SEARCH_OPTIONS_TOOL,
    START_WORKFLOW_FORM_TOOL,
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
    return Agent(model, deps_type=StateDeps[SearchState], capabilities=[capability])


def _text(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _call(tool: str, **args) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=tool, args=args)])


async def _run(agent: Agent, text: str, state: SearchState, given: FormInput | None = None, *, hitl: bool = True):
    """One run, as an adapter sets it up: the message text, and the person's response it carried (if any)."""
    state.user_input, state.hitl, state.form_input = text, hitl, given
    return await agent.run(text, deps=StateDeps(state))


def _results(messages: list[ModelMessage]) -> list[dict]:
    """What the tools answered in the run, in order."""
    return [
        part.content
        for m in messages
        if isinstance(m, ModelRequest)
        for part in m.parts
        if isinstance(part, ToolReturnPart)
    ]


async def _at_the_approval(core) -> SearchState:
    """A conversation whose form is complete and whose approval the transport has shown."""
    script = Script(
        _call(START_WORKFLOW_FORM_TOOL, workflow_key=KEY),
        _call(SUBMIT_PAGE_TOOL, product=PRODUCT),
        _call(SUBMIT_PAGE_TOOL, **FILLED),
        _call(SUBMIT_PAGE_TOOL, redundancy="protected"),
    )
    state = SearchState()
    await _run(_agent(core, script), "create a 10G lightpath for UA, protected", state)
    assert state.form_fill is not None and state.form_fill.status == "confirming"
    fs.paused(state.form_fill, pending_approval("req-1"), unseen=False)  # what the transport's pause does
    return state


class TestTheModelFillsTheForm:
    async def test_without_a_form_the_model_has_the_handoff_only_and_no_form_instructions(self):
        script = Script(_text("719 subscriptions"))
        state = SearchState()
        result = await _run(_agent(FakeCore(), script), "how many subscriptions?", state)
        assert result.output == "719 subscriptions" and state.form_reply is None and state.form_fill is None
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
        assert state.form_reply is None and state.form_fill.status == "gathering"
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

    async def test_a_complete_form_ends_the_run_on_the_approval_and_the_model_writes_no_summary(self):
        script = Script(
            _call(START_WORKFLOW_FORM_TOOL, workflow_key=KEY),
            _call(SUBMIT_PAGE_TOOL, product=PRODUCT),
            _call(SUBMIT_PAGE_TOOL, **FILLED),
            _call(SUBMIT_PAGE_TOOL, redundancy="protected"),
            _text("I have started it!"),  # never asked for
        )
        state = SearchState()
        result = await _run(_agent(FakeCore(), script), "create a 10G lightpath for UA, protected", state)
        reply = FormReply.model_validate_json(result.output)
        assert reply.status == "confirming" and reply.labels["speed"] == "10 Gbit/s" and len(script) == 4
        assert isinstance(state.form_reply, Reply) and state.form_reply.text == result.output
        assert state.form_reply.approval is not None and state.form_reply.approval.tool_name == "create_workflow"
        assert state.form_fill.status == "confirming"
        # The exchange is in the run's own message history, like any model answer.
        assert result.all_messages()[-1].parts[0].content == result.output

    async def test_a_caller_that_cannot_approve_gets_the_form_prepared(self):
        core = FakeCore()
        script = Script(
            _call(START_WORKFLOW_FORM_TOOL, workflow_key="modify_demo_lightpath", subscription_id=FakeCore.SUB),
            _call(SUBMIT_PAGE_TOOL, note="maintenance Friday"),
        )
        state = SearchState()
        result = await _run(_agent(core, script), "set the note", state, hitl=False)
        reply = FormReply.model_validate_json(result.output)
        assert reply.status == "prepared" and reply.values["note"] == "maintenance Friday"
        assert state.form_fill is None and state.form_reply.approval is None and core.created == []
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
        assert state.form_fill is None and state.form_reply is None
        assert script.tools[1] == [START_WORKFLOW_FORM_TOOL]  # the page tools are gone with the form

    async def test_a_handoff_to_an_unknown_key_is_a_retry_and_opens_nothing(self):
        script = Script(_call(START_WORKFLOW_FORM_TOOL, workflow_key="create_unicorn"), _text("No such workflow."))
        state = SearchState()
        result = await _run(_agent(FakeCore(), script), "make me a unicorn", state)
        assert result.output == "No such workflow." and len(script) == 2
        assert state.form_fill is None and state.form_reply is None


class TestThePersonDecides:
    async def test_an_approval_starts_the_workflow_and_the_model_is_skipped(self):
        core = FakeCore()
        state = await _at_the_approval(core)
        script = Script(_text("model should not run"))
        result = await _run(_agent(core, script), "[form response]", state, FormInput(decision=Decision.START))
        reply = FormReply.model_validate_json(result.output)
        assert reply.status == "started" and reply.process_id == FakeCore.PROCESS_ID and len(script) == 0
        assert state.form_fill is None and len(core.created) == 1 and state.form_reply.approval is None

    async def test_a_rejection_with_words_goes_to_the_model_which_corrects_and_the_approval_returns(self):
        core = FakeCore()
        state = await _at_the_approval(core)
        script = Script(
            _call(RESTART_FROM_PAGE_TOOL, page=1),
            _call(SUBMIT_PAGE_TOOL, **{**FILLED, "speed": "1000"}),
            _call(SUBMIT_PAGE_TOOL),  # the ticket page: nothing to add
        )
        given = FormInput(decision=Decision.CANCEL, reason="make it 1G")
        result = await _run(_agent(core, script), "make it 1G", state, given)
        assert 'did not approve the start and said: "make it 1G"' in script.instructions[0]
        assert script.tools[0] == sorted([CANCEL_FORM_TOOL, RESTART_FROM_PAGE_TOOL, START_WORKFLOW_FORM_TOOL])
        reply = FormReply.model_validate_json(result.output)
        assert reply.status == "confirming" and reply.values["speed"] == "1000" and "redundancy" not in reply.values
        assert state.form_reply.approval is not None and state.form_fill.pending is None  # a new approval to show
        assert core.created == []

    async def test_a_failing_skill_leaves_the_session_as_it_was_and_the_model_answers(self):
        class DownError(Exception):
            pass

        async def down(name, args):
            raise DownError("core unreachable")

        state = SearchState(
            form_fill=FormFillSession(
                workflow_key=KEY, status="confirming", page_inputs=[{"a": 1}], pending={"id": "r"}
            )
        )
        before = state.form_fill.model_copy(deep=True)

        class Broken(FormFillSkill):
            async def handle(self, state, call_tool):
                state.form_fill.page_inputs.clear()
                raise DownError("boom")

        script = Script(_text("Something went wrong, try again."))
        result = await _run(_agent(down, script, Broken()), "x", state, FormInput(decision=Decision.START))
        assert result.output == "Something went wrong, try again." and state.form_fill == before


class TestStreamedRuns:
    """The A2A and chat routes run the agent as a stream of events: the same seams must hold there."""

    async def test_the_decision_and_the_approval_work_on_a_streamed_run(self):
        from pydantic_ai.run import AgentRunResultEvent

        async def streamed(agent: Agent, text: str, state: SearchState, given: FormInput | None = None) -> str:
            state.user_input, state.hitl, state.form_input = text, True, given
            output = ""
            async with agent.run_stream_events(text, deps=StateDeps(state)) as events:
                async for event in events:
                    if isinstance(event, AgentRunResultEvent):
                        output = str(event.result.output)
            return output

        core = FakeCore()
        script = Script(
            _call(START_WORKFLOW_FORM_TOOL, workflow_key="modify_demo_lightpath", subscription_id=FakeCore.SUB),
            _call(SUBMIT_PAGE_TOOL, note="maintenance Friday"),
        )
        state = SearchState()
        output = await streamed(_agent(core, script), "set the note to maintenance Friday", state)
        assert FormReply.model_validate_json(output).status == "confirming" and state.form_reply.approval is not None
        fs.paused(state.form_fill, pending_approval("req-1"), unseen=False)
        idle = Script(_text("model should not run"))
        output = await streamed(_agent(core, idle), "[form response]", state, FormInput(decision=Decision.START))
        assert FormReply.model_validate_json(output).status == "started" and len(idle) == 0
        assert state.form_fill is None and len(core.created) == 1
