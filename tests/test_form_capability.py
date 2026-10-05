"""The form-fill skill as a pydantic-ai capability: it claims a turn before the model, or walks a handoff after it."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from types import SimpleNamespace

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.ui import StateDeps

from orchestrator_agent.form_fill import FormFillCapability, FormFillSkill
from orchestrator_agent.state import Decision, FormFillSession, FormInput, FormReply, Reply, SearchState
from orchestrator_agent.tool_names import START_WORKFLOW_FORM_TOOL

from .test_form_fill import FakeCore


def _agent(skill: FormFillSkill, core, *responses):
    """An agent whose model answers with the scripted responses, and the form-fill capability over ``core``."""
    calls: list[list[ModelMessage]] = []

    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        calls.append(messages)
        return responses[min(len(calls), len(responses)) - 1]

    toolset = SimpleNamespace(direct_call_tool=core)
    agent = Agent(
        FunctionModel(model),
        deps_type=StateDeps[SearchState],
        capabilities=[FormFillCapability(skill, toolset)],  # type: ignore[arg-type]
    )
    return agent, calls


def _text(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _handoff(key: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=START_WORKFLOW_FORM_TOOL, args={"workflow_key": key})])


async def _run(agent: Agent, text: str, state: SearchState, given: FormInput | None = None, *, hitl: bool = True):
    """One run, as the A2A executor sets it up: the message text, and the human's response it carried (if any)."""
    state.user_input, state.hitl, state.form_input = text, hitl, given
    return await agent.run(text, deps=StateDeps(state))


ANSWERS = FormInput(values={"customer_name": "UT", "speed": "10000"})


class TestFormFillCapability:
    async def test_a_response_to_an_open_form_claims_the_turn_and_the_model_is_skipped(self):
        agent, calls = _agent(FormFillSkill(), FakeCore(), _text("model should not run"))
        state = SearchState(form_fill=FormFillSession(workflow_key="create_demo_lightpath"))
        result = await _run(agent, "Human input supplied", state, ANSWERS)
        assert FormReply.model_validate_json(result.output).title == "Redundancy and ticket" and calls == []
        assert isinstance(state.form_reply, Reply) and state.form_reply.text == result.output
        assert [f.name for f in state.form_reply.ask] == ["redundancy", "ticket_id"]  # the stop the transport pauses on
        # The exchange is in the run's own message history, like any model answer.
        assert [type(m).__name__ for m in result.all_messages()] == ["ModelRequest", "ModelResponse"]
        assert result.all_messages()[-1].parts[0].content == result.output

    async def test_without_a_form_and_without_an_asker_the_model_answers(self):
        agent, calls = _agent(FormFillSkill(), FakeCore(), _text("719 subscriptions"))
        state = SearchState()
        result = await _run(agent, "how many subscriptions?", state)
        assert result.output == "719 subscriptions" and len(calls) == 1 and state.form_reply is None

    async def test_a_handoff_is_walked_in_the_same_run(self):
        agent, calls = _agent(FormFillSkill(), FakeCore(), _handoff("create_demo_lightpath"), _text("Opening."))
        state = SearchState()
        result = await _run(agent, "create a lightpath for UT", state)
        reply = FormReply.model_validate_json(result.output)
        assert (reply.workflow_key, reply.status, reply.page, reply.title) == (
            "create_demo_lightpath",
            "gathering",
            1,
            "Demo Lightpath",
        )
        assert len(calls) == 1  # the model chose the workflow; it never got to say "Opening."
        assert state.form_fill.status == "gathering" and state.form_reply.text == result.output
        assert result.all_messages()[-1].parts[0].content == result.output

    async def test_a_handoff_to_an_unknown_key_leaves_the_models_answer(self):
        agent, calls = _agent(FormFillSkill(), FakeCore(), _handoff("create_unicorn"), _text("No such workflow."))
        state = SearchState()
        result = await _run(agent, "make me a unicorn", state)
        assert result.output == "No such workflow." and len(calls) == 2
        assert state.form_fill is None and state.form_reply is None

    async def test_a_handoff_that_cannot_be_walked_closes_the_form_and_says_so(self):
        class DownError(Exception):
            pass

        async def half_down(name, args):
            if name == "get_workflow_form":
                raise DownError("core unreachable")
            return await FakeCore()(name, args)

        agent, calls = _agent(FormFillSkill(), half_down, _handoff("create_demo_lightpath"), _text("Opening."))
        state = SearchState()
        result = await _run(agent, "create a lightpath for UT", state)
        reply = FormReply.model_validate_json(result.output)  # not the model's "Opening.": no form is open
        assert (reply.status, reply.reason, reply.workflow_key) == (
            "failed",
            "core unreachable",
            "create_demo_lightpath",
        )
        assert state.form_fill is None and len(calls) == 1

    async def test_a_failing_skill_leaves_the_session_as_it_was_and_the_model_answers(self):
        class DownError(Exception):
            pass

        async def broken(name, args):
            raise DownError("core unreachable")

        agent, calls = _agent(FormFillSkill(), broken, _text("model answer"))
        session = FormFillSession(workflow_key="create_demo_lightpath", values={"speed": "10000"})
        state = SearchState(form_fill=session.model_copy(deep=True))
        result = await _run(agent, "Human input supplied", state, FormInput(values={"customer_name": "UT"}))
        assert result.output == "model answer" and len(calls) == 1
        assert state.form_fill == session and state.form_reply is None

    async def test_a_message_without_a_response_is_the_models_and_ends_the_form(self):
        core = FakeCore()
        agent, calls = _agent(FormFillSkill(), core, _text("You have 719 subscriptions."))
        state = SearchState(form_fill=FormFillSession(workflow_key="create_demo_lightpath"))
        await _run(
            agent, "Human input supplied", state, FormInput(values={**ANSWERS.values, "redundancy": "protected"})
        )
        assert state.form_fill.status == "confirming" and calls == []
        result = await _run(agent, "yes, start it", state)  # words are never an approval
        assert result.output == "You have 719 subscriptions." and len(calls) == 1
        assert state.form_fill is None and state.form_reply is None and not core.created

    async def test_for_a_caller_that_cannot_pause_the_handoff_opens_nothing(self):
        agent, calls = _agent(
            FormFillSkill(), FakeCore(), _handoff("create_demo_lightpath"), _text("Not possible here.")
        )
        state = SearchState()
        result = await _run(agent, "create a lightpath for UT", state, hitl=False)
        assert result.output == "Not possible here." and len(calls) == 2  # the model was told, and says so
        assert state.form_fill is None and state.form_reply is None


@pytest.mark.parametrize("decision, status", [(Decision.START, "started"), (Decision.CANCEL, "cancelled")])
async def test_the_humans_decision_goes_through_the_skill_not_the_model(decision, status):
    core = FakeCore()
    agent, calls = _agent(FormFillSkill(), core, _text("model should not run"))
    state = SearchState(form_fill=FormFillSession(workflow_key="create_demo_lightpath"))
    await _run(agent, "Human input supplied", state, FormInput(values={**ANSWERS.values, "redundancy": "protected"}))
    assert state.form_fill.status == "confirming" and state.form_reply.approval is not None
    result = await _run(agent, "Human input supplied", state, FormInput(decision=decision))
    assert calls == [] and len(core.created) == (1 if decision is Decision.START else 0)
    assert FormReply.model_validate_json(result.output).status == status
