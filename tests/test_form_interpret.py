"""A person's answers core rejected are interpreted once, a page at a time; agents are never interpreted."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from pydantic import ValidationError
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from orchestrator_agent.form_fill import FormFillSkill, ModelInterpreter
from orchestrator_agent.form_fill.interpret import output_type
from orchestrator_agent.state import FormField, FormFillSession, SearchState

from .test_form_fill import SPEED, FakeCore, open_form, turn

SPEED_FIELD = FormField(name="speed", title="Speed", kind="choice", options=SPEED["options"])
POLICER_FIELD = FormField(name="speed_policer", title="Speed Policer", kind="boolean")


def rejection(problems: dict[str, str]) -> str:
    """The tool error text fastmcp builds from core's 400 body, naming each rejected field."""
    errors = ", ".join(
        f"{{'loc': ('{name}',), 'msg': \"{msg}\", 'type': 'value_error'}}" for name, msg in problems.items()
    )
    return f"HTTP error 400: Bad Request - {{'type': 'FormValidationError', 'validation_errors': [{errors}], 'status': 400}}"


class PickyCore(FakeCore):
    """Rejects a lightpath page whose speed is not an allowed value or whose policer is not a boolean, the way core does."""

    async def __call__(self, name, args):
        if name == "get_workflow_form" and len(args["page_inputs"]) >= 2:
            page = args["page_inputs"][1]
            problems = {}
            if page.get("speed") not in SPEED["enum"]:
                problems["speed"] = "Input should be '1000', '10000' or '100000'"
            if not isinstance(page.get("speed_policer", False), bool):
                problems["speed_policer"] = "Input should be a valid boolean"
            if problems:
                raise ModelRetry(rejection(problems))
        return await super().__call__(name, args)


class FakeInterpreter:
    def __init__(self, values):
        self.values, self.calls = values, []

    async def __call__(self, fields, answers):
        self.calls.append(([f.name for f in fields], dict(answers)))
        return {name: value for name, value in self.values.items() if name in answers}


REQUEST = '{"customer_name": "UT", "speed": "ten gig", "speed_policer": "yes please"}'


class TestSkillReinterpretsRejectedAnswers:
    async def test_the_pages_rejected_fields_are_interpreted_in_one_call_and_walked_with(self):
        core, state = PickyCore(), SearchState()
        interpreter = FakeInterpreter({"speed": "10000", "speed_policer": True})
        reply = await open_form(FormFillSkill(interpret=interpreter), core, state, "create_demo_lightpath", REQUEST)
        assert interpreter.calls == [(["speed", "speed_policer"], {"speed": "ten gig", "speed_policer": "yes please"})]
        assert state.form_fill.values["speed"] == "10000" and state.form_fill.values["speed_policer"] is True
        assert "- redundancy (required)" in reply  # the walk went on past the corrected page

    async def test_without_an_interpreter_the_rejection_is_the_reply(self):
        core, state = PickyCore(), SearchState()
        reply = await open_form(FormFillSkill(), core, state, "create_demo_lightpath", REQUEST)
        assert reply.startswith("The orchestrator rejected the values") and "speed: Input should be" in reply
        assert state.form_fill.values["speed"] == "ten gig"

    async def test_answers_that_state_no_value_are_rejected_once_not_looped(self):
        core, state = PickyCore(), SearchState()
        interpreter = FakeInterpreter({})
        skill = FormFillSkill(interpret=interpreter)
        reply = await open_form(skill, core, state, "create_demo_lightpath", REQUEST)
        assert len(interpreter.calls) == 1 and reply.startswith("The orchestrator rejected the values")
        # The same words are not interpreted again on the next turn; new words are.
        await turn(skill, core, state, '{"ticket_id": "T-1"}')
        assert len(interpreter.calls) == 1
        await turn(skill, core, state, '{"speed": "ten gigabit"}')
        assert interpreter.calls[-1][1] == {"speed": "ten gigabit"}

    async def test_the_rejected_fields_are_asked_again_as_questions_with_their_options(self):
        core, state = PickyCore(), SearchState()
        state.form_fill = FormFillSession(workflow_key="create_demo_lightpath", status="opening", request=REQUEST)
        reply = await FormFillSkill().open(state, core)
        speed, policer = reply.ask
        assert speed.name == "speed" and speed.choices == ("1 Gbit/s", "10 Gbit/s", "100 Gbit/s")
        assert speed.values == ("1000", "10000", "100000") and "rejected: Input should be" in speed.question
        assert policer.name == "speed_policer" and policer.choices == ("true", "false")


class TestOutputType:
    def test_one_attribute_per_field_typed_as_its_value(self):
        interpretation = output_type([SPEED_FIELD, POLICER_FIELD])
        parsed = interpretation(speed="10000", speed_policer="true")
        assert parsed.speed == "10000" and parsed.speed_policer is True
        assert interpretation().speed is None and interpretation().speed_policer is None
        try:
            interpretation(speed="42")
        except ValidationError:
            pass
        else:
            raise AssertionError("a value outside the options was accepted")

    def test_kinds_map_to_types(self):
        assert output_type([FormField(name="n", title="N", kind="integer")])(n="12").n == 12
        ports = output_type([FormField(name="p", title="P", kind="json")])(p=[{"subscription_id": "p-1"}])
        assert ports.p == [{"subscription_id": "p-1"}]


def _scripted(values):
    seen: dict = {}

    def model(messages, info: AgentInfo) -> ModelResponse:
        seen["prompt"] = messages[-1].parts[-1].content
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=values)])

    return FunctionModel(model), seen


class TestModelInterpreter:
    async def test_one_run_covers_the_page_and_the_model_sees_every_field_with_its_answer(self):
        model, seen = _scripted({"speed": "10000", "speed_policer": True})
        answers = {"speed": "ten gig please", "speed_policer": "yes please"}
        assert await ModelInterpreter(model)([SPEED_FIELD, POLICER_FIELD], answers) == {
            "speed": "10000",
            "speed_policer": True,
        }
        assert "`10000` (10 Gbit/s)" in seen["prompt"] and "ten gig please" in seen["prompt"]
        assert "Speed Policer" in seen["prompt"] and "yes please" in seen["prompt"]

    async def test_a_single_select_list_is_wrapped_and_nulls_are_left_out(self):
        model, _ = _scripted({"n": "b", "m": None})
        n = FormField(name="n", title="N", kind="choice", options={"a": "A", "b": "B"}, as_list=True)
        m = FormField(name="m", title="M", kind="integer")
        assert await ModelInterpreter(model)([n, m], {"n": "the second", "m": "no idea"}) == {"n": ["b"]}
