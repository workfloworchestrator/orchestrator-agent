"""A person's answers core rejected are interpreted once, a page at a time; agents are never interpreted."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from orchestrator_agent.form_fill import FormFillSkill, ModelInterpreter
from orchestrator_agent.form_fill.core_bridge import page_model
from orchestrator_agent.form_fill.interpret import Interpretation, fields_model, reading_type
from orchestrator_agent.state import Decision, FormFillSession, SearchState

from .test_form_fill import LIGHTPATH_PAGE, SPEED, FakeCore, open_form, rejected, rejection, turn

LIGHTPATH = page_model(LIGHTPATH_PAGE)  # customer_name, speed (labelled options), speed_policer


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

    async def answers(self, form, words):
        self.calls.append(([name for name in form.model_fields if name in words], dict(words)))
        return {name: value for name, value in self.values.items() if name in words}

    async def message(self, form, text, decisions):
        return Interpretation()


REQUEST = '{"customer_name": "UT", "speed": "ten gig", "speed_policer": "yes please"}'


class TestSkillReinterpretsRejectedAnswers:
    async def test_the_pages_rejected_fields_are_interpreted_in_one_call_and_walked_with(self):
        core, state = PickyCore(), SearchState()
        interpreter = FakeInterpreter({"speed": "10000", "speed_policer": True})
        reply = await open_form(FormFillSkill(interpret=interpreter), core, state, "create_demo_lightpath", REQUEST)
        assert interpreter.calls == [(["speed", "speed_policer"], {"speed": "ten gig", "speed_policer": "yes please"})]
        assert state.form_fill.values["speed"] == "10000" and state.form_fill.values["speed_policer"] is True
        assert rejected(reply) == ["redundancy"]  # the walk went on past the corrected page

    async def test_without_an_interpreter_the_rejection_is_the_reply(self):
        core, state = PickyCore(), SearchState()
        reply = await open_form(FormFillSkill(), core, state, "create_demo_lightpath", REQUEST)
        assert rejected(reply) == ["speed", "speed_policer"] and reply.rejected[0]["msg"].startswith("Input should be")
        assert state.form_fill.values["speed"] == "ten gig"

    async def test_answers_that_state_no_value_are_rejected_once_not_looped(self):
        core, state = PickyCore(), SearchState()
        interpreter = FakeInterpreter({})
        skill = FormFillSkill(interpret=interpreter)
        reply = await open_form(skill, core, state, "create_demo_lightpath", REQUEST)
        assert len(interpreter.calls) == 1 and rejected(reply) == ["speed", "speed_policer"]
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
        assert speed.values == ("1000", "10000", "100000") and "— Input should be" in speed.question
        assert policer.name == "speed_policer" and policer.choices == ("true", "false")


class TestReadingType:
    """The interpreter's output type is the page model with every field optional (plus the decision)."""

    def test_one_attribute_per_field_typed_as_its_value(self):
        reading = reading_type(fields_model(LIGHTPATH, ["speed", "speed_policer"]))
        assert list(reading.model_fields) == ["speed", "speed_policer"]
        parsed = reading(speed="10000", speed_policer="true")
        assert parsed.speed == "10000" and parsed.speed_policer is True
        assert reading().speed is None and reading().speed_policer is None
        with pytest.raises(ValidationError):
            reading(speed="42")  # a value outside the options

    def test_types_follow_the_schema_and_the_decision_is_a_literal(self):
        page = page_model(
            {
                "properties": {
                    "n": {"type": "integer"},
                    "p": {
                        "items": {
                            "properties": {"subscription_id": {"type": "string"}},
                            "required": ["subscription_id"],
                        },
                        "type": "array",
                    },
                },
                "required": ["n", "p"],
            }
        )
        reading = reading_type(page, [Decision.CANCEL])
        assert reading(n="12").n == 12
        ports = reading(p=[{"subscription_id": "p-1"}], decision="cancel")
        assert ports.model_dump(exclude_unset=True) == {"p": [{"subscription_id": "p-1"}], "decision": "cancel"}
        with pytest.raises(ValidationError):
            reading(decision="start")  # not offered at this stop


def _scripted(values):
    seen: dict = {}

    def model(messages, info: AgentInfo) -> ModelResponse:
        seen["prompt"] = messages[-1].parts[-1].content
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=values)])

    return FunctionModel(model), seen


class TestModelInterpreter:
    async def test_one_run_covers_the_page_and_the_model_sees_the_asked_fields_with_their_answers(self):
        model, seen = _scripted({"speed": "10000", "speed_policer": True})
        answers = {"speed": "ten gig please", "speed_policer": "yes please"}
        assert await ModelInterpreter(model).answers(LIGHTPATH, answers) == {"speed": "10000", "speed_policer": True}
        # The prompt carries the fields' schema (with the labels behind the values) and the person's words.
        assert '"labels":{"1000":"1 Gbit/s","10000":"10 Gbit/s","100000":"100 Gbit/s"}' in seen["prompt"]
        assert "- speed: ten gig please" in seen["prompt"] and "- speed_policer: yes please" in seen["prompt"]
        assert "Speed Policer" in seen["prompt"] and "customer_name" not in seen["prompt"]  # only what was asked

    async def test_a_single_select_list_stays_a_list_and_nulls_are_left_out(self):
        model, _ = _scripted({"n": ["b"], "m": None})
        page = page_model(
            {
                "properties": {
                    "n": {"items": {"enum": ["a", "b"], "type": "string"}, "maxItems": 1, "type": "array"},
                    "m": {"type": "integer"},
                },
                "required": ["n"],
            }
        )
        assert await ModelInterpreter(model).answers(page, {"n": "the second", "m": "no idea"}) == {"n": ["b"]}

    async def test_a_message_is_read_for_values_and_only_an_outright_decision(self):
        model, seen = _scripted({"speed": "10000", "decision": None})
        read = await ModelInterpreter(model).message(LIGHTPATH, "make it ten gig", [Decision.CANCEL, Decision.START])
        assert read == Interpretation(values={"speed": "10000"}, decision=None)
        assert (
            "Decisions the message may state: cancel, start" in seen["prompt"] and "make it ten gig" in seen["prompt"]
        )
        model, _ = _scripted({"speed": None, "decision": "start"})
        read = await ModelInterpreter(model).message(LIGHTPATH, "yes, go ahead", [Decision.CANCEL, Decision.START])
        assert read == Interpretation(values={}, decision=Decision.START)
