"""What a person typed for a field and core rejected is interpreted once, a page at a time; nothing else is read."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from orchestrator.core.schemas.mcp_tools import WorkflowFormPage
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from orchestrator_agent.form_fill import FormFillSkill, ModelInterpreter
from orchestrator_agent.form_fill.interpret import fields_schema, reading_schema
from orchestrator_agent.form_fill.model import form_schema
from orchestrator_agent.state import SearchState

from .test_form_fill import (
    LIGHTPATH_FIELDS,
    LIGHTPATH_PAGE,
    SPEED,
    FakeCore,
    choice_list,
    errors,
    field,
    open_form,
    page,
    rejected,
    turn,
)


def form_of(*pages) -> dict:
    """The walked pages' askable fields as one schema, as the skill hands it to the interpreter."""
    walked = [WorkflowFormPage.model_validate(page(i, title, fields)) for i, (title, fields) in enumerate(pages)]
    return form_schema(walked)


LIGHTPATH = form_of(LIGHTPATH_PAGE)  # customer_name, speed (labelled options), speed_policer


class PickyCore(FakeCore):
    """Rejects a lightpath page whose speed is not an allowed value or whose policer is not a boolean, the way core does."""

    async def __call__(self, name, args):
        if name == "get_workflow_form" and len(args["page_inputs"]) >= 2:
            submitted = args["page_inputs"][1]
            problems = {}
            if submitted.get("speed") not in SPEED:
                problems["speed"] = "Input should be '1000', '10000' or '100000'"
            if not isinstance(submitted.get("speed_policer", False), bool):
                problems["speed_policer"] = "Input should be a valid boolean"
            if problems:
                return page(1, "Demo Lightpath", LIGHTPATH_FIELDS, errors(problems))
        return await super().__call__(name, args)


class FakeInterpreter:
    def __init__(self, values):
        self.values, self.calls = values, []

    async def answers(self, form, words):
        self.calls.append(([name for name in form["properties"] if name in words], dict(words)))
        return {name: value for name, value in self.values.items() if name in words}


REQUEST = {"customer_name": "UA", "speed": "ten gig", "speed_policer": "yes please"}  # what the person typed


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
        assert rejected(reply) == ["speed", "speed_policer"] and reply.rejected[0].msg.startswith("Input should be")
        # The rejected fields are asked again as questions with their options, and core's reason on each.
        speed, policer = reply.question("speed"), reply.question("speed_policer")
        assert reply.asked == ["speed", "speed_policer"] and "— Input should be" in speed.question
        assert speed.choices == ("1 Gbit/s", "10 Gbit/s", "100 Gbit/s") and speed.values == ("1000", "10000", "100000")
        assert policer.choices == ("True", "False") and policer.values == (True, False)
        assert state.form_fill.values["speed"] == "ten gig"

    async def test_answers_that_state_no_value_are_rejected_once_not_looped(self):
        core, state = PickyCore(), SearchState()
        interpreter = FakeInterpreter({})
        skill = FormFillSkill(interpret=interpreter)
        reply = await open_form(skill, core, state, "create_demo_lightpath", REQUEST)
        assert len(interpreter.calls) == 1 and rejected(reply) == ["speed", "speed_policer"]
        # The same words are not interpreted again on the next turn; new words are.
        await turn(skill, core, state, {"ticket_id": "T-1"})
        assert len(interpreter.calls) == 1
        await turn(skill, core, state, {"speed": "ten gigabit"})
        assert interpreter.calls[-1][1] == {"speed": "ten gigabit"}


class TestReadingSchema:
    """The interpreter's output schema is the page schema with every field optional."""

    def test_one_property_per_asked_field_as_core_defined_it(self):
        reading = reading_schema(fields_schema(LIGHTPATH, ["speed", "speed_policer"]))
        assert list(reading["properties"]) == ["speed", "speed_policer"] and "required" not in reading
        assert reading["properties"]["speed_policer"] == {"default": False, "title": "Speed Policer", "type": "boolean"}
        assert "customer_name" not in reading["properties"]  # not asked: not read

    def test_a_choice_between_several_options_is_read_as_every_option_the_words_fit(self):
        form = form_of(
            (
                None,
                [
                    field("speed", options=SPEED),
                    field("only", options={"x": "x"}),
                    choice_list("many", {"a": "a", "b": "b"}),
                    field("confirm", format="accept"),
                ],
            )
        )
        properties = reading_schema(form)["properties"]
        speed = properties["speed"]
        assert speed["type"] == "array" and speed["title"] == "Speed" and "enum" not in speed
        assert speed["items"]["enum"] == list(SPEED) and speed["items"]["options"] == SPEED  # shown as people call them
        # One option, a list of options and consent are read as the value itself: there is nothing to tell apart.
        assert properties["only"]["enum"] == ["x"] and "items" not in properties["only"]
        assert properties["many"]["type"] == "array" and properties["many"]["items"]["enum"] == ["a", "b"]
        assert properties["confirm"]["format"] == "accept" and "items" not in properties["confirm"]

    def test_the_schema_is_cores_own_less_what_is_only_shown_and_the_widget_hints(self):
        port = field("", "object", required=False, fields=[field("subscription_id")])
        shown = field("note", required=False, format="label", read_only=True)
        form = form_of((None, [field("n", "integer"), field("p", "list", item=port), shown]))
        reading = reading_schema(form)
        assert list(reading["properties"]) == ["n", "p"]  # a field that is only shown is no reading's
        ports = reading["properties"]["p"]["items"]["properties"]
        assert ports == {"subscription_id": {"title": "Subscription Id", "type": "string"}}
        assert "decision" not in reading["properties"]  # a decision is never something a model reads
        hinted = {"type": "string", "enum": ["p-1"], "uniforms": {"productIds": ["p-1"]}, "title": "Product"}
        schema = {"type": "object", "properties": {"product": hinted}, "required": ["product"]}
        walked = [WorkflowFormPage(page=0, complete=False, status="next", schema=schema)]
        assert form_schema(walked)["properties"]["product"] == {"type": "string", "enum": ["p-1"], "title": "Product"}


def _scripted(values):
    seen: dict = {}

    def model(messages, info: AgentInfo) -> ModelResponse:
        seen["prompt"] = messages[-1].parts[-1].content
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=values)])

    return FunctionModel(model), seen


class TestModelInterpreter:
    async def test_one_run_covers_the_page_and_the_model_sees_the_asked_fields_with_their_answers(self):
        model, seen = _scripted({"speed": ["10000"], "speed_policer": True})
        answers = {"speed": "ten gig please", "speed_policer": "yes please"}
        assert await ModelInterpreter(model).answers(LIGHTPATH, answers) == {"speed": "10000", "speed_policer": True}
        # The prompt carries the fields' schema (with the labels behind the values) and the person's words.
        assert '"options":{"1000":"1 Gbit/s","10000":"10 Gbit/s","100000":"100 Gbit/s"}' in seen["prompt"]
        assert "- speed: ten gig please" in seen["prompt"] and "- speed_policer: yes please" in seen["prompt"]
        assert "Speed Policer" in seen["prompt"] and "customer_name" not in seen["prompt"]  # only what was asked

    async def test_a_single_select_list_stays_a_list_and_nulls_are_left_out(self):
        model, _ = _scripted({"n": ["b"], "m": None})
        page = form_of(
            (
                None,
                [
                    choice_list("n", {"a": "a", "b": "b"}, constraints={"max_length": 1}),
                    field("m", "integer", required=False, nullable=True),
                ],
            )
        )
        assert await ModelInterpreter(model).answers(page, {"n": "the second", "m": "no idea"}) == {"n": ["b"]}

    async def test_words_that_fit_more_than_one_option_or_none_leave_the_field_open(self):
        # "fast" names two speeds: picking one would be a guess, so the person is asked again.
        for fits in (["10000", "100000"], []):
            model, _ = _scripted({"speed": fits, "customer_name": "UA"})
            words = {"speed": "fast", "customer_name": "for UT"}
            assert await ModelInterpreter(model).answers(LIGHTPATH, words) == {"customer_name": "UA"}
