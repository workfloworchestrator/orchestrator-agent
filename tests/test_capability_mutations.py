"""Mutation opt-in belongs to a real capability and a selected set of tools."""

import pytest
from pydantic_ai import ExternalToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from orchestrator_agent.agent import build_agent, new_deps
from orchestrator_agent.tool_names import WRITE_TOOL_NAMES
from orchestrator_agent.tool_policy import allow_capability_mutations


class MutationCapability(AbstractCapability):
    id = "maintenance"
    description = "Explicit maintenance mutations"

    def __init__(self, toolset, names, *, bypass=True, deferred=False):
        self.toolset = toolset
        self.names = names
        self.bypass = bypass
        self.defer_loading = deferred

    def get_toolset(self):
        if not self.bypass:
            return self.toolset
        return allow_capability_mutations(self.toolset, tool_names=self.names)


def _tools(name):
    calls = []
    source = FunctionToolset()

    async def selected() -> str:
        calls.append(name)
        return "Mutation complete"

    async def unselected() -> str:
        calls.append("unselected_write")
        return "Unselected mutation"

    source.add_function(selected, name=name, metadata={"annotations": {"readOnlyHint": False}})
    source.add_function(unselected, name="unselected_write", metadata={"annotations": {"readOnlyHint": False}})
    return source, calls


@pytest.mark.parametrize("source_kind", ["local", "core"])
@pytest.mark.parametrize("name", ["modify_customer", *WRITE_TOOL_NAMES])
@pytest.mark.parametrize("bypass", [False, True])
async def test_only_explicitly_selected_capability_mutations_execute(monkeypatch, source_kind, name, bypass):
    source, calls = _tools(name)
    core = source if source_kind == "core" else FunctionToolset()
    monkeypatch.setattr("orchestrator_agent.agent.build_core_toolset", lambda: core)
    cap = MutationCapability(source, {name}, bypass=bypass)
    seen = []

    def model(messages, info):
        seen.append({tool.name for tool in info.function_tools})
        if len(seen) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="unselected_write", args={})])
        if len(seen) == 2:
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args={})])
        return ModelResponse(parts=[TextPart(content="Finished")])

    agent = build_agent(FunctionModel(model))
    async with agent:
        result = await agent.run("Perform maintenance", deps=new_deps(), capabilities=[cap])

    assert result.output == "Finished"
    assert calls == ([name] if bypass else [])
    assert all("unselected_write" not in tools and (name in tools) is bypass for tools in seen)


async def test_a_wrapped_toolset_without_a_capability_grants_no_bypass(monkeypatch):
    source, calls = _tools("resume_workflow_process")
    monkeypatch.setattr("orchestrator_agent.agent.build_core_toolset", FunctionToolset)
    turns = []

    def model(messages, info):
        turns.append(info)
        if len(turns) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="resume_workflow_process", args={})])
        return ModelResponse(parts=[TextPart(content="No mutation")])

    agent = build_agent(FunctionModel(model))
    async with agent:
        result = await agent.run(
            "Resume",
            deps=new_deps(),
            toolsets=[allow_capability_mutations(source, tool_names={"resume_workflow_process"})],
        )
    assert result.output == "No mutation" and calls == []


async def test_deferred_capability_cannot_mutate_until_loaded(monkeypatch):
    source, calls = _tools("modify_customer")
    monkeypatch.setattr("orchestrator_agent.agent.build_core_toolset", FunctionToolset)
    cap = MutationCapability(source, {"modify_customer"}, deferred=True)
    seen = []

    def model(messages, info):
        seen.append({tool.name for tool in info.function_tools})
        if len(seen) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="modify_customer", args={})])
        if len(seen) == 2:
            return ModelResponse(parts=[ToolCallPart(tool_name="load_capability", args={"id": "maintenance"})])
        if len(seen) == 3:
            return ModelResponse(parts=[ToolCallPart(tool_name="modify_customer", args={})])
        return ModelResponse(parts=[TextPart(content="Finished")])

    agent = build_agent(FunctionModel(model))
    async with agent:
        result = await agent.run("Change customer", deps=new_deps(), capabilities=[cap])
    assert result.output == "Finished" and calls == ["modify_customer"]
    assert "modify_customer" not in seen[0] and "modify_customer" not in seen[1]
    assert "modify_customer" in seen[2]


async def test_capability_bypass_never_enables_frontend_tools(monkeypatch):
    monkeypatch.setattr("orchestrator_agent.agent.build_core_toolset", FunctionToolset)
    source = ExternalToolset([ToolDefinition(name="modify_customer", parameters_json_schema={"type": "object"})])
    cap = MutationCapability(source, {"modify_customer"})
    turns = []

    def model(messages, info):
        turns.append(info)
        if len(turns) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="modify_customer", args={})])
        return ModelResponse(parts=[TextPart(content="No frontend mutation")])

    agent = build_agent(FunctionModel(model))
    async with agent:
        result = await agent.run("Change customer", deps=new_deps(), capabilities=[cap])
    assert result.output == "No frontend mutation"
    assert all("modify_customer" not in {tool.name for tool in turn.function_tools} for turn in turns)


@pytest.mark.parametrize("name", ["load_capability", "start_workflow_form"])
def test_bypass_rejects_reserved_local_orchestration_names(name):
    with pytest.raises(ValueError, match="Local orchestration names"):
        allow_capability_mutations(FunctionToolset(), tool_names={name})


def test_bypass_requires_explicit_tool_selection():
    with pytest.raises(ValueError, match="Select at least one tool"):
        allow_capability_mutations(FunctionToolset(), tool_names=set())
