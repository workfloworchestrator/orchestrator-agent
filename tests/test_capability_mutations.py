"""Mutation opt-in uses a real capability or the exact-name bypass list."""

from unittest.mock import AsyncMock

import pytest
from pydantic_ai import ExternalToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from orchestrator_agent.agent import build_agent, new_deps
from orchestrator_agent.form_fill import FormFillSkill
from orchestrator_agent.settings import AgentSettings, agent_settings
from orchestrator_agent.tool_names import WRITE_TOOL_NAMES


class MutationCapability(AbstractCapability):
    id = "maintenance"
    description = "Explicit maintenance mutations"

    def __init__(self, toolset, *, deferred=False):
        self.toolset = toolset
        self.defer_loading = deferred

    def get_toolset(self):
        return self.toolset


def _tools(name):
    calls = []
    source = FunctionToolset()

    async def selected() -> str:
        calls.append(name)
        return "Mutation complete"

    async def unselected() -> str:
        calls.append("unselected_write")
        return "Unselected mutation"

    metadata = None if name == "custom_unannotated_tool" else {"annotations": {"readOnlyHint": False}}
    source.add_function(selected, name=name, metadata=metadata)
    source.add_function(unselected, name="unselected_write", metadata={"annotations": {"readOnlyHint": False}})
    return source, calls


@pytest.mark.parametrize("source_kind", ["local", "core"])
@pytest.mark.parametrize("name", ["modify_customer", *WRITE_TOOL_NAMES])
@pytest.mark.parametrize("registered", [False, True])
async def test_capability_supplied_mutations_execute_without_an_extra_wrapper(
    monkeypatch, source_kind, name, registered
):
    source, calls = _tools(name)
    core = source if source_kind == "core" else FunctionToolset()
    monkeypatch.setattr("orchestrator_agent.agent.build_core_toolset", lambda: core)
    cap = MutationCapability(source.filtered(lambda _ctx, tool: tool.name == name))
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
        result = await agent.run("Perform maintenance", deps=new_deps(), capabilities=[cap] if registered else None)

    assert result.output == "Finished"
    assert calls == ([name] if registered else [])
    assert all("unselected_write" not in tools and (name in tools) is registered for tools in seen)


async def test_a_toolset_without_a_capability_or_bypass_entry_cannot_mutate(monkeypatch):
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
            toolsets=[source],
        )
    assert result.output == "No mutation" and calls == []


@pytest.mark.parametrize("bypass", [False, True])
async def test_deferred_capability_cannot_mutate_until_loaded(monkeypatch, bypass):
    source, calls = _tools("modify_customer")
    monkeypatch.setattr("orchestrator_agent.agent.build_core_toolset", FunctionToolset)
    monkeypatch.setattr(agent_settings, "AGENT_TOOL_BYPASS", ["modify_customer"] if bypass else [])
    cap = MutationCapability(source.filtered(lambda _ctx, tool: tool.name == "modify_customer"), deferred=True)
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


@pytest.mark.parametrize("bypass", [False, True])
async def test_capability_and_bypass_list_never_enable_frontend_tools(monkeypatch, bypass):
    monkeypatch.setattr("orchestrator_agent.agent.build_core_toolset", FunctionToolset)
    source = ExternalToolset([ToolDefinition(name="modify_customer", parameters_json_schema={"type": "object"})])
    monkeypatch.setattr(agent_settings, "AGENT_TOOL_BYPASS", ["modify_customer"] if bypass else [])
    cap = MutationCapability(source)
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
        AgentSettings(AGENT_TOOL_BYPASS=[name])


@pytest.mark.parametrize("forms", [False, True])
@pytest.mark.parametrize("source_kind", ["core", "local"])
@pytest.mark.parametrize("name", ["modify_customer", "custom_unannotated_tool", *WRITE_TOOL_NAMES])
async def test_bypass_list_allows_only_named_tools_without_a_capability(monkeypatch, forms, source_kind, name):
    source, calls = _tools(name)
    core = source if source_kind == "core" else FunctionToolset()
    monkeypatch.setattr(core, "direct_call_tool", AsyncMock(), raising=False)
    monkeypatch.setattr("orchestrator_agent.agent.build_core_toolset", lambda: core)
    monkeypatch.setattr(agent_settings, "AGENT_TOOL_BYPASS", [name])
    seen = []

    def model(messages, info):
        seen.append({tool.name for tool in info.function_tools})
        if len(seen) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="unselected_write", args={})])
        if len(seen) == 2:
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args={})])
        return ModelResponse(parts=[TextPart(content="Finished")])

    agent = build_agent(FunctionModel(model), form_fill=FormFillSkill() if forms else None)
    async with agent:
        result = await agent.run(
            "Perform maintenance", deps=new_deps(), toolsets=[source] if source_kind == "local" else None
        )

    assert result.output == "Finished" and calls == [name]
    assert all(name in tools and "unselected_write" not in tools for tools in seen)
    core.direct_call_tool.assert_not_awaited()


def test_bypass_list_is_empty_by_default_and_loads_from_json_environment(monkeypatch):
    monkeypatch.delenv("AGENT_TOOL_BYPASS", raising=False)
    assert AgentSettings().AGENT_TOOL_BYPASS == []
    monkeypatch.setenv("AGENT_TOOL_BYPASS", '["resume_workflow_process", "modify_customer"]')
    assert AgentSettings().AGENT_TOOL_BYPASS == ["resume_workflow_process", "modify_customer"]
