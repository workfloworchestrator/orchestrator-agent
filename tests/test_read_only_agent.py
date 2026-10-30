"""The real agent assembly must reject writes and undeclared tools on every interface."""

from unittest.mock import AsyncMock

import pytest
from ag_ui.core import RunAgentInput, Tool, UserMessage
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.ui.ag_ui import AGUIAdapter

from orchestrator_agent.agent import build_agent, new_deps
from orchestrator_agent.form_fill import FormFillSkill
from orchestrator_agent.tool_names import WRITE_TOOL_NAMES


@pytest.mark.parametrize("forms", [False, True], ids=["mcp-and-agui", "a2a-and-chat"])
@pytest.mark.parametrize("attempted_tool", [*WRITE_TOOL_NAMES, "delete_customer", "custom_unannotated_tool"])
@pytest.mark.parametrize("extra_toolset", [False, True], ids=["core-tool", "frontend-tool"])
async def test_model_cannot_execute_writes_or_undeclared_tools(monkeypatch, forms, attempted_tool, extra_toolset):
    effects = []
    core = FunctionToolset()

    async def mutation() -> str:
        effects.append(attempted_tool)
        return "Changed customer"

    async def search() -> dict:
        return {"results": []}

    core.add_function(search, name="search", metadata={"annotations": {"readOnlyHint": True}})
    # Known workflow writes remain blocked even if mislabeled. Other tools need an explicit read-only declaration.
    extra = FunctionToolset() if extra_toolset else core
    metadata = (
        None
        if attempted_tool == "custom_unannotated_tool"
        else {"annotations": {"readOnlyHint": attempted_tool in WRITE_TOOL_NAMES}}
    )
    extra.add_function(mutation, name=attempted_tool, metadata=metadata)
    monkeypatch.setattr(core, "direct_call_tool", AsyncMock(), raising=False)
    monkeypatch.setattr("orchestrator_agent.agent.build_core_toolset", lambda: core)
    seen = []

    def model(messages, info):
        seen.append({tool.name for tool in info.function_tools})
        if len(seen) == 1:
            # Deliberately guess a hidden tool name, bypassing the model's advertised schemas.
            return ModelResponse(parts=[ToolCallPart(tool_name=attempted_tool, args={})])
        if len(seen) == 2:
            return ModelResponse(parts=[ToolCallPart(tool_name="search", args={})])
        return ModelResponse(parts=[TextPart(content="Finished reading")])

    agent = build_agent(FunctionModel(model), form_fill=FormFillSkill() if forms else None)
    async with agent:
        result = await agent.run("Change this customer", deps=new_deps(), toolsets=[extra] if extra_toolset else None)

    assert result.output == "Finished reading"
    assert effects == []
    assert all(attempted_tool not in tools and "search" in tools for tools in seen)
    core.direct_call_tool.assert_not_awaited()


@pytest.mark.parametrize("attempted_tool", ["delete_customer", "get_process_status"])
async def test_agui_frontend_cannot_supply_mutations_under_unknown_or_allowed_names(monkeypatch, attempted_tool):
    core = FunctionToolset()
    monkeypatch.setattr("orchestrator_agent.agent.build_core_toolset", lambda: core)
    seen = []

    def model(messages, info):
        seen.append({tool.name for tool in info.function_tools})
        if len(seen) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name=attempted_tool, args={})])
        return ModelResponse(parts=[TextPart(content="No frontend action")])

    agent = build_agent(FunctionModel(model))
    adapter = AGUIAdapter(
        agent=agent,
        run_input=RunAgentInput(
            thread_id="thread",
            run_id="run",
            messages=[UserMessage(id="user", content="Change this customer")],
            tools=[Tool(name=attempted_tool, description="Change customer", parameters={"type": "object"})],
            state={},
            context=[],
            forwarded_props={},
        ),
    )
    async with agent:
        result = await agent.run("Change this customer", deps=new_deps(), toolsets=[adapter.toolset])

    assert result.output == "No frontend action"  # No deferred frontend action was emitted either.
    assert all(attempted_tool not in tools for tools in seen)
