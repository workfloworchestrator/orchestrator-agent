"""Tests for the shared MCP client: outbound token binding."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

import json
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_core import SchemaValidator, core_schema

from orchestrator_agent.auth import token_manager
from orchestrator_agent.mcp_client import (
    _ContextVarBearerAuth,
    _outbound_token,
    bind_outbound_token,
    build_core_toolset,
    decode_stringified_json,
    explain_core_rejection,
    inline_argument_refs,
)


@pytest.mark.parametrize("token,expected", [("tok-123", "tok-123"), (None, None)])
def test_bind_outbound_token(token, expected):
    assert _outbound_token.get() is None
    with bind_outbound_token(token):
        assert _outbound_token.get() == expected
    assert _outbound_token.get() is None


async def _run_request(auth: httpx.Auth, handler) -> tuple[httpx.Response, list[str | None]]:
    """Drive a real httpx request through ``auth`` and return the response plus each attempt's Authorization header."""
    seen_headers = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("authorization"))
        return handler(len(seen_headers))

    async with httpx.AsyncClient(transport=httpx.MockTransport(wrapped), auth=auth) as client:
        response = await client.get("https://core.example.com/mcp/")
    return response, seen_headers


@pytest.mark.parametrize(
    "first_status,expect_refresh",
    [
        pytest.param(401, True, id="401-refreshes-and-retries"),
        pytest.param(403, True, id="403-refreshes-and-retries"),
        pytest.param(200, False, id="200-no-refresh"),
        pytest.param(404, False, id="404-no-refresh"),
    ],
)
async def test_service_token_refreshed_on_auth_error(monkeypatch, first_status, expect_refresh):
    monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", True)
    monkeypatch.setattr(token_manager, "get_token", AsyncMock(return_value="old-token"))
    refresh_mock = AsyncMock(return_value="new-token")
    monkeypatch.setattr(token_manager, "refresh_token", refresh_mock)

    response, seen_headers = await _run_request(
        _ContextVarBearerAuth(), lambda attempt: httpx.Response(first_status if attempt == 1 else 200)
    )

    expected_headers = ["Bearer old-token", "Bearer new-token"] if expect_refresh else ["Bearer old-token"]
    assert seen_headers == expected_headers
    assert refresh_mock.await_count == (1 if expect_refresh else 0)
    assert response.status_code == (first_status if not expect_refresh else 200)


async def test_per_run_token_not_refreshed_on_401(monkeypatch):
    refresh_mock = AsyncMock(return_value="new-token")
    monkeypatch.setattr(token_manager, "refresh_token", refresh_mock)

    with bind_outbound_token("user-token"):
        response, seen_headers = await _run_request(_ContextVarBearerAuth(), lambda _attempt: httpx.Response(401))

    assert seen_headers == ["Bearer user-token"]
    refresh_mock.assert_not_awaited()
    assert response.status_code == 401


async def test_no_retry_when_outbound_auth_disabled(monkeypatch):
    monkeypatch.setattr("orchestrator_agent.auth.agent_settings.OAUTH2_OUTBOUND_ACTIVE", False)
    monkeypatch.setattr(token_manager, "get_token", AsyncMock(return_value=None))
    refresh_mock = AsyncMock(return_value="new-token")
    monkeypatch.setattr(token_manager, "refresh_token", refresh_mock)

    response, seen_headers = await _run_request(_ContextVarBearerAuth(), lambda _attempt: httpx.Response(401))

    assert seen_headers == [None]
    refresh_mock.assert_not_awaited()
    assert response.status_code == 401


FILTERS = {"op": "AND", "children": [{"path": "subscription.status", "condition": {"op": "eq", "value": "active"}}]}
# The shapes core's search tool declares: a nullable referenced model, a plain string and an enum reference.
PROPERTIES = {
    "filters": {"anyOf": [{"$ref": "#/$defs/FilterTree"}, {"type": "null"}]},
    "group_by": {"anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}]},
    "query_text": {"type": "string"},
    "entity_type": {"$ref": "#/$defs/EntityType"},
    "limit": {"type": "integer"},
}


@pytest.mark.parametrize(
    "args,expected",
    [
        pytest.param({"filters": json.dumps(FILTERS)}, {"filters": FILTERS}, id="stringified-object-decoded"),
        pytest.param({"filters": FILTERS}, {"filters": FILTERS}, id="real-object-untouched"),
        pytest.param({"filters": None}, {"filters": None}, id="null-untouched"),
        pytest.param(
            {"group_by": '["status", "product"]'}, {"group_by": ["status", "product"]}, id="stringified-array"
        ),
        pytest.param({"filters": "{not json"}, {"filters": "{not json"}, id="invalid-json-left-for-core-to-reject"),
        pytest.param({"filters": "42"}, {"filters": "42"}, id="json-scalar-not-a-container"),
        pytest.param(
            {"query_text": '{"a": 1}'}, {"query_text": '{"a": 1}'}, id="plain-string-parameter-never-rewritten"
        ),
        pytest.param(
            {"entity_type": "SUBSCRIPTION", "limit": 5}, {"entity_type": "SUBSCRIPTION", "limit": 5}, id="other-types"
        ),
        pytest.param({"unlisted": '{"a": 1}'}, {"unlisted": '{"a": 1}'}, id="argument-not-in-schema"),
    ],
)
def test_decode_stringified_json(args, expected):
    assert decode_stringified_json(args, PROPERTIES) == expected


def _tool(name: str, schema: dict) -> ToolsetTool:
    """A real tool object, as the base toolset returns it: get_tools rebuilds it with dataclasses.replace."""
    return ToolsetTool(
        toolset=cast("Any", None),
        tool_def=ToolDefinition(name=name, parameters_json_schema=schema),
        max_retries=1,
        args_validator=SchemaValidator(core_schema.any_schema()),
    )


async def test_core_toolset_decodes_arguments_before_calling_core(monkeypatch):
    toolset = build_core_toolset()
    schema = {"properties": PROPERTIES}
    tools = {"search": _tool("search", schema)}
    monkeypatch.setattr(MCPToolset, "get_tools", AsyncMock(return_value=tools))
    await toolset.get_tools(None)  # what a run does before the model can call a tool
    call_tool = AsyncMock(return_value="result")

    result = await toolset.process_tool_call(
        None, call_tool, "search", {"entity_type": "SUBSCRIPTION", "filters": json.dumps(FILTERS)}
    )

    assert result == "result"
    call_tool.assert_awaited_once_with("search", {"entity_type": "SUBSCRIPTION", "filters": FILTERS})


async def test_core_toolset_passes_unknown_tool_arguments_through(monkeypatch):
    toolset = build_core_toolset()
    monkeypatch.setattr(MCPToolset, "get_tools", AsyncMock(return_value={}))
    await toolset.get_tools(None)
    call_tool = AsyncMock(return_value="result")

    await toolset.process_tool_call(None, call_tool, "mystery", {"filters": json.dumps(FILTERS)})

    call_tool.assert_awaited_once_with("mystery", {"filters": json.dumps(FILTERS)})


def _rejection(*details: dict) -> str:
    """The tool error text as core's MCP server produced it for a 422."""
    return (
        f"Error calling tool 'search': HTTP error 422: Unprocessable Content - {json.dumps({'detail': list(details)})}"
    )


STRING_FOR_OBJECT = {
    "type": "model_attributes_type",
    "loc": ["body", "filters"],
    "msg": "Input should be a valid dictionary or object to extract fields from",
    "input": "{}",
}
BAD_ENUM = {"type": "enum", "loc": ["body", "entity_type"], "msg": "Input should be 'SUBSCRIPTION' or 'PRODUCT'"}


@pytest.mark.parametrize(
    "error,args,expected_lines,hinted",
    [
        pytest.param(
            _rejection(STRING_FOR_OBJECT),
            {"filters": "{}"},
            [
                "- filters: Input should be a valid dictionary or object to extract fields from (you sent filters as a string"
            ],
            True,
            id="string-sent-for-object-says-so",
        ),
        pytest.param(
            _rejection(STRING_FOR_OBJECT),
            {"filters": {"op": "AND"}},
            ["- filters: Input should be a valid dictionary or object to extract fields from"],
            False,
            id="no-string-hint-when-an-object-was-sent",
        ),
        pytest.param(
            _rejection(BAD_ENUM, STRING_FOR_OBJECT),
            {"entity_type": "NODE", "filters": "{}"},
            ["- entity_type: Input should be 'SUBSCRIPTION' or 'PRODUCT'", "- filters: Input should be a valid"],
            True,
            id="one-line-per-rejected-argument",
        ),
    ],
)
def test_explain_core_rejection_lists_each_argument(error, args, expected_lines, hinted):
    explained = explain_core_rejection(error, args)

    assert explained.startswith("orchestrator-core rejected the arguments, so nothing was executed:")
    assert all(line in explained for line in expected_lines)
    assert ("you sent" in explained) is hinted


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            "Error calling tool 'search': HTTP error 500: Internal Server Error - boom", id="not-a-validation-error"
        ),
        pytest.param(
            "Error calling tool 'search': HTTP error 422: Unprocessable Content - {not json}", id="unparseable-body"
        ),
        pytest.param(
            'Error calling tool \'search\': HTTP error 422: Unprocessable Content - {"detail": "plain text"}',
            id="detail-not-a-list",
        ),
        pytest.param(
            "Error calling tool 'search': HTTP error 422: Unprocessable Content - {\"detail\": []}", id="empty-detail"
        ),
    ],
)
def test_explain_core_rejection_leaves_other_errors_alone(error):
    assert explain_core_rejection(error, {"filters": "{}"}) == error


async def test_core_toolset_rewords_a_validation_rejection_for_the_model(monkeypatch):
    toolset = build_core_toolset()
    schema = {"properties": {"filters": {"type": "string"}}}  # a plain string: the decoder leaves it for core to reject
    tools = {"search": _tool("search", schema)}
    monkeypatch.setattr(MCPToolset, "get_tools", AsyncMock(return_value=tools))
    await toolset.get_tools(None)  # what a run does before the model can call a tool
    call_tool = AsyncMock(side_effect=ModelRetry(_rejection(STRING_FOR_OBJECT)))

    with pytest.raises(ModelRetry, match="you sent filters as a string"):
        await toolset.process_tool_call(None, call_tool, "search", {"filters": "{}"})


# core's schemas: a self-referencing tree (filters), an enum and a union of models behind references
DEFS = {
    "Tree": {"type": "object", "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Tree"}}}},
    "Kind": {"type": "string", "enum": ["A", "B"]},
    "Count": {"type": "object", "properties": {"type": {"const": "count"}}},
    "Field": {"type": "object", "properties": {"type": {"const": "sum"}}},
}
TREE = {"type": "object", "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Tree"}}}}


@pytest.mark.parametrize(
    "argument,expected",
    [
        pytest.param(
            {"$ref": "#/$defs/Kind", "description": "which"},
            {"type": "string", "enum": ["A", "B"], "description": "which"},
            id="plain-reference-keeps-the-argument-description",
        ),
        pytest.param(
            {"anyOf": [{"$ref": "#/$defs/Tree"}, {"type": "null"}], "description": "tree"},
            {"anyOf": [TREE, {"type": "null"}], "description": "tree"},
            id="nullable-reference-recursion-left-alone",
        ),
        pytest.param(
            {
                "anyOf": [
                    {"items": {"anyOf": [{"$ref": "#/$defs/Count"}, {"$ref": "#/$defs/Field"}]}, "type": "array"},
                    {"type": "null"},
                ]
            },
            {"anyOf": [{"items": {"anyOf": [DEFS["Count"], DEFS["Field"]]}, "type": "array"}, {"type": "null"}]},
            id="references-in-a-lists-union-items",
        ),
        pytest.param({"type": "integer", "default": 5}, {"type": "integer", "default": 5}, id="no-reference-untouched"),
        pytest.param(
            {"type": "array", "items": True},
            {"type": "array", "items": True},
            id="boolean-schema-is-not-a-node-to-resolve",
        ),
        pytest.param(
            {"anyOf": [{"anyOf": [{"anyOf": [{"anyOf": [{"anyOf": [{"$ref": "#/$defs/Kind"}]}]}]}]}]},
            {"anyOf": [{"anyOf": [{"anyOf": [{"anyOf": [{"anyOf": [{"$ref": "#/$defs/Kind"}]}]}]}]}]},
            id="reference-beyond-the-hop-limit-left-alone",
        ),
    ],
)
def test_inline_argument_refs(argument, expected):
    schema = {"type": "object", "properties": {"arg": argument}, "$defs": DEFS}

    inlined = inline_argument_refs(schema)

    assert inlined["properties"]["arg"] == expected
    assert inlined["$defs"] == DEFS  # nested references still need their definitions


def test_inline_argument_refs_leaves_a_schema_without_arguments_alone():
    schema = {"type": "object", "$defs": DEFS}

    assert inline_argument_refs(schema) == schema


async def test_core_toolset_sends_schemas_without_argument_level_references(monkeypatch):
    schema = {"type": "object", "properties": {"filters": {"$ref": "#/$defs/Tree"}}, "$defs": DEFS}
    tool = _tool("search", schema)
    monkeypatch.setattr(MCPToolset, "get_tools", AsyncMock(return_value={"search": tool}))

    tools = await build_core_toolset().get_tools(None)

    assert tools["search"].tool_def.parameters_json_schema["properties"]["filters"] == TREE
    assert tool.tool_def.parameters_json_schema["properties"]["filters"] == {"$ref": "#/$defs/Tree"}  # not mutated
