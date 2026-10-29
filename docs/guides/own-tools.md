# Give the agent your own tools

!!! info "Goal"
    Let the agent answer questions that the built-in tools cannot, for example about data that only your orchestrator
    has, by exposing a route of your orchestrator as a tool.

## Steps

The agent uses the tools that the MCP server of Orchestrator Core serves. A tool is a FastAPI route with the tag
`agent-exposed`. Orchestrator Core turns it into an MCP tool, and the agent offers it to the model without any change
in the agent.

1. Add a route to your orchestrator and tag it:

    ```python
    from fastapi import APIRouter

    from orchestrator.core.mcp.server import AGENT_EXPOSED_TAG, READONLY_TOOL

    router = APIRouter()


    @router.get(
        "/customers/{customer_id}/contracts",
        tags=[AGENT_EXPOSED_TAG],
        operation_id="list_customer_contracts",
        openapi_extra=READONLY_TOOL,
    )
    def list_customer_contracts(customer_id: str) -> list[Contract]:
        """List the contracts of one customer, newest first.

        Use this when the user asks which contracts or services a customer has.
        """
        ...
    ```

    - `operation_id` becomes the name of the tool.
    - The docstring becomes the description, and the model picks the tool by it. Say when to use the tool, not only
      what it returns. See [How it works](../architecture/index.md#orchestrator-core-search-index-and-mcp-server) for
      why the tool description carries this.
    - The Pydantic models of the route become the parameter schema.
    - `openapi_extra` sets the hints that clients use to judge safety: `READONLY_TOOL`, `WRITE_TOOL`,
      `IDEMPOTENT_WRITE_TOOL` or `DESTRUCTIVE_TOOL`, all from `orchestrator.core.mcp.server`.

2. Restart the orchestrator. Orchestrator Core builds the list of tools once, when `OrchestratorCore(...)` is created.

!!! warning "Expose read-only tools only"
    The agent passes a new tool to the model as it is, in every conversation. It only hides the write tools of
    Orchestrator Core that it knows by name. A tool of yours that changes something would be called by the model
    without the approval of a person. See [why workflows need approval](../architecture/index.md#why-workflows-need-approval).

## Verify it works

Open the API docs of your orchestrator at `http://<core>/api/docs#/agent-exposed`. Every tool is listed there with
its name (the `operationId`), its description and its parameters. Your route should be among them. Then ask the
agent a question that needs the tool, for example *Which contracts does customer 12345678 have?*

## Reference

- [MCP server of Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/), with the default tool set
- [Teach the agent your domain](domain-context.md)
