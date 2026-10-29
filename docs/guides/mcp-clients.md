# Use MCP clients

!!! info "Goal"
    Call the agent, or the MCP server of Orchestrator Core, from any MCP client such as Claude Code or an agent
    framework.

There are two MCP servers, and they serve different tools:

| Server | URL | Tools |
| --- | --- | --- |
| The agent | `http://<agent>:8080/mcp` | The agent itself, with tools such as `search`, `aggregate` and `ask` that take a question in plain language |
| Orchestrator Core | `http://<core>:8080/mcp/` | The raw tools: `search`, `aggregate`, `list_workflows`, `get_subscription_details` and more |

Use the agent when you want answers to questions, and Orchestrator Core when your own client or model should pick the
tools.

## Steps

1. For Orchestrator Core, set `MCP_ENABLED=True` and install it with the `mcp` extra. The transport is streamable HTTP.
2. Add the server to your client. For Claude Code, create a `.mcp.json` in your project:

    ```json
    {
      "mcpServers": {
        "wfo-search": {
          "type": "http",
          "url": "http://localhost:8080/mcp"
        }
      }
    }
    ```

3. When authentication is on, send the same bearer token as for the REST API in the `Authorization` header.

## Verify it works

Run the demo client of the agent repository. It runs a search, an aggregation and an `ask`:

```shell
uv run demos/mcp_client.py
```

Or call one tool:

```shell
uv run demos/mcp_client.py search "active subscriptions"
```

The client connects to `http://localhost:8080/mcp`. Edit `URL` in `demos/mcp_client.py` for another address.

## Reference

- [MCP server of Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/)
- [Endpoints and protocols](../reference/endpoints.md#mcp)
