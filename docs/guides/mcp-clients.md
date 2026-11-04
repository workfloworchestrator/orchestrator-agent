# Use MCP clients

The agent serves its own tools over MCP at `http://<agent>:8080/mcp`: one per [plugin](../reference/plugins.md)
(`search`, `aggregate`, `entity`, `export`) plus `ask`. Each takes a question in plain language, so an MCP client such
as Claude Code can hand WFO questions to the agent and get answers back.

!!! warning "No workflow approval over MCP"
    The workflow plugin and its approval step are not available on `/mcp`, but the write tools of Core are not hidden
    either: the model behind this endpoint can start workflows without asking a person. Use it for questions, and
    start workflows from [chat](workflows-from-chat.md). See
    [Read tools and write tools](../reference/runtime.md#read-tools-and-write-tools).

To have your own client or model pick Core's tools directly instead, connect to the MCP server of Orchestrator Core.
See the [Core MCP reference](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/).

## Steps

1. Add the agent to your client. For Claude Code, create a `.mcp.json` in your project:

    ```json
    {
      "mcpServers": {
        "orchestrator-agent": {
          "type": "http",
          "url": "http://localhost:8080/mcp"
        }
      }
    }
    ```

2. When the agent authenticates requests (`OAUTH2_ACTIVE=true`, the default), send a bearer token from your OIDC
   provider in the `Authorization` header. In Claude Code, add it to the server entry; `${WFO_TOKEN}` is read from
   your environment:

    ```json
    "headers": {
      "Authorization": "Bearer ${WFO_TOKEN}"
    }
    ```

    See [Authentication](../reference/configuration.md#authentication).

## Verify it works

Check from the project directory that Claude Code can reach the agent:

```shell
claude mcp list
```

`orchestrator-agent` should be listed as connected. Inside a Claude Code session, `/mcp` shows the same status and
the tools the agent offers.

Then ask a question that needs your orchestrator:

```text
Using the orchestrator-agent tools, how many active subscriptions are there per product?
```

Claude Code asks permission to call a tool such as `mcp__orchestrator-agent__aggregate`, and answers with the counts
the agent returns. If the call fails with a 401, the `Authorization` header from step 2 is missing or the token is not
valid.

## Reference

- [Endpoints and protocols](../reference/endpoints.md#mcp)
