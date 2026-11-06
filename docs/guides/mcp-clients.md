# How to use the agent with MCP

This guide shows you how to connect an MCP client, such as Claude Code, to the agent so that it can hand WFO
questions to the agent and get answers back. The agent offers one tool per [plugin](../reference/plugins.md)
(`search`, `aggregate`, `entity`, `export`) plus `ask`, each taking a question in plain language.

If you want your client's own model to pick the tools of Orchestrator Core instead, connect to the
[MCP server of Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/) and skip this
guide.

!!! warning "Use this endpoint for questions, not for workflows"
    The agent's `/mcp` has no workflow approval step, and the model behind it can call the write tools of Core. Its
    calls to Orchestrator Core use the agent's service account when outbound authentication is enabled; your token
    is not forwarded. Start workflows from
    [chat](https://workfloworchestrator.org/learn/orchestrator-agent/use-cases/workflows-from-chat/) instead. See
    [Read tools and write tools](../reference/runtime.md#read-tools-and-write-tools) and
    [Authentication and authorization](../architecture/auth.md#outbound-who-the-agent-acts-as).

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

2. Replace the address if your agent listens elsewhere. If it authenticates requests (`OAUTH2_ACTIVE=true`, the
   default), export a valid user token as `WFO_TOKEN` in the shell that launches Claude Code and add this `headers`
   object inside the `orchestrator-agent` server entry:

    ```json
    "headers": {
      "Authorization": "Bearer ${WFO_TOKEN}"
    }
    ```

   Claude Code expands `${WFO_TOKEN}` from its environment. See its
   [MCP configuration documentation](https://code.claude.com/docs/en/mcp#environment-variable-expansion-in-mcp-json).

## Verify it works

1. From the project directory, check that Claude Code can reach the agent:

    ```shell
    claude mcp list
    ```

    `orchestrator-agent` should be listed as connected. Inside a Claude Code session, `/mcp` shows the same status
    and the tools the agent offers.

2. Ask a question that needs your orchestrator:

    ```text
    Using the orchestrator-agent tools, how many active subscriptions are there per product?
    ```

    Depending on your permission settings, Claude Code may ask to call a tool such as
    `mcp__orchestrator-agent__aggregate`. Its answer should contain the counts returned by the agent. If the call fails
    with a 401, the `Authorization` header from step 2 is missing or the token is not valid.

Each MCP tool call starts independently. To export a search result, request the search and export together or pass
the query id returned earlier; the agent does not remember a previous MCP call's results.

## Reference

- [Endpoints and protocols: MCP](../reference/endpoints.md#mcp)
- [Configuration: authentication](../reference/configuration.md#authentication)
