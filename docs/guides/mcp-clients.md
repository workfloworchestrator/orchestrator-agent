# How to use the agent with MCP

This guide shows you how to connect an MCP client, such as Claude Code, to the agent so that it can hand WFO
questions to the agent and get answers back. The agent offers one tool per [plugin](../reference/plugins.md)
(`search`, `aggregate`, `entity`, `export`) plus `ask`, each taking a question in plain language.

If you want your client's own model to pick the tools of Orchestrator Core instead, connect to the
[MCP server of Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/) and skip this
guide.

!!! warning "Use this endpoint for questions, not for workflows"
    The agent's `/mcp` has no workflow approval step, and the model behind it can call the write tools of Core. Its
    calls to Orchestrator Core also always run as the agent's service account, not as you. Start workflows from
    [chat](workflows-from-chat.md) instead. See
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

2. If the agent authenticates requests (`OAUTH2_ACTIVE=true`, the default), add a bearer token from your OIDC
   provider to the server entry. Claude Code reads `${WFO_TOKEN}` from your environment:

    ```json
    "headers": {
      "Authorization": "Bearer ${WFO_TOKEN}"
    }
    ```

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

    Claude Code asks permission to call a tool such as `mcp__orchestrator-agent__aggregate`, and answers with the
    counts the agent returns. If the call fails with a 401, the `Authorization` header from step 2 is missing or the
    token is not valid.

## Reference

- [Endpoints and protocols: MCP](../reference/endpoints.md#mcp)
- [Configuration: authentication](../reference/configuration.md#authentication)
