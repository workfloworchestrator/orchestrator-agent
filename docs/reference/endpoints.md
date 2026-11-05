# Endpoints and protocols

The agent is one HTTP service that exposes the same agent over four protocols.

| Client | Protocol | Endpoint |
| --- | --- | --- |
| LibreChat and other OpenAI-compatible clients | OpenAI chat completions | `POST /v1/chat/completions`, `GET /v1/models` |
| Other agents | A2A (protocol v1.0) | `POST /`, agent card at `GET /.well-known/agent-card.json` |
| Web frontends | AG-UI | `POST /agui` |
| MCP clients | MCP | `/mcp` |
| Monitoring | REST | `GET /health` |

## OpenAI chat completions

`POST /v1/chat/completions` serves the agent as a chat model. `GET /v1/models` lists a single model id, `wfo`.
Workflow forms are shown through the client's ask-user tool, which LibreChat provides. See
[Connect LibreChat](../guides/librechat.md) for the headers the agent expects.

This is not the full OpenAI API. It is a small, compatible subset that lets chat clients talk to the agent. The agent
implements it itself, not pydantic-ai.

- Only `POST /v1/chat/completions` and `GET /v1/models` exist. Other OpenAI endpoints, such as embeddings, are not
  available.
- Of the request body the agent uses little, because it keeps its own memory of a conversation:
    - `messages`: only the text of the last `user` message, plus `tool` messages when the request answers a form
      card. Earlier, `system` and `assistant` messages are not read, and neither are non-text parts such as images.
    - `stream`: with `true` the answer comes back as server-sent chunks, but it is computed in full first, so it does
      not arrive token by token.
    - `tools`: only whether the list is empty is checked, which tells the agent whether the chat can show a form
      card. The tools themselves are never called, apart from LibreChat's ask-user tool.
    - `model` is echoed back in the response. The model is always the agent's own.
    - All other fields, such as `temperature`, `max_tokens` and `tool_choice`, are ignored.
- The `X-Conversation-Id` and `X-Agent-Client` headers select the conversation and the client. See
  [Connect LibreChat](../guides/librechat.md).
- A response has one choice, which ends with `stop` or `tool_calls`, and has no `usage`.

For the format of requests and responses, see the
[OpenAI chat completions reference](https://platform.openai.com/docs/api-reference/chat). Only what is listed here
applies to the agent.

## A2A

`POST /` is a JSON-RPC endpoint that speaks A2A protocol v1.0 (`SendMessage`, `SendStreamingMessage`). Clients
discover the agent through its card at `/.well-known/agent-card.json`, which advertises one skill per
[plugin](plugins.md) and the public `BASE_URL`.

Callers that have not moved to v1.0 are still served: the card also advertises a 0.3 interface and the
agent accepts `message/send` and `message/stream`. A 0.3 caller gets everything except workflow forms, which need the
human-in-the-loop extension described in [Workflow forms](runtime.md#workflow-forms).

The endpoint is built on the [A2A Python SDK](https://github.com/a2aproject/a2a-python), not on pydantic-ai. It covers
what a client needs to ask the agent questions, not the whole protocol:

- The only transport is JSON-RPC over HTTP. The REST and gRPC bindings are not offered.
- The card advertises streaming. Tasks are kept in memory, so they are lost when the agent restarts.

For the protocol itself, see the [A2A specification](https://a2a-protocol.org/latest/).

## AG-UI

`POST /agui` streams the agent's answer to a frontend as server-sent events. A follow-up question on the same
thread keeps its history. See [Build a frontend with AG-UI](../guides/ag-ui.md).

The endpoint uses the AG-UI adapter of pydantic-ai, so what it supports is what that adapter supports. See
[AG-UI in pydantic-ai](https://pydantic.dev/docs/ai/integrations/ui/ag-ui/) for how messages, tool calls and state
map to AG-UI events. The agent pins `ag-ui-protocol` 0.1.10, which streams reasoning as `THINKING_*` events.

For the protocol itself, see the [AG-UI documentation](https://docs.ag-ui.com/).

## MCP

`/mcp` exposes the agent's own tools to MCP clients. It is a `FastMCP` server from the
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), over stateless streamable HTTP. See
[Use the agent with MCP](../guides/mcp-clients.md).

It serves tools only: one per [plugin](plugins.md) plus `ask`. It registers no resources or prompts, and workflow
plugins are not available here.

For the protocol itself, see the [MCP documentation](https://modelcontextprotocol.io/).

## Authentication

With `OAUTH2_ACTIVE=true` (the default) incoming requests are authenticated with OIDC. With
`OAUTH2_ACTIVE=false` the agent accepts unauthenticated requests. Calls from the agent to Orchestrator Core
are controlled separately by `OAUTH2_OUTBOUND_ACTIVE`. See [Configuration](configuration.md#authentication).
