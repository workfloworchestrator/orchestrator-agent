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

## A2A

`POST /` is a JSON-RPC endpoint that speaks A2A protocol v1.0 (`SendMessage`, `SendStreamingMessage`). Clients
discover the agent through its card at `/.well-known/agent-card.json`, which advertises one skill per
[plugin](plugins.md) and the public `BASE_URL`.

Callers that have not moved to v1.0 are still served: the card also advertises a 0.3 interface and the
agent accepts `message/send` and `message/stream`. A 0.3 caller gets everything except workflow forms, which need the
human-in-the-loop extension described in [How it works](../architecture/index.md#workflow-form-fill).

## AG-UI

`POST /agui` streams the agent's answer to a frontend as server-sent events. A follow-up question on the same
thread keeps its history. See [Build a frontend with AG-UI](../guides/ag-ui.md).

## MCP

`/mcp` exposes the agent's own tools to MCP clients. See [Use MCP clients](../guides/mcp-clients.md).

## Authentication

With `OAUTH2_ACTIVE=true` (the default) incoming requests are authenticated with OIDC. With
`OAUTH2_ACTIVE=false` the agent accepts unauthenticated requests. Calls from the agent to Orchestrator Core
are controlled separately by `OAUTH2_OUTBOUND_ACTIVE`. See [Configuration](configuration.md#authentication).
