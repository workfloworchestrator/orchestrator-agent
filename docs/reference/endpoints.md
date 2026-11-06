# Endpoints and protocols

The agent is one HTTP service with four protocol interfaces. Their capabilities, identity forwarding and
conversation handling differ as described below.

| Client | Protocol | Endpoint |
| --- | --- | --- |
| LibreChat and other OpenAI-compatible clients | OpenAI chat completions | `POST /v1/chat/completions`, `GET /v1/models` |
| Other agents | A2A (protocol v1.0) | `POST /`, agent card at `GET /.well-known/agent-card.json` |
| Web frontends | AG-UI | `POST /agui/` |
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

`POST /agui/` streams events to a frontend. `/agui` redirects to this trailing-slash route. `runId` must be a UUID;
`threadId` selects persisted agent state. Conversation messages come from `RunAgentInput.messages`: the endpoint
does not replay a stored message history merely because the thread id is reused.

Artifact-bearing `TOOL_CALL_RESULT` events contain a JSON-encoded reference:

| Result | Reference fields, in addition to `description` |
| --- | --- |
| Search or aggregation | `query_id`, `total_results`, `visualization_type`, `search_type`, optional `rendered_block` |
| Entity lookup | `entity_id`, `entity_type` |
| Export | `query_id`, `download_url` |

Query rows can be fetched from Core's `GET /api/search/queries/{id}/results` endpoint with the caller's Core
credentials. A `rendered_block` contains a `type` (`markdown` or `mermaid`) and `content`; this frontend is responsible
for displaying it. Unlike the A2A and chat adapters, the AG-UI adapter does not append these blocks to the final text.
An export URL may be a relative path on Core and must be resolved against Core's base URL, not the agent's.

Frontend tools supplied in `RunAgentInput.tools` are passed to the underlying Pydantic AI adapter. They are not
filtered out by this implementation. The endpoint has no workflow form-fill skill or built-in approval path;
the model can also call Core's write tools. See [Read tools and write tools](runtime.md#read-tools-and-write-tools).

The endpoint uses the AG-UI adapter of pydantic-ai, so what it supports is what that adapter supports. See
[AG-UI in pydantic-ai](https://pydantic.dev/docs/ai/integrations/ui/ag-ui/) for how messages, tool calls and state
map to AG-UI events. The agent pins `ag-ui-protocol` 0.1.10, which streams reasoning as `THINKING_*` events.

For the protocol itself, see the [AG-UI documentation](https://docs.ag-ui.com/).

## MCP

`/mcp` exposes the agent's own tools to MCP clients. It is a `FastMCP` server from the
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), over stateless streamable HTTP. See
[Use the agent with MCP](../guides/mcp-clients.md).

It serves tools only: one per advertised [plugin](plugins.md) plus `ask`. It registers no resources or prompts, and
workflow plugins are not available here. Each call is independent. A previous search is not remembered for a later
export call; callers must carry the relevant query information, or request the search and export in one call.

Responses contain the model's answer and the raw JSON of artifact-bearing tool results. They do not receive the
table/chart text appended by the A2A and chat adapters.

For the protocol itself, see the [MCP documentation](https://modelcontextprotocol.io/).

## Authentication

With `OAUTH2_ACTIVE=true` (the default) incoming requests are authenticated with OIDC. With
`OAUTH2_ACTIVE=false` the agent accepts unauthenticated requests. Calls from the agent to Orchestrator Core
are controlled separately by `OAUTH2_OUTBOUND_ACTIVE`. See [Configuration](configuration.md#authentication) for the
settings and [Authentication and authorization](../architecture/auth.md) for how the checks fit together.

### Public paths

`GET /health` and `GET /.well-known/agent-card.json` never require a token.

### Responses

| Status | When |
| --- | --- |
| 401 | No bearer token was sent |
| 403 | The identity provider's userinfo endpoint rejected the token, or OPA denied the request. A denial includes the OPA decision id |
| 503 | OIDC discovery returned a non-200 response, an OPA transport error occurred, or `OPA_URL` is not set while authorization is on |

These are the handled authentication and authorization responses, not an exhaustive error mapping. Unhandled errors,
including an OIDC network failure or a malformed upstream response, can result in 500.

### OPA input

With `OAUTH2_AUTHORIZATION_ACTIVE=true` the agent posts this to `OPA_URL` for every request, and continues when
`result` is `true`:

```json
{
  "input": {
    "...": "the claims returned by the userinfo endpoint",
    "resource": "/v1/chat/completions",
    "method": "POST",
    "arguments": { "path": {}, "query": {}, "json": { "...": "the request body" } }
  }
}
```

An example policy that admits only members of one group, assuming the identity provider returns a `groups` claim:

```rego
package agent

default allow := false

allow if "network-operators" in input.groups
```

### User token forwarded to Orchestrator Core

| Endpoint | Token read from | When absent |
| --- | --- | --- |
| `POST /v1/chat/completions` | `Authorization` header | service account |
| `POST /agui/` | `Authorization` header | service account |
| A2A `POST /` | `auth_token` or `authToken` in the message metadata | service account |
| `/mcp` | not forwarded | always the service account |

The service account uses the OAuth2 client-credentials grant against `OAUTH2_TOKEN_URL`. Its token is cached and
refreshed before its reported expiry and once after a Core 401 or 403. A forwarded user token is not refreshed by the
agent. `OAUTH2_OUTBOUND_ACTIVE` controls service-account authentication; a token bound to a run is forwarded as-is.
When service-account authentication is disabled and there is no forwarded token, the request carries no bearer token.
