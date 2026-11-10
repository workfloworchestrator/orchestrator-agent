# Endpoints and protocols

The agent is one HTTP service with four protocol interfaces. Their capabilities, identity forwarding and
conversation handling differ as described below.

| Client | Protocol | Endpoint |
| --- | --- | --- |
| LibreChat and other OpenAI-compatible clients | OpenAI chat completions | `POST /v1/chat/completions`, `GET /v1/models` |
| Other agents | A2A (protocol v1.0) | `POST /`, agent card at `GET /.well-known/agent-card.json` |
| Web frontends | AG-UI | `POST /agui/`, chat management under `/chats/` |
| MCP clients | MCP | `/mcp` |
| Monitoring | REST | `GET /health/`; `/health` redirects here |

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
- Export preparation can succeed, but this response does not carry the export's download metadata. See
  [Export delivery](#export-delivery).

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
`threadId` selects the stored conversation. Send the new turn in `RunAgentInput.messages`: the agent replays its
stored message history and records the run's messages for later turns. Client-supplied `state` is replaced by
the server's stored state.

With authentication enabled, a new thread belongs to the caller's userinfo `sub`. Missing subjects receive 401;
continuing another owner's thread, or a thread from A2A/chat completions, receives 403. With authentication off,
threads are unowned and shared. See [Conversation persistence](runtime.md#conversation-persistence).

Artifact-bearing `TOOL_CALL_RESULT` events contain a JSON-encoded reference:

| Result | Reference fields, in addition to `description` |
| --- | --- |
| Search or aggregation | `query_id`, `total_results`, `visualization_type`, `search_type`, optional `rendered_block` and `chart` |
| Entity lookup | `entity_id`, `entity_type` |
| Export | `query_id`, `download_url` |

Query rows can be fetched from Core's `GET /api/search/queries/{id}/results` endpoint with the caller's Core
credentials. A `rendered_block` contains a `type` (`markdown` or `mermaid`) and `content`; this frontend is responsible
for displaying it. Unlike the A2A and chat adapters, the AG-UI adapter does not append these blocks to the final text.
An aggregate can also carry `chart` data: `kind` (`pie`, `bar` or `line`), `title`, `metric` and `points` containing
`label`/`value` pairs. A frontend can draw a chart from that data instead of rendering Mermaid.

An AG-UI export's `download_url` is `exports/{query_id}`, relative to `/chats/{thread_id}/`. The agent's chats API
fetches the file from Core with the caller's bearer token, or the configured service fallback, and returns the CSV.

Frontend tools supplied in `RunAgentInput.tools` are passed to the underlying Pydantic AI adapter. They are not
filtered out by this implementation. The shared agent's write tool gate hides Core's three named workflow mutations;
it does not impose an approval policy on every custom frontend tool. See
[Read tools and write tools](runtime.md#read-tools-and-write-tools).

### Workflow interrupts

The endpoint shares the form-fill skill used by A2A and chat completions. To receive forms, send a
`protocolVersion` that supports interrupts: the agent accepts version 0.1.19 or later. Without that declaration,
the form handoff reports that the client cannot show a form; Core's three mutations remain hidden from the model.

A form stop ends the run with `RUN_FINISHED` and an interrupt outcome. Its interrupt has a reason, a response
schema and metadata describing the questions or the start to approve. The next run uses the same `threadId` and
sends a `resume` entry with the matching `interruptId`, `status: "resolved"` and one of these payloads:

| Interrupt reason | Resume `payload` |
| --- | --- |
| `ask_user` | `{"answers": [[0], ["typed answer"]]}`: one list per question, containing selected choice positions or typed text |
| `tool_call` | `{"approved": true}` to start, or `{"approved": false}` to cancel; the value must be a JSON boolean |

A resume entry with `status: "cancelled"` cancels the form. An unreadable response to a pending stop shows it
again. Sending a new chat message without a resume entry ends the open form and sends the message to the model.
Form reply text is rendered as Markdown; the interrupt also carries the structured form data.

The endpoint uses the AG-UI adapter of pydantic-ai, so what it supports is what that adapter supports. See
[AG-UI in pydantic-ai](https://pydantic.dev/docs/ai/integrations/ui/ag-ui/) for how messages, tool calls and state
map to AG-UI events. The agent requires `ag-ui-protocol>=1.0.0,<2` and locks version 1.0.0 on `main`.
That protocol uses `REASONING_*` events for reasoning.

The agent's `/openapi.json` includes the artifact, interrupt and resume-payload models as component schemas,
alongside the route schemas. It is protected by the same authentication middleware as the other non-public paths.

### Chats API

These routes serve AG-UI threads only. They check ownership using the authenticated subject; a thread from another
protocol is not served here. With authentication off they serve the unowned AG-UI threads.

| Route | Result |
| --- | --- |
| `GET /chats/?limit=50` | The caller's chats, newest first; `limit` is 1–200 |
| `GET /chats/{thread_id}?last=50` | Chat summary, the last messages as AG-UI messages, and any pending interrupt; `last` is 1–500 |
| `GET /chats/{thread_id}/options/{field}?q=...&limit=20` | Options for a searchable field in the pending form; `q` is at most 200 characters and `limit` is 1–50 |
| `GET /chats/{thread_id}/exports/{query_id}` | A CSV export prepared by this chat, fetched from Core with the selected bearer token |
| `DELETE /chats/{thread_id}` | Delete the chat's snapshots and run records; returns 204 |

Reading another owner's chat returns 403; a missing chat or a thread from another protocol returns 404.
Option searches require a matching searchable question in the pending stop. Export downloads require an export
recorded in that chat; a missing export returns 404, and other upstream export failures return 502.

For the protocol itself, see the [AG-UI documentation](https://docs.ag-ui.com/).

## MCP

`/mcp` exposes the agent's own tools to MCP clients. It is a `FastMCP` server from the
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), over stateless streamable HTTP. See
[Use the agent with MCP](../guides/mcp-clients.md).

It serves tools only: one per advertised [plugin](plugins.md) plus `ask`. It registers no resources or prompts, and
workflow plugins are not available here. Each call is independent. A previous search is not remembered for a later
export call; callers must carry the relevant query information, or request the search and export in one call.

Responses contain the model's answer and the raw JSON of artifact-bearing tool results. They do not receive the
table/chart text appended by the A2A and chat adapters. For exports this is the model-visible payload after the
plugin removes Core's standard link fields; see [Export delivery](#export-delivery).

For the protocol itself, see the [MCP documentation](https://modelcontextprotocol.io/).

## Export delivery

AG-UI carries `ExportArtifact.download_url` and provides the authenticated chat export proxy. The export plugin
strips Core's standard `download_path` and `message` from the model-visible tool result. Chat completions do not
deliver export metadata; A2A's data artifacts and MCP's result JSON use that stripped tool payload rather than
the `ExportArtifact` metadata. Preparing an export through those interfaces therefore does not provide the
standard download URL. The built-in download flow currently requires AG-UI.

## Health check

`GET /health/` runs `SELECT 1` against the configured database. It returns 200 on success and 503 on a database
operational error. It does not test the model, Core MCP, the identity provider or OPA. `GET /health` redirects here.

## Authentication

With `OAUTH2_ACTIVE=true` (the default) incoming requests are authenticated with OIDC. With
`OAUTH2_ACTIVE=false` the agent accepts unauthenticated requests. Calls from the agent to Orchestrator Core
are controlled separately by `OAUTH2_OUTBOUND_ACTIVE`. See [Configuration](configuration.md#authentication) for the
settings and [Authentication and authorization](../explanation/auth.md) for how the checks fit together.

### Public paths

`GET /health` and `GET /.well-known/agent-card.json` never require a token.

### Responses

| Status | When |
| --- | --- |
| 401 | No bearer token was sent, or an AG-UI/chats request has no authenticated subject |
| 403 | The identity provider's userinfo endpoint rejected the token, or OPA denied the request. A denial includes the OPA decision id |
| 503 | OIDC discovery returned a non-200 response, an OPA transport error occurred, or `OPA_URL` is not set while authorization is on |

These are the handled authentication and authorization responses, not an exhaustive error mapping. Unhandled errors,
including an OIDC network failure or a malformed upstream response, can result in 500.

### OPA input

With both incoming authentication and authorization enabled, the agent posts this to `OPA_URL` for each
authenticated request outside the public paths, and continues when
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
| Chat option searches and export downloads | `Authorization` header | service account |
| A2A `POST /` | `auth_token` or `authToken` in the message metadata | service account |
| `/mcp` | not forwarded | service account when enabled; otherwise no bearer token |

The service account uses the OAuth2 client-credentials grant against `OAUTH2_TOKEN_URL`. Its token is cached and
refreshed before its reported expiry and once after a Core 401 or 403. A forwarded user token is not refreshed by the
agent. `OAUTH2_OUTBOUND_ACTIVE` controls service-account authentication; a token bound to a run is forwarded as-is.
When service-account authentication is disabled and there is no forwarded token, the request carries no bearer token.
