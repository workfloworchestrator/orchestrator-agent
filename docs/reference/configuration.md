---
render_macros: false
---

# Configuration

The agent is configured through environment variables. Copy `.env.example` from the repository to `.env` and edit it.

## Connection

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URI` | *(required)* | PostgreSQL connection URI for the WFO database. A bare `postgresql://` scheme is coerced to `postgresql+psycopg://` |
| `WFO_CORE_MCP_URL` | `http://localhost:8080/mcp` | URL of the Orchestrator Core MCP server that serves the domain tools the agent calls |
| `BASE_URL` | `http://localhost:8080` | Public URL of the agent, advertised in the A2A agent card |

## Model

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_MODEL` | `openai:gpt-4o` | LLM in `provider:model` format |
| `AGENT_API_BASE` | *(none)* | Base URL of an OpenAI-compatible provider, or the Azure endpoint |
| `AGENT_API_KEY` | *(none)* | API key for the provider |
| `AGENT_API_VERSION` | *(none)* | API version for Azure OpenAI, for example `2024-12-01-preview` |

See [Use your own LLM](../guides/own-llm.md) for examples.

## Domain context

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_DOMAIN_CONTEXT` | *(empty)* | Free text appended to the system prompt, for example identifier conventions and the fields they map to. Empty disables the section |

See [Teach the agent your domain](../guides/domain-context.md).

## Authentication

| Variable | Default | Description |
| --- | --- | --- |
| `OAUTH2_ACTIVE` | `true` | Authenticate incoming requests with OIDC |
| `OAUTH2_OUTBOUND_ACTIVE` | *(unset)* | Authenticate outgoing requests to Orchestrator Core with OAuth2 client credentials. When unset it follows `OAUTH2_ACTIVE` |
| `OIDC_BASE_URL` | *(none)* | Base URL of the OIDC provider. Required when `OAUTH2_ACTIVE=true` |
| `OIDC_CONF_URL` | *(none)* | OIDC discovery document URL. Required when `OAUTH2_ACTIVE=true` |
| `OAUTH2_RESOURCE_SERVER_ID` | *(none)* | OAuth2 client id. Required when `OAUTH2_ACTIVE=true` |
| `OAUTH2_RESOURCE_SERVER_SECRET` | *(none)* | OAuth2 client secret. Required when `OAUTH2_ACTIVE=true` |
| `OAUTH2_TOKEN_URL` | *(none)* | Token endpoint for the outgoing client-credentials requests |

## Observability

| Variable | Default | Description |
| --- | --- | --- |
| `LANGFUSE_ENABLED` | `false` | Send OpenTelemetry traces to Langfuse. Needs the `langfuse` extra and `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` |

See [Security and observability](../guides/security-observability.md).

## LibreChat

The keys of `librechat.yaml` that the agent relies on. The full example is `examples/librechat.yaml` in the
repository, and [Connect LibreChat](../guides/librechat.md) walks through it.

| Key | Value | Why |
| --- | --- | --- |
| `endpoints.custom[].baseURL` | `http://<agent>:8080/v1` | The agent's OpenAI-compatible endpoint |
| `endpoints.custom[].models.default` | `['wfo']` | The one model id the agent serves |
| `endpoints.custom[].titleConvo` | `false` | Otherwise LibreChat asks the agent for a chat title before the first answer |
| `endpoints.custom[].headers.X-Conversation-Id` | `'{% raw %}{{LIBRECHAT_BODY_CONVERSATIONID}}{% endraw %}'` | The agent keeps a conversation's memory and its open form under this id |
| `endpoints.custom[].headers.X-Agent-Client` | `'librechat'` | Required: the agent opens a workflow form only for a caller that identifies itself as LibreChat |
| `endpoints.custom[].headers.Authorization` | `'Bearer {% raw %}{{LIBRECHAT_OPENID_ACCESS_TOKEN}}{% endraw %}'` | Optional: forwards the signed-in user's token when the agent runs with `OAUTH2_ACTIVE=true` |
| `modelSpecs.list[].askUserQuestion` | `true` | Offers the ask-user tool to the agent. Without it no workflow form is opened. Needs LibreChat 0.8.8 or later |
