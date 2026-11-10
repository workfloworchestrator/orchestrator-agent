---
render_macros: false
---

# Configuration

The agent reads process environment variables at import time. The application does not automatically load `.env`.
`uv run --env-file .env ...` loads it for a source launch; Docker Compose's `env_file` passes it to a container.
See [Run the agent with your Orchestrator Core](../getting-started.md) for setup.

## Connection

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URI` | *(required)* | PostgreSQL connection URI for the WFO database. Prefer the explicit `postgresql+psycopg://` scheme. A bare `postgresql://` also selects psycopg with the SQLAlchemy 2.1 version locked on `main` |
| `WFO_CORE_MCP_URL` | `http://localhost:8080/mcp` | URL of the Orchestrator Core MCP server that serves the domain tools the agent calls |
| `WFO_CORE_API_URL` | *(derived)* | Base URL for a form field's relative `optionsSource.url` and for Core export downloads. When unset, uses the origin of `WFO_CORE_MCP_URL` |
| `WFO_OPTIONS_HOSTS` | `[]` | Additional hosts allowed for absolute form-option URLs. Supply a JSON list, for example `["inventory.example.com:8443"]`. Entries match the URL's host and port exactly; allowed destinations receive the run's bearer token or fallback service token |
| `BASE_URL` | `http://localhost:8080` | Public URL of the agent, advertised in the A2A agent card |

## Model

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_MODEL` | `openai:gpt-4o` | LLM in `provider:model` format |
| `AGENT_API_BASE` | *(none)* | Base URL of an OpenAI-compatible provider, or the Azure endpoint |
| `AGENT_API_KEY` | *(none)* | API key for the provider |
| `AGENT_API_VERSION` | *(none)* | API version for Azure OpenAI, for example `2024-12-01-preview` |

When `AGENT_API_BASE`, `AGENT_API_KEY` and `AGENT_API_VERSION` are all unset, the provider reads its own environment
variable for the key, such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

When one of them is set, the `azure:` prefix in `AGENT_MODEL`, or a set `AGENT_API_VERSION`, selects the Azure OpenAI
provider. Otherwise the custom path always constructs an OpenAI chat model, even if `AGENT_MODEL` names another
provider. An empty environment value is still set; only unset values select the native provider path.

See [How to change the language model](../guides/own-llm.md) for examples and provider dependencies.

## Domain context

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_DOMAIN_CONTEXT` | *(empty)* | Free text appended to the system prompt, for example identifier conventions and the fields they map to. Empty disables the section |

See [Add domain context](../guides/domain-context.md).

## Authentication

| Variable | Default | Description |
| --- | --- | --- |
| `OAUTH2_ACTIVE` | `true` | Authenticate incoming requests with OIDC |
| `OAUTH2_AUTHORIZATION_ACTIVE` | `true` | Authorize authenticated incoming requests with an Open Policy Agent (OPA). Only applies when `OAUTH2_ACTIVE=true`. Set to `false` if you do not run OPA |
| `OPA_URL` | *(none)* | OPA decision endpoint. Required when `OAUTH2_ACTIVE=true` and `OAUTH2_AUTHORIZATION_ACTIVE=true`; without it protected requests fail with 503 |
| `OAUTH2_OUTBOUND_ACTIVE` | *(unset)* | Enable service-account client-credentials authentication to Core. When unset it follows `OAUTH2_ACTIVE`; it does not suppress forwarding a user token bound to a run |
| `OIDC_BASE_URL` | *(none)* | Provider base URL accepted by `oauth2_lib`; this agent resolves discovery and userinfo through `OIDC_CONF_URL` |
| `OIDC_CONF_URL` | *(none)* | OIDC discovery document URL. Required when `OAUTH2_ACTIVE=true` |
| `OAUTH2_RESOURCE_SERVER_ID` | *(none)* | Client id used for the outbound service account's client-credentials grant |
| `OAUTH2_RESOURCE_SERVER_SECRET` | *(none)* | Client secret used for the outbound service account's client-credentials grant; inbound userinfo lookup does not use these client credentials |
| `OAUTH2_TOKEN_URL` | *(none)* | Token endpoint for the outgoing client-credentials requests. Required when outgoing authentication is on |

## Observability

| Variable | Default | Description |
| --- | --- | --- |
| `LANGFUSE_ENABLED` | `false` | Send OpenTelemetry traces to Langfuse. Needs the `langfuse` extra and `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` |

See [How to configure authentication and authorization](../guides/secure-agent.md) and [How to trace the agent](../guides/tracing.md).

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
| `endpoints.custom[].headers.Authorization` | `'Bearer {% raw %}{{LIBRECHAT_OPENID_ACCESS_TOKEN}}{% endraw %}'` | Forwards the signed-in user's token to Core, independently of the inbound and service-account authentication toggles. Required to pass the agent's inbound check when `OAUTH2_ACTIVE=true` |
| `modelSpecs.list[].askUserQuestion` | `true` | Offers the ask-user tool to the agent. Without it no workflow form is opened. Needs LibreChat 0.8.8 or later |
