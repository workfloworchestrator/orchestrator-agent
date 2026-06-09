# orchestrator-agent

[![Container](https://ghcr-badge.egpl.dev/workfloworchestrator/orchestrator-agent/latest_tag?trim=major&label=container)](https://github.com/workfloworchestrator/orchestrator-agent/pkgs/container/orchestrator-agent)

Standalone WFO search agent for deployment. Exposes the orchestration search agent via AG-UI, A2A, and MCP protocols.

## Quick Start

```bash
cp .env.example .env
# Edit .env with your DATABASE_URI and LLM settings

uv sync
uv run uvicorn orchestrator_agent.app:app --port 8080
```

## Endpoints

| Path | Protocol | Description |
| --- | --- | --- |
| `POST /agui` | AG-UI | SSE streaming for frontend |
| `POST /` | A2A | Agent-to-agent JSON-RPC (`message/send`, `message/stream`) |
| `GET /.well-known/agent.json` | A2A | Agent card discovery |
| `GET /.well-known/agent-card.json` | A2A | Agent card discovery (alias) |
| `/mcp` | MCP | Model Context Protocol tools |
| `GET /health` | REST | Health check |

## Docker

```bash
docker compose up --build
```

## Demos

```bash
# Install demo dependencies
uv sync --group demo

# AG-UI: stream a search query
uv run demos/agui_client.py "find active subscriptions"

# AG-UI: follow-up on the same thread
uv run demos/agui_client.py "export them" <thread-id>

# MCP: run all smoke tests (search, aggregate, ask)
uv run demos/mcp_client.py

# MCP: single tool
uv run demos/mcp_client.py search "active subscriptions"

# A2A: delegate to the agent via CrewAI
uv run demos/a2a_client.py <subscription-uuid>
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URI` | *(required)* | PostgreSQL connection URI for the WFO database |
| `ORCHESTRATOR_API_URL` | `http://localhost:8080` | URL of the orchestrator-core API |
| `WFO_CORE_MCP_URL` | `http://localhost:8080/mcp` | URL of orchestrator-core's MCP server (serves the domain tools the agent calls) |
| `BASE_URL` | `http://localhost:8080` | Public URL of this agent service |
| `AGENT_MODEL` | `openai:gpt-4o` | LLM model in `provider:model` format |
| `AGENT_API_BASE` | *(none)* | Custom base URL for the LLM provider (OpenAI-compatible) or Azure endpoint |
| `AGENT_API_KEY` | *(none)* | API key for the LLM provider |
| `AGENT_API_VERSION` | *(none)* | API version for Azure OpenAI (e.g. `2024-12-01-preview`) |
| `AGENT_DEBUG` | `false` | Enable debug logging for agent execution |
| `SEARCH_RESULT_LIMIT` | `10` | Default maximum number of results a search returns when the model doesn't request a specific count. The model can still request more per query (e.g. "top 50") |
| `AGENT_SEARCH_EFFORT` | `medium` | How hard the agent tries before deferring to you. `high` = up to two broadening fallback searches when a filtered search returns nothing (most persistent); `medium` = one broadening pass, then report no matches; `low` = no silent broadening and the planner prefers asking a clarifying question |
| `AGENT_DOMAIN_CONTEXT` | *(empty)* | Optional free-text domain knowledge injected into the search prompt (e.g. identifier conventions and their filter fields). Empty disables the section |
| `OAUTH2_ACTIVE` | `true` | Enable OIDC authentication on incoming requests (via `oauth2_lib`) |
| `OAUTH2_OUTBOUND_ACTIVE` | *(unset)* | Enable OAuth2 client-credentials auth on outgoing requests to orchestrator-core. When unset, follows `OAUTH2_ACTIVE`; set to `true`/`false` to control outbound auth independently of incoming auth |
| `OIDC_BASE_URL` | *(none)* | Base URL of the OIDC provider (required when `OAUTH2_ACTIVE=true`) |
| `OIDC_CONF_URL` | *(none)* | OIDC discovery document URL (required when `OAUTH2_ACTIVE=true`) |
| `OAUTH2_RESOURCE_SERVER_ID` | *(none)* | OAuth2 client ID / resource server ID (required when `OAUTH2_ACTIVE=true`) |
| `OAUTH2_RESOURCE_SERVER_SECRET` | *(none)* | OAuth2 client secret / resource server secret (required when `OAUTH2_ACTIVE=true`) |
| `OAUTH2_TOKEN_URL` | *(none)* | OAuth2 token endpoint for outgoing client-credentials requests |

### Custom LLM endpoint

By default the agent uses the standard OpenAI API with the `OPENAI_API_KEY` environment variable. To use a custom OpenAI-compatible endpoint, set `AGENT_API_BASE` and/or `AGENT_API_KEY`:

```bash
# Local Ollama
AGENT_MODEL=openai:llama3
AGENT_API_BASE=http://localhost:11434/v1

# LiteLLM proxy or other OpenAI-compatible endpoint
AGENT_MODEL=openai:gpt-4o
AGENT_API_BASE=https://my-proxy.example.com/v1
AGENT_API_KEY=sk-custom-key

# Azure OpenAI
AGENT_MODEL=azure:gpt-4o
AGENT_API_BASE=https://my-resource.openai.azure.com/
AGENT_API_KEY=azure-api-key
AGENT_API_VERSION=2024-12-01-preview
```

The `azure:` prefix on `AGENT_MODEL` (or setting `AGENT_API_VERSION`) selects the Azure provider automatically. When none of `AGENT_API_BASE`, `AGENT_API_KEY`, or `AGENT_API_VERSION` is set, `AGENT_MODEL` is passed directly to pydantic-ai as a model string (existing behavior).

### Identifier-aware search

When a user references an entity by a concrete identifier — a customer name, a subscription id, or a code/number such as `IS4443`, `4433`, or `id 1234` — the search skill is guided to extract that token and use it as a high-signal search key: it discovers the matching field and filters with `like` (substring/typo-tolerant), and only falls back to plain ranking when no field clearly matches.

The agent also picks a **retriever** per query:

- **HYBRID** (semantic + fuzzy keyword) — for identifier/code/name-centric lookups.
- **SEMANTIC** — for descriptive or sentence-like queries.
- **FUZZY** — for exact tokens, and whenever embeddings are unavailable.

SEMANTIC and HYBRID require embeddings (configured in orchestrator-core via `EMBEDDING_API_ENABLED`). When embeddings are disabled, the agent automatically uses FUZZY and the prompt stops offering the embedding-based options — so the feature degrades safely with no configuration change.

Use `AGENT_DOMAIN_CONTEXT` to teach the agent deployment-specific conventions that it cannot infer, for example:

```bash
AGENT_DOMAIN_CONTEXT="Circuit codes look like IS#### and map to the imsCircuitId field — filter with like. Customer references are 8-digit numbers — field customerId."
```

This text is injected verbatim as a `## Domain Knowledge` section in the search prompt; leaving it empty omits the section entirely.

## Architecture

This repo is a thin MCP client. The agent is a plain pydantic-ai `Agent` configured with **capabilities** (`capabilities/`). The full orchestrator-core MCP toolset is passed directly to the `Agent`, so the MCP server (`WFO_CORE_MCP_URL`, gated by `AgentTag.EXPOSED`) is the single source of which tools the model can call — new tools appear automatically with no agent change. Domain capabilities contribute only instructions. The search infrastructure (query engine, filters, retrievers, indexing, DB models) lives in `orchestrator-core`.

```
Request ──► AG-UI adapter ──► async with Agent: ──► run_stream_events()
         ──► A2A adapter  ──►   (always-on capabilities + full MCP toolset)
         ──► MCP adapter  ──►   └─► MCPToolset ──► orchestrator-core /mcp
```

Capabilities are always-on (no on-demand `load_capability`). Domain capabilities — **search**, **aggregate**, **entity** (details/lookup), **export** — supply instructions only. Three hook capabilities add cross-cutting behaviour: `FilterPathGuard` enforces that any `filters`/`group_by` call is preceded by `discover_filter_paths` (paths are DB-specific and must not be guessed); `ProcessHistory` does sliding-window history trimming; and `ArtifactCapability` maps each MCP tool's JSON result into `QueryArtifact` / `DataArtifact` / `ExportArtifact` metadata for the AG-UI/A2A transport. Grouped aggregations and search results are also rendered deterministically in code (a Mermaid chart / Markdown table carried as a `RenderedBlock`) and injected into the answer, so they appear even on text-only clients. MCP tools require an open session, so every run happens inside `async with agent:`.

The **A2A adapter** uses [a2a-sdk](https://github.com/google/a2a-sdk) server primitives (`AgentExecutor`, `DefaultRequestHandler`, `A2AFastAPIApplication`). The SDK handles JSON-RPC routing, SSE streaming, task lifecycle, and agent card serving. The adapter implements a single `WFOAgentExecutor.execute()` method that drives the pydantic-ai event stream and publishes A2A events via `TaskUpdater`. The `AgentCard.skills` list is projected from the advertised capability specs (`skills_from_specs`), keeping the advertised skills in sync with the configured capabilities.
