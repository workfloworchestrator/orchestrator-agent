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
# Install demo dependencies (included in the default dev group)
uv sync

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
| `WFO_CORE_MCP_URL` | `http://localhost:8080/mcp` | URL of orchestrator-core's MCP server (serves the domain tools the agent calls) |
| `BASE_URL` | `http://localhost:8080` | Public URL of this agent service |
| `AGENT_MODEL` | `openai:gpt-4o` | LLM model in `provider:model` format |
| `AGENT_API_BASE` | *(none)* | Custom base URL for the LLM provider (OpenAI-compatible) or Azure endpoint |
| `AGENT_API_KEY` | *(none)* | API key for the LLM provider |
| `AGENT_API_VERSION` | *(none)* | API version for Azure OpenAI (e.g. `2024-12-01-preview`) |
| `AGENT_DOMAIN_CONTEXT` | *(empty)* | Optional free-text domain knowledge appended to the agent system prompt (e.g. identifier conventions and their filter fields). Empty disables the section |
| `OAUTH2_ACTIVE` | `true` | Enable OIDC authentication on incoming requests (via `oauth2_lib`) |
| `OAUTH2_OUTBOUND_ACTIVE` | *(unset)* | Enable OAuth2 client-credentials auth on outgoing requests to orchestrator-core. When unset, follows `OAUTH2_ACTIVE`; set to `true`/`false` to control outbound auth independently of incoming auth |
| `OIDC_BASE_URL` | *(none)* | Base URL of the OIDC provider (required when `OAUTH2_ACTIVE=true`) |
| `OIDC_CONF_URL` | *(none)* | OIDC discovery document URL (required when `OAUTH2_ACTIVE=true`) |
| `OAUTH2_RESOURCE_SERVER_ID` | *(none)* | OAuth2 client ID / resource server ID (required when `OAUTH2_ACTIVE=true`) |
| `OAUTH2_RESOURCE_SERVER_SECRET` | *(none)* | OAuth2 client secret / resource server secret (required when `OAUTH2_ACTIVE=true`) |
| `OAUTH2_TOKEN_URL` | *(none)* | OAuth2 token endpoint for outgoing client-credentials requests |
| `LANGFUSE_ENABLED` | `false` | Enable Langfuse OpenTelemetry tracing. Requires the `langfuse` extra and the `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` environment variables |

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

This text is injected verbatim as a `## Domain Knowledge` section appended to the agent system prompt (so every capability sees it); leaving it empty omits the section entirely.

## Architecture

This repo is a thin MCP client. The agent is a plain pydantic-ai `Agent` configured with **capabilities** (`capabilities/`). The full orchestrator-core MCP toolset (`WFO_CORE_MCP_URL`) is passed to the `Agent`; each plugin **declares the tools it owns** via `tools:`. When a plugin is deferred, the cross-cutting `DeferredToolGate` hides its owned tools (and `defer_loading` hides its instructions) until the model calls `load_capability` — so the model can't bypass a multi-step skill by calling its tool without the guidance. Tools owned by no plugin (the filtering tools, any new server tool) are always available, so they appear automatically. The search infrastructure (query engine, filters, retrievers, indexing, DB models) lives in `orchestrator-core`.

```
Request ──► AG-UI adapter ──► async with Agent: ──► run_stream_events()
         ──► A2A adapter  ──►   (full MCP toolset; gate hides deferred plugins' tools)
         ──► MCP adapter  ──►   └─► MCPToolset ──► orchestrator-core /mcp
```

Capabilities are always-on (`defer_loading=False`). The domain capabilities — **search**, **aggregate**, **entity** (details/lookup), **export** — are **plugins**: authored Markdown files loaded at startup (see [Plugins](#plugins) below). Each plugin becomes one pydantic-ai capability that bundles its instructions and, when it declares `artifact:`, the matching behaviour — mapping *its* tools' JSON results into `QueryArtifact` / `DataArtifact` / `ExportArtifact` metadata for the AG-UI/A2A transport (`capabilities/behavior/`). Two hooks remain cross-cutting (single invariants across all tools, not plugin-owned): `FilterPathGuard` enforces that any `filters`/`group_by` call is preceded by `discover_filter_paths` (paths are DB-specific and must not be guessed), and `ProcessHistory` does sliding-window history trimming. Grouped aggregations and search results are rendered deterministically in code (a Mermaid chart / Markdown table carried as a `RenderedBlock`) and injected into the answer, so they appear even on text-only clients. MCP tools require an open session, so every run happens inside `async with agent:`.

### Plugins

Each domain capability is a **plugin** — a single Markdown file with YAML frontmatter under
`capabilities/plugins/`:

```markdown
---
id: search
description: Find subscriptions, products, workflows…
a2a_tags: [search, query, fuzzy, semantic]
examples: [Find all active subscriptions]
defer_loading: false          # required: always-on (false) or load on demand (true)
tools: [SEARCH_TOOL]          # tools this plugin OWNS (constants from tool_names.py)
artifact: query               # map results to an artifact (query/data/export); omit = instructions-only
---
# Searching
…determine the entity_type, then run the search…
```

The body **is** the prompt, used verbatim — no template language, no substitution, no includes. How
to *use* the tools (filtering, operator choice, discovery order) lives in the MCP tool descriptions
on orchestrator-core, not here.

- **Prompts describe intent and do not name MCP tools.** The model binds "run the search" → the
  `search` tool from the tool's own description; the plugin owns exactly the action tool it needs, so
  the choice is unambiguous, and `FilterPathGuard` enforces the discover-before-filter order. This is
  the "fat tools, thin prompts" model — verified live (a search query still calls
  `discover_filter_paths`→`search`). It keeps prompts free of tool names without any substitution
  machinery.
- A plugin **owns the tools it lists** in `tools:` (constants from `tool_names.py`, resolved to live
  names by `owned_tool_names`). Ownership is *not* about prompt references — it drives artifact
  mapping (a tool's results → the declared `artifact:`) and `DeferredToolGate` (those tools hide with
  the plugin when deferred). A typo'd constant fails loud at startup. The constants are verified
  against the live MCP server at startup (`verify_tool_contract`), so the code and the server can't
  drift apart silently.

The agent-level **system prompt** is **not a plugin** — so it lives at
`capabilities/system_prompt.md`, beside `plugins/` rather than inside it; the operator's
`AGENT_DOMAIN_CONTEXT` is appended to it (so it reaches every capability, not just search).
Files prefixed `_` are never loaded as plugins.

**Adding a plugin:** drop a `<id>.md` file with frontmatter + body into the built-in `plugins/`
directory and restart — a fork, like any other behaviour change. Each plugin projects to an A2A
`AgentSkill` (when `advertise: true`) and to one pydantic-ai capability.

**Artifact behaviour is declared, not coded.** A plugin maps its tool results to a rich artifact by
declaring `artifact: query` (or `data`/`export`) in frontmatter — the values are the `ArtifactType`
enum, so a bad one fails at load. The loader binds it to a shared **builder function**
(`ARTIFACT_BUILDERS` in `capabilities/behavior/`), carried by one `PluginCapability`. So a new plugin
that returns a standard result needs **no code** — it can declare `artifact: query` and get the
table/chart mapping for free. A plugin with no
`artifact:` is instructions-only. `PluginCapability` handles ownership filtering (by the plugin's
`tools:`) and artifact attachment; the builders (`query_artifact`/`data_artifact`/`export_artifact`)
are the reusable, testable units. Cross-cutting hooks (`FilterPathGuard`, history trimming, the
`DeferredToolGate`) are *not* plugins — they live in `capabilities/hooks.py` because they're single
invariants across all tools.

*(A genuinely new artifact type with bespoke rendering — e.g. a custom Mermaid graph for a future
tool — adds a value to `ArtifactType` and a builder in `ARTIFACT_BUILDERS`, or eventually ships as
co-located plugin code; the declarative binding covers reuse of the standard types.)*

**`defer_loading`.** Frontmatter `defer_loading: true` makes a capability load on demand: its
instructions stay hidden (pydantic-ai's `load_capability`) **and** `DeferredToolGate` hides its owned
tools until the model loads it — so the tool and its guidance are revealed together by one
`load_capability`, and the model can't call the tool without the instructions (verified live: a
deferred `search` makes the model `load_capability`→`discover_filter_paths`→`search` instead of
guessing). pydantic-ai's defer alone hides only instructions, not toolset tools — the gate closes
that gap, keying on the same `tools:` ownership. All built-ins are `false` (always-on). The seam
exists for when the capability set grows large enough that on-demand loading is worth the routing.

The **A2A adapter** uses [a2a-sdk](https://github.com/google/a2a-sdk) server primitives (`AgentExecutor`, `DefaultRequestHandler`, `A2AFastAPIApplication`). The SDK handles JSON-RPC routing, SSE streaming, task lifecycle, and agent card serving. The adapter implements a single `WFOAgentExecutor.execute()` method that drives the pydantic-ai event stream and publishes A2A events via `TaskUpdater`. The `AgentCard.skills` list is projected from the advertised capability specs (`skills_from_specs`), keeping the advertised skills in sync with the configured capabilities.
