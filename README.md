# orchestrator-agent

[![Container](https://ghcr-badge.egpl.dev/workfloworchestrator/orchestrator-agent/latest_tag?trim=major&label=container)](https://github.com/workfloworchestrator/orchestrator-agent/pkgs/container/orchestrator-agent)

Standalone WFO search agent for deployment. Exposes the orchestration search agent via AG-UI, A2A, and MCP protocols, and as an OpenAI-compatible chat model for LibreChat.

Documentation: <https://workfloworchestrator.org/orchestrator-agent/>. The sources are in [`docs/`](docs/) and are
published as a sub-project of workfloworchestrator.org; preview them with `uv run --group docs mkdocs serve`.

## Quick Start

```bash
cp .env.example .env
# Edit .env with your DATABASE_URI and LLM settings

uv sync
uv run --env-file .env uvicorn orchestrator_agent.app:app --port 8080
```

Or follow [Run the agent with your Orchestrator Core](docs/getting-started.md) to configure the database, Core MCP
connection and model before launching. Keep the agent on its default port 8080; when sharing one host, give Core
another port. The source launch explicitly loads `.env` with `uv run --env-file .env`.

## Endpoints

| Path | Protocol | Description |
| --- | --- | --- |
| `POST /agui/` | AG-UI | SSE streaming, stored conversation history and a workflow start's approval through an interrupt; supports the [chat popup](#chat-popup-ag-ui); `/agui` redirects here |
| `/chats/` | REST | AG-UI chat history, the approval a chat waits on and export downloads; see [Endpoints and protocols](docs/reference/endpoints.md#chats-api) |
| `POST /` | A2A | Agent-to-agent JSON-RPC, protocol v1.0 (`SendMessage`, `SendStreamingMessage`) |
| `GET /.well-known/agent-card.json` | A2A | Agent card discovery |
| `POST /v1/chat/completions` | OpenAI chat completions | The agent as a chat model for LibreChat; a workflow start's approval through the client's ask-user tool |
| `GET /v1/models` | OpenAI | Lists the one model id (`wfo`) |
| `/mcp` | MCP | Model Context Protocol tools |
| `GET /health/` | REST | Database health check; `/health` redirects here |

## Docker

The repository's Compose file builds the agent image, loads `.env` and uses host networking. Its command listens on
8080. Set `BASE_URL` to the address of that listener, and point `WFO_CORE_MCP_URL` at Core on a different port or host
before starting it. Services on separate hosts or container addresses can each use 8080.

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

```

## Chat popup (AG-UI)

The agent as a chat popup in the bottom-right corner of a page is the React component `WfoChat` in
[orchestrator-agent-chat](https://github.com/workfloworchestrator/orchestrator-agent-chat): it talks to the
agent's `/agui/` route with the AG-UI client, renders the agent's Markdown tables, draws its charts from the
data the agent attaches to an aggregate result, shows the tools as they run, keeps the thread across reloads
of the tab, and sends what the host page knows (the subscription on screen) as AG-UI context. Its props and
theming are in that repository's README.

`demos/web/` is a page with nothing on it but the popup, installed from that repository:

```bash
cd demos/web && npm install && AGENT_URL=http://127.0.0.1:8080 npm run dev     # http://localhost:5173
```

The page needs no CORS on the agent: its dev server forwards `/agui/` and `/chats` to it, so the browser sees
one origin. With OAuth2 on, the popup's `token` prop is the bearer the agent forwards to core.

**The contract.** The popup's types come from this agent's OpenAPI document, which also names what travels
inside AG-UI messages (`adapters/ag_ui/contract.py`): the artifact reference a tool result carries with its
chart data, the metadata of a workflow's approval and the answer that resolves it. The popup checks in a copy of the
document; when a route or one of those models changes here, refresh it there (`npm run contract:pull`
against a running agent, or write it from this checkout with
`uv run python -m orchestrator_agent.adapters.ag_ui.contract <path>`).

**Workflow forms in the popup.** A form is filled in the conversation, by the model, with no component per
field (see [Workflow form-fill](#workflow-form-fill-the-model-fills-a-person-approves)). The popup shows the
form's tool calls as one group, from the `workflow_form` activity message the agent sends, and one card: the
start to approve with the values the workflow gets. The transport is AG-UI's own interrupt lifecycle: the
model's `start_workflow` call ends the run with an interrupt outcome, the card is that interrupt, and the next
run carries its resume entry (`adapters/ag_ui/interrupts.py`). The card offers Approve, Reject and "Talk about
this"; while it is open the message box is blocked, and the agent refuses a run for that thread without the
decision (409).

**Chats live on the agent.** The popup sends one message per run; the agent replays the thread's earlier
turns from its own store (core's `graph_snapshots`, where the other routes keep their memory too) and records
the run's messages. A small API next to the route serves them back: `GET /chats/` lists the person's chats,
newest first; `GET /chats/{thread_id}?last=50` gives one with its last messages as AG-UI messages and the form
approval it waits on; `DELETE /chats/{thread_id}` forgets it; `GET /chats/{thread_id}/exports/{query_id}` serves an
export the chat prepared as a CSV file, as the person asking. The popup opens on that list (a chat whose
approval waits is marked), and opening one loads its history and shows a pending approval again. A thread belongs to the subject of the bearer that
created it: only that subject continues, reads or deletes it, and a thread written by another route (kagent,
LibreChat) is never served here. The other routes are unchanged: they keep trusting the conversation id their
client forwards. With OAuth2 off there is no subject, every thread is unowned and every caller sees them — a
development setting.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URI` | *(required)* | PostgreSQL connection URI for the WFO database |
| `WFO_CORE_MCP_URL` | `http://localhost:8080/mcp` | URL of orchestrator-core's MCP server (serves the domain tools the agent calls) |
| `WFO_CORE_API_URL` | *(derived)* | Base URL of core's REST API, where a chat's export file is fetched from. Unset: the origin of `WFO_CORE_MCP_URL` |
| `BASE_URL` | `http://localhost:8080` | Public URL of this agent service |
| `AGENT_MODEL` | `openai:gpt-4o` | LLM model in `provider:model` format |
| `AGENT_API_BASE` | *(none)* | Custom base URL for the LLM provider (OpenAI-compatible) or Azure endpoint |
| `AGENT_API_KEY` | *(none)* | API key for the LLM provider |
| `AGENT_API_VERSION` | *(none)* | API version for Azure OpenAI (e.g. `2024-12-01-preview`) |
| `AGENT_MODEL_SETTINGS` | `{}` | JSON object of model settings sent with every request, e.g. `{"temperature": 0.2, "thinking": "high"}`. See [Model settings](docs/reference/configuration.md#model-settings) |
| `AGENT_DOMAIN_CONTEXT` | *(empty)* | Optional free-text domain knowledge appended to the agent system prompt (e.g. identifier conventions and their filter fields). Empty disables the section |
| `OAUTH2_ACTIVE` | `true` | Enable OIDC authentication on incoming requests (via `oauth2_lib`) |
| `OAUTH2_AUTHORIZATION_ACTIVE` | `true` | Enable request authorization through OPA when incoming authentication is on |
| `OPA_URL` | *(none)* | OPA decision URL; required when authentication and authorization are on |
| `OAUTH2_OUTBOUND_ACTIVE` | *(unset)* | Enable OAuth2 client-credentials auth on outgoing requests to orchestrator-core. When unset, follows `OAUTH2_ACTIVE`; set to `true`/`false` to control outbound auth independently of incoming auth |
| `OIDC_BASE_URL` | *(none)* | Provider base URL accepted by `oauth2_lib`; the agent uses `OIDC_CONF_URL` for discovery |
| `OIDC_CONF_URL` | *(none)* | OIDC discovery document URL (required when `OAUTH2_ACTIVE=true`) |
| `OAUTH2_RESOURCE_SERVER_ID` | *(none)* | OAuth2 client id for the outbound service-account grant |
| `OAUTH2_RESOURCE_SERVER_SECRET` | *(none)* | OAuth2 client secret for the outbound service-account grant |
| `OAUTH2_TOKEN_URL` | *(none)* | OAuth2 token endpoint for outgoing client-credentials requests |
| `LANGFUSE_ENABLED` | `false` | Enable Langfuse OpenTelemetry tracing. Requires the `langfuse` extra and the `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` environment variables |

The application reads the process environment, not `.env` automatically. See [Configuration](docs/reference/configuration.md)
for the complete settings and [authentication setup](docs/guides/secure-agent.md) for identity forwarding and service accounts.

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

Put these settings in the environment loaded by your launch. Remove unused overrides entirely before selecting a
native provider; an empty assignment is still set. The custom path always constructs an OpenAI chat model unless
Azure is selected. Other providers can require additional SDK dependencies; see [Change the language model](docs/guides/own-llm.md).

### Identifier-aware search

Core's search-tool descriptions guide identifier lookups toward the corresponding field and a `like` filter when
appropriate. `like` is a case-insensitive wildcard match, for example `%IS4443%`; it does not itself tolerate typos.
Fuzzy retrieval provides approximate text ranking. See [Search behavior](docs/explanation/index.md#how-a-question-becomes-a-search).

The agent also picks a **retriever** per query:

- **HYBRID** (semantic + fuzzy keyword) — for identifier/code/name-centric lookups.
- **SEMANTIC** — for descriptive or sentence-like queries.
- **FUZZY** — for exact tokens, and whenever embeddings are unavailable.

SEMANTIC and HYBRID require embeddings configured in Core via `EMBEDDING_API_ENABLED`. Core's tool descriptions
describe the retrievers available in that deployment. See [Core search](https://workfloworchestrator.org/orchestrator-core/reference-docs/ai-search/#running-without-embeddings)
for fallback behavior and explicit retriever requests.

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
         ──► chat adapter ──►   (one agent with the form-fill skill behind all three)
         ──► MCP adapter  ──►   └─► MCPToolset ──► orchestrator-core /mcp
```

All built-in plugins are always on (`defer_loading=False`). **Search**, **aggregate**, **entity** and **export** are
Markdown plugins loaded at startup; **workflow** is added where the form-fill skill runs, on A2A, chat completions and AG-UI.
Each plugin contributes instructions and, when it declares `artifact:`, maps its tools' results to typed metadata
in `capabilities/behavior/`. See [Plugins](#plugins).

The cross-cutting guards cover deferred-tool visibility, discovery before filtering/grouping and history trimming.
On A2A, chat completions and AG-UI, `WriteToolGate` also hides Core's three workflow mutations so the
[form-fill skill](#workflow-form-fill-the-model-fills-a-person-approves) controls starts. This gate is absent from the agent's MCP interface.

Search tables and grouped charts are rendered in code. A2A and chat completions append them to the answer; AG-UI
carries rendered blocks and optional `ChartData` in artifact metadata, so a frontend can draw its own charts,
while MCP returns raw JSON. Each run opens an MCP session of its own to Core, as the person it runs for.
See [Endpoints and protocols](docs/reference/endpoints.md) for protocol-specific behavior.

Export downloads are delivered through AG-UI's artifact metadata and authenticated chats API. The export plugin
removes Core's standard link fields from model-visible results; chat completions, A2A and MCP currently do not
deliver that download URL. See [Export delivery](docs/reference/endpoints.md#export-delivery).

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
  the plugin when deferred). A typo'd constant fails during capability construction. Startup's
  `verify_tool_contract` checks the Core names listed in `ALL_TOOL_NAMES` against the live server; local
  tools and new constants outside that list are not covered by the check.

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

### Workflow form-fill (the model fills, a person approves)

Starting a workflow is a conversation with a hard edge. The model fills the workflow's input form with the
person, page by page; the form's state, its option lists and the start itself are code (`form_fill/`), and
nothing starts without a person's structured approval. The approval is pydantic-ai's own tool approval, which
AG-UI carries as an interrupt (the chat popup). No client needs a component per form field: the questions are
the model's own words. Chat completions carries it through the client's ask-user tool (LibreChat). A2A cannot show an
approval yet, so there a form is prepared, not started.

1. **Opens** — deciding *that* a request starts a workflow and *which* one is a judgment call, so the model
   makes it: it reads `list_workflows` and calls `start_workflow_form(workflow_key, subscription_id?)` (any
   key of core's catalogue, tasks included; a key core does not know is a tool retry). The tool opens the
   form on core's first page. A subscription id the model passes is submitted by code where the first page
   asks for one, so core's verdict on it is the first thing the model reads. One form is open at a time.
2. **Fills** — while a form is open the model has the page tools, built per model request from the form's
   state (`form_fill/tools.py`): `submit_page`, whose parameters are the current page's JSON Schema as core
   wrote it (less the frontend's hints, the fields that are only shown, and consent), `search_options` for a
   long list, `restart_from_page` to change an accepted page, and `cancel_form`. It reads what the person
   said into the page, asks in plain words for what is missing, and fixes what core rejected. Its
   instructions are rendered per request (`form_fill/instructions.py`): the rules, the pages accepted with
   their values, the current page and core's last verdict.
3. **Core judges** — the pages core accepted are exactly what `create_workflow` is called with, and core's
   validation is the only validation. A short option list is in the tool's schema (`enum`, with `options` as
   labels). A long one is found with `search_options`, which asks core's `get_workflow_form_options`; core
   rejects a submitted value that is not among the field's options, so the model cannot invent an id. What a
   page fills by itself is filled by code: consent (pydantic-forms' `Accept`: the approval at the end is the
   person's consent, before anything is written) and a required choice with one option.
4. **Asks for approval** — when core says the form is complete the model calls `start_workflow`, which takes
   no arguments. Unapproved, the call stops the run (`ApprovalRequired`) with what the person is shown: the
   `create_workflow` call as it will be made (`workflow_key`, `json_data`: exactly the accepted pages) plus
   `labels` saying what its ids stand for, rendered by code. The model never writes what the person approves.
5. **The person decides.** Approve starts. Reject closes the form, in code. A rejection with words ("Talk
   about this" in the popup) leaves the form open: the words come back as the result of the `start_workflow`
   call, the model acts on them (`restart_from_page` and the corrected page, or an answer) and calls
   `start_workflow` again.
6. **Starts** — `create_workflow` from code, once per approval, and only inside an approved `start_workflow`
   call. Only core's own verdict on the pages reopens the form (the page it rejects is current again, with
   its words); any other failure leaves it unknown whether the process started, so the form closes.

**What the model can and cannot do.** Inside a page the model is in the loop, so its quality matters there:
it may misread a sentence or pick between two options it should have asked about, and the approval is where
the person sees it. What it cannot do is structural: it has no write tool, `start_workflow` does nothing
before the person approved, core refuses a value that is not among a field's options, the model does not
write the approval, and typed words are never a start.

**A page as core describes it.** Core's form tool (`get_workflow_form`) describes each page as one JSON
Schema (`schema`): every reference inlined, so a choice or a nested model is read in place; `readOnly` on a
field that is only shown; a choice's values in `enum` with their labels in `options`; the limits a list is
validated against; consent as `format: "accept"`; each field's label and help text in `title` and
`description`. A field whose options live elsewhere (`optionsSource`, pydantic-forms ≥ 2.10) is read by core:
a short list arrives as a choice, a long one keeps its `optionsSource` and is searched with core's
`get_workflow_form_options`, and core labels the accepted values for the approval. Asked for a verdict
(`verdict="result"`), a rejected page comes back as a result with core's errors. `form_fill/model.py` reads
the schema into one `FieldSpec` per field and `form_fill/pages.py` makes of it what the model's tool takes
and what the approval shows. No model is rebuilt from the schema. Core's tool results are validated with
core's own response models and its tool arguments are built from core's request models (the agent depends on
orchestrator-core), so nothing here reads a tool error's text.

In the agent the skill runs in (A2A, chat completions and AG-UI share one agent) the model never sees
core's write tools (`WriteToolGate` hides `create_workflow` / `resume_workflow_process` /
`abort_workflow_process`), so a workflow is only ever started by this skill; the MCP agent runs without the
skill, so it gets neither the gate nor the `workflow` plugin, and its model can call core's write tools
directly.

**A2A protocol version.** The A2A endpoint speaks protocol v1.0 (a2a-sdk 1.x, protobuf types:
`SendMessage`, `SendStreamingMessage`; the agent card is at `/.well-known/agent-card.json`; a v1 client
sends `A2A-Version: 1.0`). Callers that have not moved yet are still served: the card also advertises a
0.3 interface and the SDK's compatibility layer accepts `message/send` / `message/stream`
(`enable_v0_3_compat`), as the A2A project recommends for a staged migration. Drop the flag and the 0.3 interface once
no caller needs them.

**Which conversation a call belongs to.** Memory and the open form are keyed on the chat a parent agent
forwards with every call (`x-kagent-root-context-id`, else `x-kagent-parent-context-id`), and on the A2A
`contextId` only when there is none. kagent's remote-agent tool does not give each chat its own context:
by default one context serves every chat that tool handles, and with session isolation each call gets a
new one. Keyed on the context alone, one chat could continue or end another chat's form.

The **A2A adapter** uses [a2a-sdk](https://github.com/a2aproject/a2a-python) server primitives (`AgentExecutor`, `DefaultRequestHandler`, and the route factories). The SDK handles JSON-RPC routing, SSE streaming, task lifecycle, and agent card serving. The adapter implements a single `WFOAgentExecutor.execute()` method that drives the pydantic-ai event stream and publishes A2A events via `TaskUpdater`. The `AgentCard.skills` list is projected from the advertised capability specs (`skills_from_specs`), keeping the advertised skills in sync with the configured capabilities.
