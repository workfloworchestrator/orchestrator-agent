# orchestrator-agent

[![Container](https://ghcr-badge.egpl.dev/workfloworchestrator/orchestrator-agent/latest_tag?trim=major&label=container)](https://github.com/workfloworchestrator/orchestrator-agent/pkgs/container/orchestrator-agent)

Standalone WFO search agent for deployment. Exposes the orchestration search agent via AG-UI, A2A, and MCP protocols, and as an OpenAI-compatible chat model for LibreChat.

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
| `POST /agui/` | AG-UI | The agent behind a frontend (the chat popup, [orchestrator-agent-chat](https://github.com/workfloworchestrator/orchestrator-agent-chat); a bare page for it in `demos/web`); a workflow form's approval through an AG-UI interrupt |
| `GET /chats/`, `GET /chats/{thread_id}`, `DELETE /chats/{thread_id}`, `GET /chats/{thread_id}/exports/{query_id}` | REST | The person's AG-UI chats: list them, read one (its last messages, the approval it waits on), forget one, download an export the chat prepared as a CSV file |
| `POST /` | A2A | Agent-to-agent JSON-RPC, protocol v1.0 (`SendMessage`, `SendStreamingMessage`) |
| `GET /.well-known/agent-card.json` | A2A | Agent card discovery |
| `POST /v1/chat/completions` | OpenAI chat completions | The agent as a chat model for LibreChat; a workflow form's approval through its ask-user tool |
| `GET /v1/models` | OpenAI | Lists the one model id (`wfo`) |
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

**Workflow forms in the popup.** A form is filled the same way as through kagent and LibreChat (see
[Workflow form-fill](#workflow-form-fill-the-model-fills-a-person-approves)): in the conversation, by the
model, with no component per field. The popup shows one card, the start to approve with the values the
workflow gets, and sends the person's decision back. The transport is AG-UI's own interrupt lifecycle
(`adapters/ag_ui/interrupts.py`): the approval ends the run with an interrupt outcome, the card is that
interrupt, and the next run carries its resume entry. No model reads the decision. A message typed while the
card is open rejects the start with those words, and the model acts on them before the start is offered again.

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
| `BASE_URL` | `http://localhost:8080` | Public URL of this agent service |
| `AGENT_MODEL` | `openai:gpt-4o` | LLM model in `provider:model` format |
| `AGENT_API_BASE` | *(none)* | Custom base URL for the LLM provider (OpenAI-compatible) or Azure endpoint |
| `AGENT_API_KEY` | *(none)* | API key for the LLM provider |
| `AGENT_API_VERSION` | *(none)* | API version for Azure OpenAI (e.g. `2024-12-01-preview`) |
| `AGENT_DOMAIN_CONTEXT` | *(empty)* | Optional free-text domain knowledge appended to the agent system prompt (e.g. identifier conventions and their filter fields). Empty disables the section |
| `WFO_CORE_API_URL` | *(derived)* | Base URL a form field's relative options source (`optionsSource.url`) is fetched from. Unset: the origin of `WFO_CORE_MCP_URL` |
| `WFO_OPTIONS_HOSTS` | `[]` | Hosts besides core's that a form field's absolute options source may be fetched from, with the person's token |
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
         ──► chat adapter ──►   (one agent with the form-fill skill behind all three)
         ──► MCP adapter  ──►   └─► MCPToolset ──► orchestrator-core /mcp
```

Capabilities are always-on (`defer_loading=False`). The domain capabilities — **search**, **aggregate**, **entity** (details/lookup), **export**, **workflow** (start a workflow via its input form) — are **plugins**: authored Markdown files loaded at startup (see [Plugins](#plugins) below). Each plugin becomes one pydantic-ai capability that bundles its instructions and, when it declares `artifact:`, the matching behaviour — mapping *its* tools' JSON results into `QueryArtifact` / `DataArtifact` / `ExportArtifact` metadata for the AG-UI/A2A transport (`capabilities/behavior/`). Three hooks remain cross-cutting (single invariants across all tools, not plugin-owned): `FilterPathGuard` enforces that any `filters`/`group_by` call is preceded by `discover_filter_paths` (paths are DB-specific and must not be guessed), `WriteToolGate` hides core's write tools from the model (a workflow is started by the form-fill skill, after a person approved — see [Workflow form-fill](#workflow-form-fill-the-model-fills-a-person-approves)), and `ProcessHistory` does sliding-window history trimming. Grouped aggregations and search results are rendered deterministically in code (a Mermaid chart / Markdown table carried as a `RenderedBlock`) and injected into the answer, so they appear even on text-only clients; an aggregate also carries its chart as data (`ChartData`) for a client that draws its own, the popup. MCP tools require an open session, so every run happens inside `async with agent:`.

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

### Workflow form-fill (the model fills, a person approves)

Starting a workflow is a conversation with a hard edge. The model fills the workflow's input form with the
person, page by page; the state machine, every option list and the start itself are code (`form_fill/`), and
nothing starts without a person's structured approval — kagent's human-in-the-loop extension over A2A,
LibreChat's ask-user tool, or an AG-UI interrupt (the chat popup). No client needs a component per form
field: the questions are the model's own words. A caller that cannot show an approval can have a form
prepared, not started.

1. **Hands off** — deciding *that* a request starts a workflow and *which* one is a judgment call, so the
   model makes it: it reads `list_workflows` and calls `start_workflow_form(workflow_key, subscription_id?)`
   (any key of core's catalogue, tasks included; a key core does not know is a tool retry). The tool opens
   the form on core's first page. A subscription id the model passes is submitted by code where the first
   page asks for one, so core's verdict on it (a subscription that is not in sync) is the first thing the
   model reads.
2. **Fills** — while a form is open the model has the page tools, built per model request from the form's
   state: `submit_page`, whose parameters are the current page's JSON Schema as core wrote it (less the
   frontend's hints, the fields that are only shown, and consent), `search_options` for a long list,
   `restart_from_page` to change an accepted page, and `cancel_form`. It reads what the person said into
   the page, asks in plain words for what is missing, and fixes what core rejected. Its instructions are
   rendered per request (`form_fill/instructions.py`): the rules, the pages accepted with their values, the
   current page, core's last verdict, the options its searches found.
3. **Code keeps the machine** (`form_fill/session.py`, `skill.py`). The pages core accepted are exactly what
   `create_workflow` is called with, and core's validation is the only validation. A short option list is in
   the tool's schema (`enum` with `options` as labels); a long one is found with `search_options`, and such a
   field only takes a value a search returned in this form — checked before core, which takes any string for
   it — so the model cannot invent an id. What a page fills by itself is filled by code: consent
   (pydantic-forms' `Accept`: the approval at the end is the person's consent, before anything is written)
   and a required choice with one option. A page has a budget of tool calls, after which the tools tell the
   model to ask the person.
4. **Confirms** — when core says the form is complete the run ends on the approval, in place of the model:
   the `create_workflow` call as it will be made (`workflow_key`, `json_data`: exactly the accepted pages)
   plus `labels` saying what its ids stand for, rendered by code from the accepted pages. A form that ends
   in the workflow's own summary page (core's summary form) has its tables on the reply as `summary`. The
   model never writes what the person approves.
5. **The person decides.** Approve starts. Reject cancels. A rejection with words — typed under the card in
   the popup or LibreChat, or kagent's rejection reason — or a message instead of a decision reopens the
   form: the model acts on it (`restart_from_page`, then the corrected page) and the approval returns.
6. **Starts** — `create_workflow` from code, once per approval, and only as the answer to an approval that is
   out. Only core's own verdict on the pages reopens the form (the page it rejects is current again, with
   its words); any other failure leaves it unknown whether the process started, so the form closes.

The skill's own replies are one JSON object each (`FormReply`: `status` confirming / started / cancelled /
prepared / failed, or gathering for a start core refused; the `values` with their `labels`, the `defaults`
that apply, the `process_id`, or the `reason`) — core's data, nothing phrased here. A person in a chat reads
them as Markdown (`adapters/words.py`); a calling agent gets the JSON.

**What the model can and cannot do.** Inside a page the model is in the loop, so its quality matters there:
it may misread a sentence or pick between two options it should have asked about, and the approval is where
the person sees it. `AGENT_MODEL` should be a gpt-4o class model or better. What it cannot do is structural:
it has no write tool, it cannot submit a searched value no search returned, it does not see or write the
approval, and typed words are never a start. One exception in the transports is forced by how a parent
runtime works: kagent pauses once per tool call, so an approval that answers a response is not shown until
the parent calls again; it is marked `unseen`, and the parent's next message shows it instead of reopening
the form.

**A page as core describes it.** Core's form tool (`get_workflow_form`, orchestrator-core ≥ 5.5) describes
each page as one JSON Schema (`schema`): every reference inlined, so a choice or a nested model is read in
place; `readOnly` on a field that is only shown; a choice's values in `enum` with their labels in `options`;
the limits a list is validated against; consent as `format: "accept"`; each field's label and help text in
`title` and `description`; and, for a field whose options live elsewhere, where they come from
(`optionsSource`, pydantic-forms ≥ 2.10), which `form_fill/widgets/` reads as the person asking. Asked for a
verdict (`verdict="result"`), a rejected page comes back as a result with core's errors. `form_fill/model.py`
reads the schema into one `FieldSpec` per field and `form_fill/pages.py` makes of it what the model's tool
takes and what the approval shows. No model is rebuilt from the schema. Core's tool results are validated
with core's own response models and its tool arguments are built from core's request models (the agent
depends on orchestrator-core), so nothing here reads a tool error's text.

In the agent the skill runs in (A2A, chat completions and AG-UI share one agent) the model never sees
core's write tools (`WriteToolGate` hides `create_workflow` / `resume_workflow_process` /
`abort_workflow_process`), so a workflow is only ever started by this skill; the MCP agent runs without the
skill, so it gets neither the gate nor the `workflow` plugin and behaves as before.
