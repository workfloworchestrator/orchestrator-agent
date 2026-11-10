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
| `POST /agui/` | AG-UI | SSE streaming, stored conversation history and workflow forms through interrupts; `/agui` redirects here |
| `/chats/` | REST | AG-UI chat history, pending forms, option searches and export downloads; see [Endpoints and protocols](docs/reference/endpoints.md#chats-api) |
| `POST /` | A2A | Agent-to-agent JSON-RPC, protocol v1.0 (`SendMessage`, `SendStreamingMessage`) |
| `GET /.well-known/agent-card.json` | A2A | Agent card discovery |
| `POST /v1/chat/completions` | OpenAI chat completions | The agent as a chat model for LibreChat; workflow forms through its ask-user tool |
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

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URI` | *(required)* | PostgreSQL connection URI for the WFO database |
| `WFO_CORE_MCP_URL` | `http://localhost:8080/mcp` | URL of orchestrator-core's MCP server (serves the domain tools the agent calls) |
| `WFO_CORE_API_URL` | *(derived)* | Base URL for relative form-option URLs and Core export downloads; when unset, uses the origin of `WFO_CORE_MCP_URL` |
| `WFO_OPTIONS_HOSTS` | `[]` | JSON list of additional hosts allowed for form-option HTTP requests, including a port when present. These destinations receive the run's bearer token or fallback service token |
| `BASE_URL` | `http://localhost:8080` | Public URL of this agent service |
| `AGENT_MODEL` | `openai:gpt-4o` | LLM model in `provider:model` format |
| `AGENT_API_BASE` | *(none)* | Custom base URL for the LLM provider (OpenAI-compatible) or Azure endpoint |
| `AGENT_API_KEY` | *(none)* | API key for the LLM provider |
| `AGENT_API_VERSION` | *(none)* | API version for Azure OpenAI (e.g. `2024-12-01-preview`) |
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
         ──► chat adapter ──►   (the A2A agent, called by LibreChat as its model)
         ──► MCP adapter  ──►   └─► MCPToolset ──► orchestrator-core /mcp
```

All built-in plugins are always on (`defer_loading=False`). **Search**, **aggregate**, **entity** and **export** are
Markdown plugins loaded at startup; **workflow** is added where the form-fill skill runs, on A2A, chat completions and AG-UI.
Each plugin contributes instructions and, when it declares `artifact:`, maps its tools' results to typed metadata
in `capabilities/behavior/`. See [Plugins](#plugins).

The cross-cutting guards cover deferred-tool visibility, discovery before filtering/grouping and history trimming.
On A2A, chat completions and AG-UI, `WriteToolGate` also hides Core's three workflow mutations so the
[form-fill skill](#workflow-form-fill-a2a-human-in-the-loop) controls starts. This gate is absent from the agent's MCP interface.

Search tables and grouped charts are rendered in code. A2A and chat completions append them to the answer; AG-UI
carries rendered blocks in artifact metadata, while MCP returns raw JSON. MCP sessions are opened per run with
`async with agent:`. See [Endpoints and protocols](docs/reference/endpoints.md) for protocol-specific behavior.

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

### Workflow form-fill (A2A, human in the loop)

Starting a workflow is **not done by the model**, and never on the strength of chat text. It is a
deterministic skill (`form_fill/`) that runs inside the agent as a pydantic-ai capability and is answered
by a person through native pauses — kagent's human-in-the-loop extension over A2A, LibreChat's ask-user tool,
or AG-UI interrupts. Every stop carries the questions of a form page, or the start to approve;
only the person's structured response to it continues the form. A caller that cannot show a pause
gets no form (the handoff tool says so and the model relays it). The skill:

1. **Routes** — deciding *that* a request starts a workflow and *which* one is a judgment call, so the
   model makes it: it reads `list_workflows` and hands off with the `start_workflow_form(workflow_key)`
   tool (any key of core's catalogue, tasks included). That is all the model does — on the request that
   follows, the skill walks the first pages and its first stop ends the run in place of the model's
   text. The subscription id the model passes fills the form's first page; a subscription core will not
   run the workflow on is rejected by core's own subscription page, and that rejection is a question
   like any other.
2. **Walks the form from page 0** with core's `get_workflow_form`, using the answers so far; a required
   choice with a single option (a single-product workflow's product page) is taken, each walk anew.
   Pages are dynamic (later pages depend on earlier answers), so the walk re-runs from page 0 every
   turn: a changed answer just changes what core generates next.
3. **Stops** for unanswered fields, consent, unresolved option choices or a page core rejected.
   Core's rejected fields carry its messages; other stops can happen before a page is submitted.
   Questions show the field's title, whether it is required, and its options as chips shown by their
   labels (a boolean is two chips; a list field takes several). A picked chip travels as the value
   behind it; an unanswered question sends nothing, so the form's default applies. A page of optional
   fields nobody has touched is asked once. Consent (pydantic-forms' `Accept`) is only ever the person's
   explicit `ACCEPTED` for the page asking it, and it is void once what it was given for changes. A
   refusal that is not core's verdict on the page's values (a passing failure in core) leaves the form
   open: the page is asked again with what core said.
4. **Confirms** — at `complete: true` the stop is an approval of the `create_workflow` call, shown as it
   will be made (`workflow_key`, `json_data`: exactly the validated pages) plus `labels` saying what its
   ids stand for. Approve starts; reject cancels. A form that ends in the workflow's own summary page
   (core's summary form) has its tables on the reply as `summary`. LibreChat's approval always shows the
   values as they will be sent — that is what is approved — and adds those tables under them: what the
   workflow's author wants confirmed, before and after for a modify.
5. **Starts** — `create_workflow` from code with exactly the last walk's validated pages. The approved branch
   makes one attempt; concurrent requests are not deduplicated across workers. Only core's own validation errors on the pages reopen the form (the rejected
   fields are asked again); any other failure leaves it unknown whether the process started, so the
   form closes.

Internally, the skill's reply text is one JSON object (`FormReply`: `status` gathering / confirming / started /
cancelled / failed, core's `rejected` errors as they came, the `values` so far with their `labels`, the
`defaults` that apply, the `process_id`, or the `reason`). A2A passes this to the calling agent. LibreChat and
AG-UI render it as Markdown for the person through `adapters/words.py`, alongside their structured form stops.

**No chat text is read.** Not by this code and not by a model: a message that carries no response to
the pending stop means the stop went unanswered (the person rejected it, or moved on), so the form is
over and the message goes to the model like any other. Typing "yes" is not an approval. There is one
exception, forced by how a parent runtime works: kagent pauses once per tool call, so the stop that
answers a response is not shown until the parent calls again; such a stop is marked `unseen`, and the
parent's next message (kagent's parent is told to call again with `continue`) shows it instead of ending
the form. Corrections after the fact are not offered: reject, and start the form again.

Core validates every submitted page. What a person *types* for an ordinary field (a number in words, an option described, a structured
field such as a contact) is words, not a value: core rejects it, and it is then interpreted once, a page
at a time, into the fields' values — one run of the agent's model whose output schema is the page schema
with every field optional (`form_fill/interpret.py`) — and core decides again. Words that fit more than
one option of a field are not a value: the model is asked for every option the words fit, and the field
is filled only when that is exactly one — otherwise the person is asked again. The person sees the result
before anything starts, at the approval. The interpreter is one protocol attribute of the skill, so
another engine drops in behind the same protocol.

**One JSON Schema per page.** Core returns a self-contained page schema, with references inlined,
`readOnly` markers, choice values and labels, defaults and descriptions. `form_fill/model.py` reads it into
`FieldSpec` records for the questions and approval labels. The interpreter uses an optional-field variant of
the schema through pydantic-ai's `StructuredDict`; it does not rebuild a Pydantic model. The session persists
the walked schemas and answers. Core's tool results are validated with Core's response models, and its
structured page verdicts supply the validation errors.

**Form-option sources.** Fields with `optionsSource` have their options fetched over HTTP before the page is
read. Typed answers for those fields are resolved against the fetched options before submission to Core: an
exact match supplies the option value; otherwise a model chooses the matching candidates. Ambiguous or
unmatched words are asked again. Relative sources use `WFO_CORE_API_URL`, or Core's MCP origin when unset;
absolute sources must use that base's host and port or an entry in `WFO_OPTIONS_HOSTS`. These calls use the
same bearer-token selection as MCP.

The write-tool gate hides Core's three known workflow mutations where the form-fill skill runs: A2A,
chat completions and AG-UI. The MCP agent has neither that gate nor the workflow plugin, so its model can call
Core's write tools directly. The gate matches tool names, not `readOnlyHint` annotations, and custom tools are not
given an approval step automatically. AG-UI frontend tools are also passed to its underlying adapter. See
[Read tools and write tools](docs/reference/runtime.md#read-tools-and-write-tools).

On A2A, chat completions and AG-UI, built-in workflow starts go through the separate form skill after a matching approval
response.
The calling client must obtain that approval from a person. Conversation and query persistence
still write to the database. Routing in front of the model and prefilling pages from the
conversation with a typed decision engine (Jev) live on a separate branch; a handed-off key core does not
know is a tool retry, so the model corrects itself.

**A2A protocol version.** The A2A endpoint speaks protocol v1.0 (a2a-sdk 1.x, protobuf types:
`SendMessage`, `SendStreamingMessage`; the agent card is at `/.well-known/agent-card.json`; a v1 client
sends `A2A-Version: 1.0`). Callers that have not moved yet are still served: the card also advertises a
0.3 interface and the SDK's compatibility layer accepts `message/send` / `message/stream`
(`enable_v0_3_compat`), as the A2A project recommends for a staged migration. A 0.3 caller gets
everything but workflow forms, which need the extension below. Drop the flag and the 0.3 interface once
no caller needs them.

**kagent human-in-the-loop.** kagent ≥ 1.0.0-alpha1 uses another agent as a tool through its remote A2A
tool and activates the extension `https://kagent.dev/extensions/hitl/v1` (`A2A-Extensions` header). A
stop of the skill is then an `input-required` task whose status message carries, under that URI in its
metadata, an `ask_user_request` (the questions) or a `tool_approval_request` (the start); the person's
response comes back as a message on the same task with the matching `ask_user_response` /
`tool_approval_response`. The wire shapes mirror kagent's `go/api/a2a/hitl.go` (`adapters/a2a/kagent.py`;
kagent's own Python models need a package that is not published), and the same module maps a
response onto the skill's input and a stop onto the pause. Through a kagent parent the questions and the
approval are relayed to the person and their response is forwarded by the runtime, not by the parent's
model. A rejection that reaches this agent cancels the form. kagent's Go runtime does not forward one
today (its ADK ends the parent's tool call before the tool can pass it on), which is why an unanswered
stop also ends the form. One thing is asked of the parent's instruction: when a call to this agent returns while
the form is not finished, call it again (with `continue`) so the next prompt is shown.

**Which conversation a call belongs to.** Memory and the open form are keyed on the chat a parent agent
forwards with every call (`x-kagent-root-context-id`, else `x-kagent-parent-context-id`), and on the A2A
`contextId` only when there is none. kagent's remote-agent tool does not give each chat its own context:
by default one context serves every chat that tool handles, and with session isolation each call gets a
new one. Keyed on the context alone, one chat could continue or end another chat's form.

**LibreChat directly (chat completions).** LibreChat has no A2A client, but from v0.8.8 it has an
ask-user tool, and that is a second human-in-the-loop transport for the same skill. LibreChat calls
`POST /v1/chat/completions` on this agent as a custom endpoint (`adapters/chat/completions.py`). The
adapter knows the caller can show a stop from the `X-Agent-Client: librechat` header, which the endpoint
is configured to send (LibreChat does not identify itself on its own), together with the chat offering a
tool at all: a chat on a model spec with `askUserQuestion: true` carries one, `ask_user_question`, and a
chat without the spec carries none and cannot show a card. The tool's name is not what is checked. A stop
of the skill then goes out as a call of that tool: LibreChat shows the questions as a card (options as buttons, a text
box), pauses its run, and sends the answers back as the tool message of the next request, a picked option
as its value verbatim. `adapters/chat/librechat.py` maps both ways. No model sits between the click and
the skill: LibreChat has none of its own on this path, and a form turn skips this agent's model as on
A2A. What the tool does not take is handled in the adapter:

- four questions a call: a page with more fields is shown as several cards, one after the other;
- twelve options a question, with one slot reserved for "Keep default" on optional fields: a field with more
  is asked as free text, its options listed. Ordinary fields go to Core and, if rejected, to the interpreter;
  sourced-option fields resolve the words against their fetched options before submission;
- every question must be answered: an optional field gets a "Keep default" option;
- the card always takes typed text: an option's value is a token only that stop knows, so typing
  "Approve" approves nothing and the approval is shown again;
- "Skip" keeps the defaults of optional fields and cancels the form when a required field (or the
  approval) is skipped.

A chat message sent instead of an answer ends the form, as on A2A; a chat without the model spec gets no
tool and so no form, and neither does a caller that does not send the header. Memory and the open form are keyed on the `X-Conversation-Id` header, and the
`Authorization` bearer token is forwarded to core. The LibreChat side (a complete file is in
[`examples/librechat.yaml`](examples/librechat.yaml)):

```yaml
endpoints:
  custom:
    - name: 'WFO agent'
      apiKey: 'unused'            # the Authorization header below takes precedence
      baseURL: 'http://orchestrator-agent:8080/v1'
      models: { default: ['wfo'], fetch: false }
      titleConvo: false           # otherwise LibreChat asks this agent for a chat title first
      headers:
        X-Conversation-Id: '{{LIBRECHAT_BODY_CONVERSATIONID}}'
        X-Agent-Client: 'librechat' # required: only a caller that says it is LibreChat gets a form
        Authorization: 'Bearer {{LIBRECHAT_OPENID_ACCESS_TOKEN}}'
modelSpecs:
  list:
    - name: 'wfo'
      label: 'WFO'
      askUserQuestion: true       # offers the ask-user tool: without it no form is opened
      preset: { endpoint: 'WFO agent', model: 'wfo' }
```

A pending card survives a browser reload; it survives a LibreChat restart only when LibreChat runs with
Redis (`USE_REDIS=true`).

**AG-UI forms and chats.** A client whose `protocolVersion` supports interrupts can show the same skill's
questions and approval. A stop ends the run with an interrupt outcome; the next run answers it with a matching
resume entry. The agent replays stored history for the `threadId` and ignores client-supplied state. AG-UI and
its `/chats/` API check the authenticated subject against the thread owner and refuse threads from other
protocols. With authentication off, threads are unowned and shared. A2A and chat completions still trust their
clients to isolate conversation identifiers. See [Endpoints and protocols](docs/reference/endpoints.md#ag-ui).

The **A2A adapter** uses [a2a-sdk](https://github.com/a2aproject/a2a-python) server primitives (`AgentExecutor`, `DefaultRequestHandler`, and the route factories). The SDK handles JSON-RPC routing, SSE streaming, task lifecycle, and agent card serving. The adapter implements a single `WFOAgentExecutor.execute()` method that drives the pydantic-ai event stream and publishes A2A events via `TaskUpdater`. The `AgentCard.skills` list is projected from the advertised capability specs (`skills_from_specs`), keeping the advertised skills in sync with the configured capabilities.
