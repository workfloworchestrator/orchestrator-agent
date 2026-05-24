# Architecture

## Why BYO and not a declarative kagent agent?

A kagent declarative agent is a system prompt + a flat list of tools (MCPs
and/or other agents). That works for "answer this question by picking one
tool and calling it" — which is most of what the IMS / CIM / Jira / telemetry
/ alarming domain agents do. It does **not** work for the orchestrator's job,
because we need state and control flow the declarative shape doesn't expose:

- **Multi-page workflow form-fill.** Starting a workflow in WFO core is a
  loop: fetch the page schema → ask the user → submit → fetch the next page
  schema → … → confirm. Between turns we have to persist `page_inputs` and
  the current page schema across A2A calls (Postgres-backed
  `PostgresStatePersistence`, keyed by `thread_id`). A declarative agent has
  no place to put that — its only "memory" is whatever lives in the LLM's
  message history, which is exactly the wrong shape for structured per-form
  state.
- **Structured artifacts as outputs, not just text.** We emit typed
  `RenderFormArtifact` / `ConfirmRequestArtifact` payloads over A2A so Slack
  (and later LibreChat) can render real Block Kit input fields and buttons
  instead of free-text. The declarative agent only knows how to return text +
  forwarded sub-agent artifacts; it can't construct its own typed artifacts.
- **Per-turn adapter metadata routing.** A Slack button click arrives as
  `{"form_submission": {...}}` in `message.metadata`, not as user text. The
  A2A adapter routes that to the form-fill skill, which reads it from
  `state.adapter_metadata` and submits the page. A declarative agent has no
  hook to consume non-text inputs.
- **Plan-and-execute with intermediate state.** The planner emits a list of
  Tasks; each task runs as a sub-agent and writes results back to
  `SearchState` (used by downstream tasks). Declarative agents are
  single-pass.
- **Skill catalogue with direct-dispatch for delegations.** Routing to a
  domain agent is deterministic once the planner has picked the skill — no
  second LLM call. `DelegationSkill` skips pydantic-ai entirely and calls
  `call_a2a` directly. A declarative agent would burn an LLM turn on each
  hand-off.
- **Forwarding remote `input-required` artifacts.** When a downstream agent
  asks for clarification (kagent's `ask_user`), we synthesize an
  `a2a_status_input-required` artifact and surface it to the user end-to-end.
  Doing this in a declarative agent would require the runtime to know about
  the downstream agent's `TaskState.input_required` semantics; ours doesn't.
- **LibreChat-renderable wire format.** LibreChat speaks OpenAI chat-completions,
  not A2A — it has no native handler for kagent's structured artifacts
  (`adk_request_confirmation`, ADK function calls, `input-required`
  clarifications). BYO translates A2A `DataPart`s into `wfo-artifact:` markdown
  fences that the LibreChat fork's `ArtifactDispatcher` mounts as React
  components — same end-user UX as kagent's own UI. A LibreChat→kagent-direct
  path would lose every interactive artifact (clarifications, forms, choice
  chips) unless kagent UI's renderers were ported into LibreChat.

So: declarative is the right shape for the **leaf** agents (one MCP, one
prompt, one tool pick per call). BYO is the right shape for the **router /
form-driver** at the top because it owns persistent per-thread state and
emits structured, surface-renderable outputs.

**The deeper technical reason:** ADK's `AgentTool` — the mechanism a
declarative agent uses to invoke any sub-agent — text-extracts the
sub-agent's response (only `last_content.parts`'s text/code; `DataPart`
payloads are discarded; see [`agent_tool.py`](https://github.com/google/adk-python/blob/main/src/google/adk/tools/agent_tool.py)).
So a declarative orchestrator wrapping wfo-agent would receive its
sub-agent's `QueryArtifact` (chart spec) or `RenderFormArtifact`
(pydantic-forms schema), silently drop the structured `DataPart`, and
return only the LLM's text summary upstream — LibreChat would get
"there are 12 subscriptions" instead of a pie chart, and "please fill in
the form" instead of a form. A BYO orchestrator handles the A2A
protocol directly (via `a2a-sdk`) and forwards the `DataPart` through
`adapters/a2a.py` unchanged, which is what makes search visualizations
and form-fill rendering work end-to-end.

## MCP connections

The planner does not connect to MCP servers directly. Each MCP is wrapped by
a **domain agent** running in kagent; the planner delegates over A2A. The
domain agent owns the system prompt, model choice, and per-tool curation for
its MCP.

| MCP | Domain agent (kagent) | Notable tools |
|---|---|---|
| WFO core | planned dedicated WFO BYO domain agent — currently built into the planner | `list_workflows`, `get_workflow_form`, `create_workflow`, plus search/aggregation tools |
| IMS | `ims-mcp-agent` *(declarative)* | `node_by_ims_node_name`, `get_planned_works_by_circuit_id`, `service_by_ims_service_id` |
| CIM | declarative kagent agent | `get_tickets`, `create_ticket`, `get_ticket_by_id` |
| Jira | declarative kagent agent | 13 tools — list/get/create/transition/close/comment/assign on customers + tickets + locations |
| Telemetry (Influx) | `telemetry-agent` *(declarative)* | `execute_query`, `get_help`, `get_measurement_schema`, `get_measurements`, `load_database_context` |
| Alarms (Zabbix) | `alarming-agent` *(declarative)* | `event_get`, `problem_get`, `history_get`, `item_get`, `service_get`, `trend_get`, `maintenance_get`, `host_get`, `hostgroup_get`, `graph_get` |

**Current exception:** WFO operations — `search`, `aggregation`, `result_actions`
and the multi-page `workflow_form_fill` — are skills built directly into the
planner today, hitting WFO core's MCP (and REST API) without an intervening
domain agent. The plan is to extract them into a dedicated WFO BYO domain
agent so the planner becomes purely a router; for now this codebase plays
both roles. Deployed in kagent as `wfo-search` because search is its most
user-visible capability today.

## Architecture: planner + domain agents

```
       ┌─────────────────────────────────────────────────────────────────┐
       │              orchestrator-agent (planner)                       │
       │  ┌────────────────────────────────────────────────────────┐     │
       │  │  Planner: LLM classifies user intent → list of Tasks   │     │
       │  └────────────────────────────────────────────────────────┘     │
       │     │              │              │              │              │
       │   Skill          Skill          Skill          Skill             │
       │   (wfo)        (telemetry)    (alarms)       (incidents)         │
       │     │              │              │              │              │
       └─────┼──────────────┼──────────────┼──────────────┼───────────────┘
             │ A2A          │ A2A          │ A2A          │ A2A
             ▼              ▼              ▼              ▼
       ┌──────────┐   ┌────────────┐  ┌────────────┐  ┌────────────┐
       │  wfo     │   │ telemetry- │  │ alarming-  │  │ cim/jira/  │
       │  domain  │   │ agent      │  │ agent      │  │ ims-domain │
       │  agent   │   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
       └─────┬────┘         │ MCP           │ MCP           │ MCP
             │ MCP          ▼               ▼               ▼
             ▼        ┌──────────┐    ┌──────────┐   ┌─────────────────┐
        ┌──────────┐  │ Influx   │    │ Zabbix   │   │ CIM/Jira/IMS    │
        │ WFO core │  │ MCP      │    │ MCP      │   │ MCP servers     │
        │ MCP      │  └──────────┘    └──────────┘   └─────────────────┘
        └──────────┘
```

**Components in this repo:**

- **Planner** (`planner.py`) — pydantic-ai `Agent` whose only job is classifying
  user input into a sequence of `Task(action_type, reasoning)` entries. Skipped
  entirely when the inbound A2A message names a `skill_id` directly.
- **Skill registry** (`skills.py`) — one `Skill` per `TaskAction`. A skill bundles
  a system prompt, a memory scope, and a list of pydantic-ai `Toolset`s.
- **Skill runner** (`skill_runner.py`) — for each Task, spins up a pydantic-ai
  `Agent` with the skill's toolsets and prompt; the LLM picks tool calls.
- **Tools** (`tools/*.py`) — each toolset is either a thin MCP-client wrapper
  (e.g. `wfo_mcp_client.py` + `workflow_forms.py` today) or, going forward, an
  A2A delegate to a domain agent (planned).
- **Protocol adapters** (`adapters/{openai,a2a,ag_ui,mcp}.py`) — four inbound
  surfaces, one agent process. LibreChat reaches the agent over OpenAI
  chat-completions (`/v1/chat/completions`, streaming SSE); AG-UI clients
  (CopilotKit-style embeds) over `/agui/`; other agents over A2A (`/`, JSON-RPC)
  or MCP (`/mcp`). All four surfaces share the same planner, skill registry
  and Postgres-backed state — only the wire protocol differs.

**Where domain agents fit going forward:**

- New skills planned for IMS, CIM, Jira (and reusing `telemetry-agent` /
  `alarming-agent` from kagent) become **A2A clients** in
  `tools/<domain>_a2a.py`. Each domain-agent wrapping is one skill in
  `skills.py`. The planner picks them like it picks any other skill.
- The MCP-client wrapper pattern in `wfo_mcp_client.py` is the prototype for
  *domain agents themselves* — they wrap a single MCP and expose their tools
  over A2A.
