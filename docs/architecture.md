# Architecture

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
- **Protocol adapters** (`adapters/{a2a,ag_ui,mcp}.py`) — three inbound
  surfaces. Slack reaches the agent via A2A; LibreChat via AG-UI; other agents
  via MCP.

**Where domain agents fit going forward:**

- New skills planned for IMS, CIM, Jira (and reusing `telemetry-agent` /
  `alarming-agent` from kagent) become **A2A clients** in
  `tools/<domain>_a2a.py`. Each domain-agent wrapping is one skill in
  `skills.py`. The planner picks them like it picks any other skill.
- The MCP-client wrapper pattern in `wfo_mcp_client.py` is the prototype for
  *domain agents themselves* — they wrap a single MCP and expose their tools
  over A2A.
