# orchestrator-agent

[![Container](https://ghcr-badge.egpl.dev/workfloworchestrator/orchestrator-agent/latest_tag?trim=major&label=container)](https://github.com/workfloworchestrator/orchestrator-agent/pkgs/container/orchestrator-agent)

Standalone WFO search agent for deployment. Exposes the orchestration search agent via AG-UI, A2A, and MCP protocols.

## Quick Start

```bash
cp .env.example .env
# Edit .env with your DATABASE_URI and LLM settings

uv sync
uv run uvicorn orchestrator_agent.app:app --port 8000
```

## Endpoints

| Path          | Protocol | Description                  |
| ------------- | -------- | ---------------------------- |
| `POST /agui`  | AG-UI    | SSE streaming for frontend   |
| `/a2a`        | A2A      | Agent-to-agent protocol      |
| `/mcp`        | MCP      | Model Context Protocol tools |
| `GET /health` | REST     | Health check                 |

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

## Architecture

This repo contains only the agent logic (planner, skills, tools, adapters, state, memory, prompts). The search infrastructure (query engine, filters, retrievers, indexing, DB models) lives in `orchestrator-core` and is imported as a dependency via `orchestrator-core[search]` (along with everything else for now).
