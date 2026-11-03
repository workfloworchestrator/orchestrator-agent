---
description: "Orchestrator Agent: an AI agent for Workflow Orchestrator that answers questions about your orchestrator and starts workflows from a conversation."
hide:
  - toc
---

<p align="center"><em>An AI agent for Workflow Orchestrator. Ask questions in plain language and start workflows from a conversation, over OpenAI chat, A2A, AG-UI and MCP.</em></p>

<p align="center">
    <a href="https://github.com/workfloworchestrator/orchestrator-agent/pkgs/container/orchestrator-agent" target="_blank">
    <img src="https://ghcr-badge.egpl.dev/workfloworchestrator/orchestrator-agent/latest_tag?trim=major&label=container" alt="Container">
    </a>
    <a href="https://github.com/workfloworchestrator/orchestrator-agent" target="_blank">
    <img src="https://img.shields.io/github/stars/workfloworchestrator/orchestrator-agent?style=flat-square&logo=github" alt="GitHub Stars">
    </a>
    <a href="https://discord.gg/fQkQn5ajFR" target="_blank">
    <img alt="Discord" src="https://img.shields.io/discord/1295834294270558280?style=flat-square&logo=discord&label=discord">
    </a>
</p>

<br>

__The Orchestrator Agent is a ready-to-use AI agent for the
[Workflow Orchestrator Programme](https://www.workfloworchestrator.org), funded by [GÉANT](https://geant.org) through
the NETDEV Incubator. It lets people and other agents work with the data and workflows in WFO without knowing the UI,
the API or the search syntax.__

## What is the Orchestrator Agent?

Ask a question in plain language and the agent searches your subscriptions, products, workflows and processes for the
answer. It can also start workflows from the conversation, with an approval of the person before anything changes.

It is a thin service on top of the search index and the MCP server of
[Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/), so the agent itself keeps no second copy of your data or
your workflow logic. It needs a running Orchestrator Core to work.

The language model is a separate matter: what the agent sends to it, such as search results, may be stored by the
provider of that model. If your data may not leave your own environment, run the agent with a model of your own, see
[Use your own LLM](guides/own-llm.md).

## Key Features

- **Chat with an LLM about your WFO**: search, count, look up and export subscriptions, products and processes, with tables and
  charts built from the real data. See the [plugins reference](reference/plugins.md)
- **Workflows with approval**: start a workflow from a conversation. Nothing changes until a person approves it. See
  [Start workflows from chat](guides/workflows-from-chat.md)
- **Fits your clients**: use it from LibreChat, MCP clients, other agents or your own web frontend. See
  [Endpoints and protocols](reference/endpoints.md)
- **Your own LLM**: choose the language model that fits your requirements for cost, quality and where your data may
  go. See [Use your own LLM](guides/own-llm.md)
- **Knows your domain**: teach it the terms of your domain and add your own tools, services and agents. See
  [Teach the agent your domain](guides/domain-context.md)
- **Measured and observable**: benchmark prompt and model changes before they ship, and trace requests and costs. See
  [Benchmark changes](guides/evals.md)

## What's Included?

The agent is published as a container image on the
[GitHub Container Registry](https://github.com/workfloworchestrator/orchestrator-agent/pkgs/container/orchestrator-agent)
and can also be run from source. One service exposes the agent over several protocols:

| Path | Protocol | Use it for |
| --- | --- | --- |
| `POST /v1/chat/completions` | OpenAI chat completions | LibreChat and other OpenAI-compatible clients |
| `POST /` | A2A | Other agents, such as a planner |
| `POST /agui` | AG-UI | Streaming answers, tables and charts into your own frontend |
| `/mcp` | MCP | MCP clients such as Claude Code or an agent framework |

The agent's capabilities are built from [plugins](reference/plugins.md), and you can add your own. All endpoints are
described in [Endpoints and protocols](reference/endpoints.md).

## Getting Started

### Run the agent

```shell
git clone https://github.com/workfloworchestrator/orchestrator-agent.git
cd orchestrator-agent
cp .env.example .env
uv sync
uv run uvicorn orchestrator_agent.app:app --port 8080
```

### Point it at Orchestrator Core

```shell
DATABASE_URI=postgresql+psycopg://user:pass@localhost/orchestrator
WFO_CORE_MCP_URL=http://localhost:8080/mcp
AGENT_MODEL=openai:gpt-4o
AGENT_API_KEY=<your key>
OAUTH2_ACTIVE=false
```

See [Getting started](getting-started.md) for the full walkthrough. To try the agent without setting anything up
first, follow the
[tutorial on the example orchestrator](https://workfloworchestrator.org/learn/orchestrator-agent/tutorials/example-orchestrator/).

## Technology Stack

- **[Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/)**: the search index and MCP server that
  the agent builds on
- **[Pydantic AI](https://ai.pydantic.dev)**: the agent framework
- **[A2A](https://a2a-protocol.org)**: the agent-to-agent protocol
- **[AG-UI](https://docs.ag-ui.com)**: the protocol for streaming agent output into a frontend
- **[MCP](https://modelcontextprotocol.io)**: the Model Context Protocol, for tools and clients
- **[Uvicorn](https://www.uvicorn.org)**: the ASGI server

## Community & Support

- 💬 **Questions?** Join us on [Discord](https://discord.gg/fQkQn5ajFR)
- 🐛 **Found a bug?** [Open an issue](https://github.com/workfloworchestrator/orchestrator-agent/issues)
- 🤝 **Want to contribute?** Visit the [repository](https://github.com/workfloworchestrator/orchestrator-agent)
- 📖 **How does it work?** Read the [architecture](architecture/index.md)
- 🚀 **From first try to production?** See
  [Learn / Orchestrator Agent](https://workfloworchestrator.org/learn/orchestrator-agent/)

## Special thanks

<div class="grid cards" markdown>

-   [![GÉANT](img/geant-logo.png){ align=right width="160" }](https://geant.org)
    Special thanks to **GÉANT** for funding this work through the
    [NETDEV Incubator](https://wiki.geant.org/spaces/NETDEV/pages/613122207/NETDEV+Incubator).
    Two incubator projects, [*WFO RAG agent and LLM integration*](https://wiki.geant.org/spaces/NETDEV/pages/631443164/NETDEV+Incubator+proposals)
    (2025) and *A2A protocol integration, agent refactoring and multi-agent troubleshooting* (2026), got the
    Orchestrator Agent off the ground.

</div>
