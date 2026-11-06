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
[Workflow Orchestrator Programme](https://www.workfloworchestrator.org). It lets people and other agents work with
the data and workflows in the Workflow Orchestrator without knowing the UI, the API or the search syntax.__

__The Programme is funded by [GÉANT](https://geant.org) through
[the incubator process](https://wiki.geant.org/spaces/NETDEV/pages/613122207/NETDEV+Incubator) within the Network
Development Work Package (WP6) of the GN5-2 project.__

## What is the Orchestrator Agent?

Ask a question in plain language and the agent searches your subscriptions, products, workflows and processes for the
answer. It can also start workflows from the conversation, with an approval of the person before anything changes.

### An agent and its harness

An AI agent is a language model that does more than answer: it chooses tools, reads what they return and decides on
its next step until the task is done. The model alone cannot search a database or start a workflow. The software
around it does that. This surrounding software is called the *agent harness*. It gives the model its instructions and
tools, runs the tool calls, enforces rules and approvals, keeps track of the conversation and returns the result to
whoever asked.

The Orchestrator Agent is a harness for WFO. You choose the language model; the agent provides the rest:

- **Instructions**: prompts that tell the model how to search, aggregate, look up, export and run workflows.
- **Tools**: the operations of [Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/), made
  available over MCP.
- **Guardrails**: rules that apply to every tool call, and a person's approval before a workflow changes anything.
- **Memory**: conversation history and open workflow forms, stored in the WFO database.
- **Interfaces**: chat applications, MCP clients and other agents can all talk to it.

It is a thin service on top of the search index and the MCP server of Orchestrator Core, so the agent itself keeps no
second copy of your data or your workflow logic. See [How the agent fits on top of Orchestrator Core](architecture/index.md) for details.

The language model is a separate matter: what the agent sends to it, such as search results, may be stored by the
provider of that model. If your data may not leave your own environment, run the agent with a model of your own, see
[Use your own LLM](guides/own-llm.md).

## Key Features

- **Chat with an LLM about your WFO**: search, count, look up and export subscriptions, products and processes, with tables and
  charts built from the real data. See the [plugins reference](reference/plugins.md)
- **Workflows with approval**: start a workflow from a conversation. Nothing changes until a person approves it. See
  [Start workflows from chat](https://workfloworchestrator.org/learn/orchestrator-agent/use-cases/workflows-from-chat/)
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

The agent's [capabilities](architecture/index.md#capabilities) are built from [plugins](reference/plugins.md), and you can add your own. All endpoints are
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
