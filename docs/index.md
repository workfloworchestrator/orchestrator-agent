---
description: "Orchestrator Agent: search your Workflow Orchestrator data in plain language and start workflows with approval."
hide:
  - toc
---

<p align="center"><em>An AI agent for Workflow Orchestrator. Search your data in plain language and start workflows with approval.</em></p>

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

__The Orchestrator Agent lets people and other agents work with the data and workflows in
[Workflow Orchestrator](https://www.workfloworchestrator.org) (WFO) in plain language. Connect it to your
Orchestrator Core and a language model, then use it from a chat application or another agent.__

__The Programme is funded by [GÉANT](https://geant.org) through
[the incubator process](https://wiki.geant.org/spaces/NETDEV/pages/613122207/NETDEV+Incubator) within the Network
Development Work Package (WP6) of the GN5-2 project.__

## What is the Orchestrator Agent?

The agent runs alongside [Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/). It searches your
subscriptions, products, workflows and processes, looks up details, counts and groups results, and prepares exports.
It can also guide a person through a workflow form and ask for approval before starting the workflow, using a client
that supports this interaction.

### An agent and its harness

The language model interprets requests, chooses tools and uses their results to compose an answer. The software
around the model is its *agent harness*: it provides instructions, executes tool calls, applies rules and handles
the conversation. This repository provides that harness for WFO, including plugins for common tasks, result
rendering, workflow forms and client interfaces. You choose the model.

Orchestrator Core provides the tools through MCP and owns the search index, input validation and workflow execution.
The agent stores conversation state in the WFO database; it does not maintain a separate search index or its own
workflow logic. See [How the agent fits on top of Orchestrator Core](architecture/index.md).

## Key Features

- **Search and analyse WFO data**: find records, count and group results, look up an entity by id and export results
  as CSV. Tables and charts are built in code from tool results. See the [plugins reference](reference/plugins.md).
- **Start workflows with approval**: collect and validate form inputs, then ask the person to approve the start
  through LibreChat or an A2A client with human-in-the-loop support. See
  [Start workflows from chat](https://workfloworchestrator.org/learn/orchestrator-agent/use-cases/workflows-from-chat/).
- **Connect your clients**: use the agent from [LibreChat](guides/librechat.md),
  [MCP clients](guides/mcp-clients.md) or [other agents over A2A](guides/a2a-planner.md).
- **Choose your model**: use a hosted provider or a model in your own environment. The model processes requests,
  conversation history and tool results, so choose where that data may go. See [Use your own LLM](guides/own-llm.md).
- **Adapt it to your domain**: supply your identifiers and terminology through
  [domain context](guides/domain-context.md), customise [plugins](reference/plugins.md#customise-a-plugin), or
  [connect another service](guides/connect-service.md).
- **Evaluate and observe**: [benchmark prompt and model changes](guides/evals.md),
  [add evaluation cases](guides/add-eval-case.md) and [trace requests and costs](guides/tracing.md).

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

Search, aggregation, entity lookup and export are available through all four interfaces. Workflow forms require
LibreChat configured with its ask-user tool, or an A2A client that supports the human-in-the-loop extension. MCP and
AG-UI do not offer these forms. See [Endpoints and protocols](reference/endpoints.md) for client support and
[Read tools and write tools](reference/runtime.md#read-tools-and-write-tools) for the write-tool policy on each
interface.

## Getting Started

To run the agent with your own orchestrator, follow [Getting started](getting-started.md). You need an Orchestrator
Core with MCP enabled, access to its PostgreSQL database and a language model connection.

To try it on the public example setup, follow
[Run it on the example orchestrator](https://workfloworchestrator.org/learn/orchestrator-agent/tutorials/example-orchestrator/).
The tutorial sets up Core, the agent and a chat client together.

Before exposing the agent to other users, follow [How to secure the agent](guides/secure-agent.md). For the wider
deployment picture, see [Running the agent in your own stack](https://workfloworchestrator.org/learn/orchestrator-agent/production/).

## Technology Stack

- **[Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/)**: the search index and MCP server that
  the agent builds on
- **[Pydantic AI](https://ai.pydantic.dev)**: the agent framework
- **[PostgreSQL](https://www.postgresql.org)**: conversation and run storage in the WFO database
- **[FastAPI](https://fastapi.tiangolo.com)** and **[Uvicorn](https://www.uvicorn.org)**: the HTTP application and ASGI server
- **[A2A](https://a2a-protocol.org)**: the agent-to-agent protocol
- **[AG-UI](https://docs.ag-ui.com)**: the protocol for streaming agent output into a frontend
- **[MCP](https://modelcontextprotocol.io)**: the Model Context Protocol, for tools and clients

## Community & Support

- 💬 **Questions?** Join us on [Discord](https://discord.gg/fQkQn5ajFR)
- 🐛 **Found a bug?** [Open an issue](https://github.com/workfloworchestrator/orchestrator-agent/issues)
- 🤝 **Want to contribute?** Visit the [repository](https://github.com/workfloworchestrator/orchestrator-agent)
- 📖 **How does it work?** Read the [architecture](architecture/index.md)
- 🚀 **From first try to production?** See
  [Learn / Orchestrator Agent](https://workfloworchestrator.org/learn/orchestrator-agent/)
