---
description: "Orchestrator Agent: an AI agent for Workflow Orchestrator that answers questions and starts workflows over OpenAI, A2A, AG-UI and MCP."
---

# Orchestrator Agent

The Orchestrator Agent is an AI agent for Workflow Orchestrator. It answers questions about your subscriptions,
products, workflows and processes, and starts workflows from a conversation, with an approval of the person before
anything changes. It is a thin service on top of the search index and the MCP server of
[Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/), and it needs a running Orchestrator Core to work.

To see what it can do read [What the agent can do](https://workfloworchestrator.org/learn/orchestrator-agent/).

## Protocols

The same agent is available over four protocols.

| Client | Protocol | Endpoint |
| --- | --- | --- |
| LibreChat and other OpenAI-compatible clients | OpenAI chat completions | `/v1` |
| Other agents | A2A | `/`, with the agent card at `/.well-known/agent-card.json` |
| Web frontends | AG-UI | `/agui` |
| MCP clients | MCP | `/mcp` |

See [Endpoints and protocols](reference/endpoints.md) for the details.

## Documentation

- **[Getting started](getting-started.md)**: install the agent and ask a first question
- **Guides**: connect clients and configure the agent for your setup
    - [Connect LibreChat](guides/librechat.md)
    - [Use MCP clients](guides/mcp-clients.md)
    - [Use the agent from an A2A planner](guides/a2a-planner.md)
    - [Build a frontend with AG-UI](guides/ag-ui.md)
    - [Start workflows from chat](guides/workflows-from-chat.md)
    - [Use your own LLM](guides/own-llm.md)
    - [Teach the agent your domain](guides/domain-context.md)
    - [Security and observability](guides/security-observability.md)
    - [Enable AI search and embeddings](guides/ai-search.md)
    - [Benchmark prompt and model changes](guides/evals.md)
- **[Architecture](architecture/index.md)**: how the agent works and why
- **Reference**: [endpoints](reference/endpoints.md), [configuration](reference/configuration.md),
  [plugins](reference/plugins.md) and [evals](reference/evals.md)

For use cases and a tutorial on the example orchestrator, see [Learn / Orchestrator Agent](https://workfloworchestrator.org/learn/orchestrator-agent/).
The source is on [GitHub](https://github.com/workfloworchestrator/orchestrator-agent).
