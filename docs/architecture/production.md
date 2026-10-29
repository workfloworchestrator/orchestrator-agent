# Running the agent in your own stack

!!! warning "Outline"
    This page is an outline. Each section lists what it will cover. It gives pointers, not full deployment how-tos,
    because every organisation's stack is different.

The tutorial and the demos run the agent on a laptop with no authentication. This page covers the choices you make
when you run it in your own stack: which clients connect to it, who they act as, where it runs, and what you put
around it.

## Choose your client

- Questions to start with: do people need to start workflows? Does a model of yours already pick tools? Is the caller
  a person or another agent?
- A decision flow (Mermaid) from those questions to a path: `/v1`, `/mcp`, A2A or AG-UI.
- A capability matrix in user terms, per path: memory between turns, rich results, structured data, workflow forms,
  user identity.
    - `/v1`: memory only with an `X-Conversation-Id` header. Workflow forms only for LibreChat (>= 0.8.8).
    - Agent `/mcp`: stateless, no workflow tool, no follow-ups.
    - Orchestrator Core `/mcp/`: raw tools including `create_workflow`. The client's model drives them, so the
      agent's guards do not apply.
    - A2A: the full agent. Workflow forms need the kagent HITL extension.
    - AG-UI: streaming and structured artifacts, no form filling.

## Client families

One short paragraph each: which path to use, what works, what doesn't, and a link to the how-to.

- **Chat UIs** (LibreChat, Open WebUI, any OpenAI-compatible UI) → `/v1`.
  [Connect LibreChat](../guides/librechat.md), plus a planned guide for other OpenAI-compatible clients.
- **Harnesses and IDE agents** (Claude Code, Codex CLI, Cursor, VS Code Copilot, Goose, Claude Desktop) → MCP. When
  to use the agent's MCP and when to use the MCP server of Orchestrator Core.
  [Use MCP clients](../guides/mcp-clients.md).
- **Agent platforms and planners** (kagent, docker-agent, other A2A runtimes) → A2A.
  [Use the agent from an A2A planner](../guides/a2a-planner.md).
- **Your own app** → AG-UI for a web UI, or the OpenAI SDK for backend scripts and bots.
  [Build a frontend with AG-UI](../guides/ag-ui.md).
- **A gateway in front of several clients** (LiteLLM proxy, an MCP gateway) → see [Gateways](#gateways).

## Identity

- Who holds the user's token on each path: a chat UI forwarding OIDC, a harness sending a bearer header, A2A
  propagation.
- MCP caveat: whether the agent's `/mcp` forwards the user's token to Orchestrator Core. To check in the code before
  writing.
- Fallback to a service account with client credentials (`OAUTH2_OUTBOUND_ACTIVE`), and why it grants more than the
  user may do.
- Token exchange (RFC 8693) as an option, depending on your identity provider.
- Link: [Security and observability](../guides/security-observability.md).

## Deployment shapes

Short, one paragraph each, with upstream links:

- **Compose on a VM.** The shape of the example orchestrator.
- **Kubernetes with kagent.** Declarative agents, with the Orchestrator Agent as your own image.
- **Agent only, behind an existing platform.** Your chat or agent platform calls the agent over MCP or A2A.

## Models

- Hosted (OpenAI, Azure OpenAI, Anthropic), self-hosted (Ollama, vLLM) or through a proxy (LiteLLM).
- Data residency and cost.
- Link: [Use your own LLM](../guides/own-llm.md).

## Observability

- Langfuse is built in. Other OpenTelemetry backends work with a small wrapper, see
  [Trace with another OpenTelemetry backend](../guides/security-observability.md#trace-with-another-opentelemetry-backend).
- Following one request from the chat trace to the agent trace to the audit log of Orchestrator Core
  (`traceparent`).

## Gateways

- When you need one: several clients, machine callers, quotas, a central audit log.
- Links to upstream gateway docs and to adopter write-ups.

## Evaluate before you change prompts or models

- Prompt and model drift are production concerns.
- Links: [How we evaluate the agent](evaluation.md), [Benchmark prompt and model changes](../guides/evals.md).

## Known pitfalls

- The A2A agent card URL (`BASE_URL`) set to a container address that clients can't reach.
- Internal and external hostnames mixed up in callback URLs.
- A missing `X-Conversation-Id` header, so the agent forgets the conversation.
- A LibreChat version too old for workflow forms.
- Harnesses that auto-approve destructive tools of Orchestrator Core.
- Cost runaway without per-caller limits.

## Write-ups from adopters

A neutral list of how organisations run the agent. To be filled once write-ups are public.
