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
- A feature matrix in user terms, per path: memory between turns, rich results, structured data, workflow forms,
  user identity.
    - `/v1`: memory only with an `X-Conversation-Id` header. Workflow forms only for LibreChat (>= 0.8.8).
    - Agent `/mcp`: stateless, no workflow forms, no follow-ups.
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
  [Use the agent with MCP](../guides/mcp-clients.md).
- **Agent platforms and planners** (kagent, docker-agent, other A2A runtimes) → A2A.
  [Connect another agent over A2A](../guides/a2a-planner.md).
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

### What the example setup does

In the [example orchestrator](https://github.com/workfloworchestrator/example-orchestrator), LibreChat has no A2A
client, like many chat UIs, portals and services, so it reaches the agents through a small proxy,
[kagent-a2a-proxy](https://github.com/nren-agents/kagent-a2a-proxy) (the `a2a-proxy` service in
`docker-compose.agents.yml`). The proxy turns OpenAI chat completions into A2A calls, using a fixed map from model
name to agent URL. In effect it is a hand-wired gateway for one protocol, one client and no users. That is enough for
a laptop, but it leaves open questions that matter as soon as you run the agent for real:

- **No identity.** The proxy accepts any API key and Orchestrator Core runs without OAuth, so every request is
  anonymous. Nothing downstream can decide per user or per agent.
- **Not the only way in.** LibreChat also calls the MCP servers of Orchestrator Core and NetBox directly, and the
  agents call each other over A2A. A check you add to the proxy covers only one of those paths.
- **No limits and no shared trace.** There is no rate limit or budget per caller, and LibreChat, the proxy, the agents
  and Orchestrator Core each log on their own.
- **Fixed addresses.** Agent URLs and agent cards point at container addresses, see [Known pitfalls](#known-pitfalls).

### Questions a gateway answers

Once you have real users, machine callers or MCP servers you don't own, every request has to answer the same
questions. No single agent or MCP server should answer them on its own:

- **Who is calling, and for whom?** A user, an agent acting for a user, or an agent on its own. A shared service
  account (see [Identity](#identity)) hides the difference and grants more than the caller may do.
- **Which credential goes to the next hop?** Forwarding the user's token to an MCP server you don't own hands it a
  credential your own backends accept.
- **Which agents and tools may the caller see?** MCP lists every tool of a server, including `create_workflow`,
  unless something filters that list per caller.
- **How much may the caller spend?** An alert on cost does not stop an agent stuck in a loop; a budget per user and
  per agent does.
- **What happened?** One trace from the chat message to the audit log of Orchestrator Core, see
  [Observability](#observability).

These are the issues you may run into. Virge.io helped develop the Orchestrator Agent and runs it in its own stack. On
its website it works these issues out as requirements for a gateway, in the blog post
[Why (y)our agents need a gateway (part 1)](https://virge.io/en/blog/why-your-agents-need-a-gateway-part-1/){target="_blank" rel="noopener"}
:octicons-link-external-16:.

### When you need one

Put a gateway in front once you have several clients, machine callers, quotas or a central audit log, instead of
extending the proxy of the example setup. Depending on the traffic this is an API or agent gateway in front of the
agents and MCP servers, or a model proxy such as LiteLLM between the agents and the model providers. See
[Write-ups from adopters](#write-ups-from-adopters) for how others did it.

## Evaluate before you change prompts or models

- Prompt and model drift are production concerns.
- Links: [How we evaluate the agent](evaluation.md), [Benchmark prompt and model changes](../guides/evals.md).

## Known pitfalls

- The A2A agent card URL (`BASE_URL`) set to a container address that clients can't reach.
- Internal and external hostnames mixed up in callback URLs.
- A missing `X-Conversation-Id` header, so the agent forgets the conversation.
- A LibreChat version too old for workflow forms.
- Harnesses that auto-approve destructive tools of Orchestrator Core.
- Cost runaway without per-caller limits, see [Gateways](#gateways).
- A client without A2A connected to a planner through a demo proxy that checks no identity, see [Gateways](#gateways).

## Write-ups from adopters

A neutral list of how organisations run the agent.

- Virge.io, which helped develop the Orchestrator Agent and runs it in its own stack:
  [Why (y)our agents need a gateway (part 1)](https://virge.io/en/blog/why-your-agents-need-a-gateway-part-1/){target="_blank" rel="noopener"}
  :octicons-link-external-16:. The cross-cutting questions of an agent network, and the requirements for a
  gateway that answers them.
