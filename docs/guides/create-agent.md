# Create your own agent

Run an agent with its own tools, instructions and model next to the Orchestrator Agent. A planner combines the
answers of both over [A2A](https://a2a-protocol.org/latest/). Each agent stays responsible for one system. Use this
when a question needs a system that the Orchestrator Agent does not reach, see
[Connect another service](connect-service.md).

An agent that works with the Orchestrator Agent needs three things:

- a runtime that runs the agent loop,
- tools over MCP, from the MCP servers of the systems it works with,
- an A2A endpoint with an agent card, so that a planner can find and call it.

## Choose a runtime

| Runtime | Fits | Docs |
| --- | --- | --- |
| [docker-agent](https://github.com/docker/docker-agent) | Compose or a VM. An agent is an `agent.yaml` with its model, instructions and toolsets | [docker-agent](https://github.com/docker/docker-agent) |
| [kagent](https://kagent.dev/) | Kubernetes. Agents and MCP servers are declared as resources | [kagent docs](https://kagent.dev/docs/kagent) |
| Your own code | Full control over the agent loop, for example with pydantic-ai or the A2A Python SDK | [pydantic-ai A2A](https://ai.pydantic.dev/a2a/), [A2A Python SDK](https://github.com/a2aproject/a2a-python) |

The steps below use docker-agent with Docker Compose. The example is the `inventory-agent` of the
[example orchestrator](https://github.com/workfloworchestrator/example-orchestrator), which reads NetBox through its
MCP server. The same steps apply to the other runtimes, in their own configuration format.

## Steps

### 1. Declare the agent and its tools

Create a folder for the agent, for example `docker/inventory-agent/`, with an `agent.yaml`:

```yaml
agents:
  inventory:
    model: openai/gpt-4o
    description: "Read-only NetBox network inventory agent: sites, devices, interfaces, cables, prefixes, IP addresses, VLANs and L2VPNs."
    instruction_file: instructions.md
    toolsets:
      - type: mcp
        remote:
          url: http://netbox-mcp:8000/mcp
          transport_type: streamable
        allow_private_ips: true  # needed for addresses inside a compose network
```

- `model` is the provider and model, for example `anthropic/claude-sonnet-5`. The API key comes from the environment
  of the container (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- `description` is published in the agent card. A planner reads it to decide when to call this agent, so make it
  accurate and specific.
- `toolsets` lists the MCP servers the agent uses. docker-agent blocks private addresses unless
  `allow_private_ips` is set.

To reach WFO from your own agent, delegate to the Orchestrator Agent over A2A rather than adding the MCP server of
Orchestrator Core as a toolset. A direct connection gets only the raw tools of Core.

??? info "What a direct connection to Core skips"
    The Orchestrator Agent adds the following on top of the tools of Core. An agent that connects to the MCP server
    of Core directly has none of it:

    - The [plugins](../reference/plugins.md): the prompts that tell the model how to search, aggregate, look up,
      export and start workflows, plus the tables, charts and download links built from the results in code. See
      [Plugins and guards](../reference/runtime.md#plugins-and-guards).
    - The guards, such as the filter path guard that stops the model from guessing filter paths, and history
      trimming.
    - The [form-fill skill](../reference/runtime.md#workflow-forms), which walks a workflow form with the person and
      starts the workflow only after approval. Without it, the model can call the write tools of Core directly, see
      [Read tools and write tools](../reference/runtime.md#read-tools-and-write-tools).
    - Your [domain context](domain-context.md), set with `AGENT_DOMAIN_CONTEXT`.

    See [Capabilities extend the model and its tools](../architecture/index.md#capabilities) for how these fit
    together.

### 2. Write its instructions

Put the system prompt in `instructions.md` next to `agent.yaml`. Describe what the agent is for, what data it can
reach and how it should answer. Leave how each tool is used to the descriptions of the tools themselves:

```markdown
You are a read-only inventory agent for NetBox. You answer questions about the network
inventory: sites, devices, interfaces, cables, prefixes, IP addresses, VLANs and L2VPNs.
Your tools and their usage are described by the NetBox MCP server they come from.

## Rules

- You are strictly read-only; refuse anything that asks to change inventory.
- Prefer narrow, filtered queries over dumping whole tables.
- Report identifiers verbatim so callers can cross-reference.
- If a query returns nothing, say so plainly; never invent inventory.
```

### 3. Serve it over A2A

Add a service to your compose file that runs `docker-agent serve a2a` on the folder:

```yaml
services:
  inventory-agent:
    image: docker/docker-agent:1.125.0
    command: ["serve", "a2a", "/config/agent.yaml", "--listen", "inventory-agent:8080"]
    environment:
      OPENAI_API_KEY: ${AGENT_API_KEY:-}
      TELEMETRY_ENABLED: "false"
    volumes:
      - ./docker/inventory-agent:/config:ro
    depends_on:
      - netbox-mcp
```

The agent card at `/.well-known/agent-card.json` advertises the address that the agent listens on. docker-agent has no
separate setting for the public URL, so listen on the service name, which the other agents can reach. With
`0.0.0.0` or `localhost`, the card advertises `localhost` and the planner cannot reach the agent.

The card captures the address at startup. After you restart the agent, also restart the planner.

## Verify it works

Fetch the agent card and check the URL and the description in it:

```shell
curl http://inventory-agent:8080/.well-known/agent-card.json
```

The URL should be the service name, not `localhost`, and the description should say what the agent is for.

## Next steps

### Connect it to a planner

Add the agent card URL as a peer of a planner, next to the Orchestrator Agent, and tell the planner when to use it.
See [Connect another agent over A2A](a2a-planner.md). Then ask the planner a question that needs both agents, for
example *Which sites exist in NetBox, and how many nodes are in WFO?* The answer should combine both.

### Pass the identity of the user on

When the service requires authentication, forward the token of the user instead of a service account, so the agent can
only do what the user may do. See [Security and observability](security-observability.md). The example uses a
read-only NetBox token in the MCP server, which is fine for a demo but gives every user the same access.

### Benchmark it

Before you change prompts or models, add the questions to a benchmark, see
[Benchmark prompt and model changes](evals.md).

## Reference

- [Connect another service](connect-service.md)
- [Connect another agent over A2A](a2a-planner.md)
- [Endpoints and protocols: A2A](../reference/endpoints.md#a2a)
- [Running the agent in your own stack](../architecture/production.md), for deployment shapes and identity
