# Connect another service

The Orchestrator Agent connects to one MCP server: the one of Orchestrator Core, set with `WFO_CORE_MCP_URL`. It does
not load the MCP servers of other systems. To answer questions that need data of another system, such as NetBox, a
monitoring system or a CRM, choose one of two ways.

| When the data | Do this |
| --- | --- |
| Belongs to the subscriptions in WFO, for example the contract or the monitoring status of a subscription | Add a route to your orchestrator and [expose it as a tool](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/#extending). The agent then answers from one place. |
| Lives in a system of its own | Give the system an agent of its own and let a planner combine the answers, as described below. This keeps each agent small, with its own tools and instructions. |


## Steps

The example connects NetBox, as the `agents` profile of the
[example orchestrator](https://github.com/workfloworchestrator/example-orchestrator) does.

### 1. Run an MCP server for the service

Many systems have an MCP server, see the [MCP server list](https://github.com/modelcontextprotocol/servers). For NetBox,
run [netbox-mcp](https://github.com/netboxlabs/netbox-mcp-server) over streamable HTTP:

```yaml
services:
  netbox-mcp:
    image: docker.io/netboxlabs/netbox-mcp-server:latest
    environment:
      NETBOX_URL: http://netbox:8080/
      NETBOX_TOKEN: <read-only NetBox API token>
      TRANSPORT: http
      HOST: 0.0.0.0
      PORT: "8000"
```

The tools are then served at `http://netbox-mcp:8000/mcp`. Give the server a token that can only read, so the agent
cannot change the service.

### 2. Create an agent with the MCP server as its toolset

Follow [Create your own agent](create-agent.md). Its example agent uses `http://netbox-mcp:8000/mcp` as its toolset and
is served over A2A as `inventory-agent`.

### 3. Connect the agent to a planner

Add the new agent and the Orchestrator Agent as peers of a planner, see
[Connect another agent over A2A](a2a-planner.md). In the example orchestrator, `planner-agent` has both as `a2a`
toolsets and decides from their agent cards which one to ask.

## Verify it works

Ask the planner a question that needs both systems, for example *Which sites exist in NetBox, and how many nodes are
in WFO?* The answer should combine data from NetBox and from WFO.

## Reference

- [MCP server of Orchestrator Core: extending](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/#extending)
- [Create your own agent](create-agent.md)
- [Connect another agent over A2A](a2a-planner.md)
- [Teach the agent your domain](domain-context.md)
