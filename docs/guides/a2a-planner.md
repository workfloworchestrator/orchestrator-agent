# Use the agent from an A2A planner

!!! info "Goal"
    Let a planner agent delegate questions about WFO to the Orchestrator Agent over A2A, next to other agents such as
    an inventory agent for NetBox. The planner can then answer a question that spans several systems.

## Steps

1. Check that the agent card is served. It lists one skill per [plugin](../reference/plugins.md):

    ```shell
    curl http://localhost:8080/.well-known/agent-card.json
    ```

2. Set `BASE_URL` of the agent to the address that the planner can reach. The card advertises this URL, so a wrong
   value makes the planner call an address it cannot reach.
3. Add the agent as a peer of the planner, using the URL of the agent card. In a [docker-agent](https://github.com/docker/docker-agent)
   planner this is a remote A2A toolset. In kagent it is a remote agent tool.
4. Tell the planner in its instructions when to use the agent, for example for questions about subscriptions,
   products, workflows and processes.

The agent speaks A2A protocol v1.0. Callers on 0.3 are still served, but they cannot fill in workflow forms. Forms need
the [kagent human-in-the-loop extension](../architecture/index.md#workflow-form-fill), so that the planner can relay
the questions and the approval to the person.

### Reference setup

The agents profile of the [example orchestrator](https://github.com/workfloworchestrator/example-orchestrator) is a
working setup of this kind. Its `docker-compose.agents.yml` runs:

| Service | Role |
| --- | --- |
| `wfo-agent` | The Orchestrator Agent, from the published container image |
| `netbox-mcp` | An MCP server for NetBox |
| `inventory-agent` | A docker-agent agent that answers questions about NetBox through `netbox-mcp` |
| `planner-agent` | A docker-agent agent that delegates to `wfo-agent` and `inventory-agent` over A2A |
| `a2a-proxy` | Makes the planner available as an OpenAI-compatible endpoint |
| `librechat` | The chat frontend |

A docker-agent agent is declared in an `agent.yaml` with its model, instructions and toolsets, and served with
`docker-agent serve a2a <agent.yaml> --listen <service>:8080`. Listen on the service name so that the card advertises
an address that the peers can reach. Restart the planner after you restart an agent it uses, because the card is
read at startup.

## Verify it works

Ask the planner a question that needs both systems, for example *Which sites exist in NetBox, and how many nodes are
in WFO?*. The answer should combine both. Questions that only concern WFO should be answered by the agent alone.

## Reference

- [Endpoints and protocols](../reference/endpoints.md#a2a)
- [Plugins](../reference/plugins.md)
