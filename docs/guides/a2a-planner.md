# Connect another agent over A2A

To see a planner that combines the Orchestrator Agent with a second agent, follow the
[tutorial](https://workfloworchestrator.org/learn/orchestrator-agent/tutorials/example-orchestrator/).
This guide covers connecting the agent to your own setup.

## Steps

1. Set `BASE_URL` of the Orchestrator Agent to an address that the other agent can reach. The agent card advertises
   this URL, and the caller sends its messages there. Inside Docker or Kubernetes this is the service name, not
   `localhost`.
2. Check that the agent card is served at that address and shows the right URL. The card lists one skill per
   [plugin](../reference/plugins.md):

    ```shell
    curl http://<agent>:8080/.well-known/agent-card.json
    ```

3. Add the Orchestrator Agent as a remote A2A agent of the calling agent, using the base URL of the card.

    === "docker-agent"

        In a [docker-agent](https://github.com/docker/docker-agent) `agent.yaml`, add an `a2a` toolset:

        ```yaml
        toolsets:
          - type: a2a
            name: wfo
            url: http://wfo-agent:8080/
            allow_private_ips: true  # needed for addresses inside a compose network
            timeout: 180
        ```

    === "kagent"

        In [kagent](https://kagent.dev/docs/kagent), add the Orchestrator Agent as an agent tool of the calling agent.

    === "Other runtimes"

        Use the A2A client of your runtime, for example the
        [A2A Python SDK](https://github.com/a2aproject/a2a-python), with the card URL.

4. Tell the calling agent in its instructions when to use the Orchestrator Agent, for example for questions about
   subscriptions, products, workflows and processes.
5. Restart the calling agent whenever you restart the Orchestrator Agent or change its plugins. Most runtimes read the
   card only at startup.

## Workflow forms

The agent speaks A2A protocol v1.0. Callers on 0.3 are still served, but they cannot fill in workflow forms. Forms
need the [kagent human-in-the-loop extension](../reference/runtime.md#workflow-forms), so that the calling agent can
relay the questions and the approval to the person. The calling agent must request the extension in its call, as
kagent 1.0 and later does. Without it, the calling agent can ask questions but cannot start workflows.

## Verify it works

Ask the calling agent a question about WFO, for example *How many active subscriptions are there?*. The answer
should come from the Orchestrator Agent.

## Reference

- [Endpoints and protocols: A2A](../reference/endpoints.md#a2a)
- [Configuration: `BASE_URL`](../reference/configuration.md)
- [Plugins](../reference/plugins.md)
- [Add your own tools, services and agents](own-tools.md), to build the second agent
