# How to run the agent with your Orchestrator Core

Run the agent next to your own Orchestrator Core and ask it a first question. To try it on a ready-made setup
instead, follow the [tutorial on the example orchestrator](https://workfloworchestrator.org/learn/orchestrator-agent/tutorials/example-orchestrator/).

## Prerequisites

- An Orchestrator Core with its MCP server enabled (`MCP_ENABLED=True`, installed with the `mcp` extra). See the
  [MCP server reference](https://workfloworchestrator.org/orchestrator-core/reference-docs/mcp/).
- Network access to its PostgreSQL database, with the Core migrations applied, including the agent run and
  snapshot tables. Use a Core version compatible with the agent release you run; startup also checks the live MCP
  tool names. See the
  [Core setup documentation](https://workfloworchestrator.org/orchestrator-core/getting-started/base/).
- The Core search index populated for the entity types you want to query. Follow
  [Building and refreshing the index](https://workfloworchestrator.org/orchestrator-core/reference-docs/ai-search/#building-and-refreshing-the-index).
- An API key for an LLM, or an OpenAI-compatible endpoint of your own. See
  [How to change the language model](guides/own-llm.md).
- **For Source:** Git, [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13.
- **For Docker:** Docker with support for Linux containers.
- **Optional:** embeddings enabled in Orchestrator Core for semantic and hybrid search. See
  [Running AI search](https://workfloworchestrator.org/orchestrator-core/reference-docs/ai-search/#running-it).

!!! tip "Why enable embeddings"
    The search index and embeddings live in Orchestrator Core, so configure them there. The agent works without
    embeddings, using fuzzy keyword matching and structured filters.

    With embeddings enabled, searches can also match meaning. For example, a question about customer connectivity
    can find subscriptions whose descriptions use different words. See
    [How a question becomes a search](explanation/index.md#how-a-question-becomes-a-search).

Use the **Source** or **Docker** tab in each of the three steps below, then follow [Verify it works](#verify-it-works).
Both options use port 8080 for the agent. The examples connect to an unauthenticated Core on host port 8081; replace the database
credentials and model key with your own values.

If Core requires a token, configure outgoing authentication using
[How to configure authentication and authorization](guides/secure-agent.md#3-authenticate-towards-orchestrator-core)
before starting. Keep `OAUTH2_ACTIVE=false` for a local try-out only.

## 1. Prepare the files

=== "Source"

    Clone the repository, copy its environment example and install the dependencies:

    ```shell
    git clone https://github.com/workfloworchestrator/orchestrator-agent.git
    cd orchestrator-agent
    cp .env.example .env
    uv sync
    ```

=== "Docker"

    Create a directory for your configuration:

    ```shell
    mkdir -p orchestrator-agent-docker
    cd orchestrator-agent-docker
    ```

    Create `.env` in this directory and fill it in during step 2. The container uses the published image from the
    [GitHub Container Registry](https://github.com/workfloworchestrator/orchestrator-agent/pkgs/container/orchestrator-agent).

## 2. Configure the connections

Both services default to port 8080. When they share one host, keep the agent on 8080 and give Core another port.
These examples use Core on 8081. Services on separate hosts or container addresses can each use 8080.
All settings are listed in [Configuration](reference/configuration.md).

=== "Source"

    Edit `.env` with the database and Core addresses reachable from your machine:

    ```shell
    DATABASE_URI=postgresql+psycopg://user:pass@localhost/orchestrator
    WFO_CORE_MCP_URL=http://localhost:8081/mcp
    BASE_URL=http://localhost:8080
    AGENT_MODEL=openai:gpt-4o
    AGENT_API_KEY=<your key>
    OAUTH2_ACTIVE=false
    ```

=== "Docker"

    Fill in `.env` with addresses reachable from the container. For Core and PostgreSQL running on your host, use:

    ```shell
    DATABASE_URI=postgresql+psycopg://user:pass@host.docker.internal/orchestrator
    WFO_CORE_MCP_URL=http://host.docker.internal:8081/mcp
    BASE_URL=http://localhost:8080
    AGENT_MODEL=openai:gpt-4o
    AGENT_API_KEY=<your key>
    OAUTH2_ACTIVE=false
    ```

    Core and PostgreSQL must accept connections from the container network. The run command in step 3 maps
    `host.docker.internal` to the host gateway; `localhost` inside the container refers to the agent container.

    If Core and PostgreSQL are in containers on an existing Docker network, use their service names and internal
    ports in `.env`, and add `--network <network-name>` to the run command. See
    [Docker networking](https://docs.docker.com/engine/network/drivers/bridge/#differences-between-user-defined-bridges-and-the-default-bridge).

## 3. Start the agent

=== "Source"

    Load `.env` into the process explicitly:

    ```shell
    uv run --env-file .env uvicorn orchestrator_agent.app:app --host 127.0.0.1 --port 8080
    ```

=== "Docker"

    ```shell
    docker run --rm --name orchestrator-agent \
      --env-file .env \
      --add-host host.docker.internal:host-gateway \
      --publish 127.0.0.1:8080:8080 \
      ghcr.io/workfloworchestrator/orchestrator-agent:latest
    ```

    The image listens on port 8080, and this command publishes it on your host's loopback address. Use a release tag
    from the registry instead of `latest` when you need to keep the image version fixed.

Continue with [Verify it works](#verify-it-works) in another terminal while the process or container runs.

## Verify it works

At startup the agent checks the tools of the Orchestrator Core MCP server. If a tool it needs is missing, it stops
with an error that names the tool. A connection failure skips that check with a warning, so a successful health
check alone does not establish that Core can be queried.

Check that the agent is up:

```shell
curl --fail --location http://localhost:8080/health
```

Send a question to the OpenAI-compatible endpoint:

```shell
curl --fail http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "wfo", "messages": [{"role": "user", "content": "How many subscriptions are there per status?"}]}'
```

Confirm that the answer contains counts from your own inventory. If startup reports missing tables, apply the Core
migrations. If the query fails or returns no indexed data, check the Core connection and index setup above.

## Next steps

- [Connect LibreChat](guides/librechat.md) to chat with the agent.
- [Enable embeddings](https://workfloworchestrator.org/orchestrator-core/reference-docs/ai-search/#running-it) in
  Orchestrator Core for semantic and hybrid search if you haven't already done so.
- [Add domain context](guides/domain-context.md) with your identifiers and terminology.
- [Configure authentication and authorization](guides/secure-agent.md) before exposing the service to other users.
- Learn about [endpoints and protocols](reference/endpoints.md) to connect your own clients and tools.
- Before more clients, other agents or machine callers use the agent, read
  [Gateways](https://workfloworchestrator.org/learn/orchestrator-agent/production/#gateways). The agent checks who calls it, but it sets no limits or budgets per
  caller, and the other paths to your tools need the same checks.
