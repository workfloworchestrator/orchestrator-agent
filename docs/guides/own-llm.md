# How to change the language model

This guide shows you how to run the agent on a model of your choice: hosted, self-hosted or behind a proxy. To weigh
hosted against self-hosted, see [The language model interprets the request](../explanation/index.md#where-your-data-goes).

## Steps

### 1. Pick a model that calls tools well

The agent works by calling tools, so pick a model with good tool calling. A small local model may pick the wrong
tools or arguments.

### 2. Benchmark it before you switch

Run the golden set on your current model and on the new one, and compare the scores. See
[How to benchmark prompt and model changes](evals.md).

### 3. Configure the model

Edit the `.env` loaded by your source launch, or the environment supplied to the container. Remove stale
`AGENT_API_BASE`, `AGENT_API_KEY` and `AGENT_API_VERSION` entries before selecting a different provider below.
Also unset shell overrides for values you want `.env` to supply. Do not leave unused entries as empty assignments.

=== "Hosted provider"

    Leave all three `AGENT_API_*` overrides unset. Set `AGENT_MODEL` in `provider:model` format and the provider's
    usual API key variable, for example:

    ```shell
    AGENT_MODEL=openai:gpt-4o
    OPENAI_API_KEY=sk-...
    ```

    For a native provider other than OpenAI, install the corresponding Pydantic AI extra. For example, to run a
    source checkout configured for Anthropic:

    ```shell
    uv run --with 'pydantic-ai-slim[anthropic]>=2,<3' --env-file .env \
      uvicorn orchestrator_agent.app:app --host 127.0.0.1 --port 8080
    ```

    Use that command on subsequent launches too, or declare the dependency in your own fork. For a container,
    include the provider dependency when building its image. See the
    [Pydantic AI model documentation](https://pydantic.dev/docs/ai/models/overview/) for provider extras and variables.

=== "Ollama"

    Pull a model with tool support, such as `ollama pull llama3.1`, and configure:

    ```shell
    AGENT_MODEL=openai:llama3.1
    AGENT_API_BASE=http://localhost:11434/v1
    ```

    Leave `AGENT_API_VERSION` unset. Use an endpoint reachable from the agent; `localhost` inside a container refers
    to that container. See [Ollama's model reference](https://ollama.com/library/llama3.1) for tool support.

=== "LiteLLM or another proxy"

    Leave `AGENT_API_VERSION` unset, then configure:

    ```shell
    AGENT_MODEL=openai:gpt-4o
    AGENT_API_BASE=https://my-proxy.example.com/v1
    AGENT_API_KEY=sk-custom-key
    ```

=== "Azure OpenAI"

    ```shell
    AGENT_MODEL=azure:gpt-4o
    AGENT_API_BASE=https://my-resource.openai.azure.com/
    AGENT_API_KEY=azure-api-key
    AGENT_API_VERSION=2024-12-01-preview
    ```

### 4. Restart the agent

For a source launch, use `uv run --env-file .env uvicorn orchestrator_agent.app:app --host 127.0.0.1 --port 8080`,
adding the provider dependency as above when needed. Recreate the container if you changed its environment, and
rebuild it if you added a dependency. Use your deployment's existing listen address and port if different.

## Verify it works

1. Ask a question that needs a tool, for example *How many subscriptions are there per status?*, through
   [LibreChat](librechat.md) or an [MCP client](mcp-clients.md#verify-it-works).
2. The answer should contain the counts. If the agent answers without them, or the request fails, the model may not
   support tool calling, or `AGENT_API_BASE` or the key is wrong.

## Reference

- [Configuration: model](../reference/configuration.md#model)
