# How to use your own LLM

This guide shows you how to run the agent on a model of your choice: hosted, self-hosted or behind a proxy. To weigh
hosted against self-hosted, see [The language model interprets the request](../architecture/index.md#where-your-data-goes).

## Steps

### 1. Pick a model that calls tools well

The agent works by calling tools, so pick a model with good tool calling. A small local model may pick the wrong
tools or arguments.

### 2. Benchmark it before you switch

Run the golden set on your current model and on the new one, and compare the scores. See
[Benchmark prompt and model changes](evals.md).

### 3. Configure the model

=== "Hosted provider"

    Set `AGENT_MODEL` in `provider:model` format and the provider's usual API key variable:

    ```shell
    AGENT_MODEL=openai:gpt-4o
    OPENAI_API_KEY=sk-...
    ```

    See the [pydantic-ai model list](https://ai.pydantic.dev/models/) for the providers and their variables.

=== "Ollama"

    ```shell
    AGENT_MODEL=openai:llama3
    AGENT_API_BASE=http://localhost:11434/v1
    ```

=== "LiteLLM or another proxy"

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

## Verify it works

1. Ask a question that needs a tool, for example *How many subscriptions are there per status?*, through
   [LibreChat](librechat.md) or an [MCP client](mcp-clients.md#verify-it-works).
2. The answer should contain the counts. If the agent answers without them, or the request fails, the model may not
   support tool calling, or `AGENT_API_BASE` or the key is wrong.

## Reference

- [Configuration: model](../reference/configuration.md#model)
