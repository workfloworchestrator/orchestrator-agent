# Use your own LLM

!!! info "Goal"
    Run the agent on a model of your choice: hosted (OpenAI, Azure OpenAI), self-hosted (for example Ollama) or behind
    a proxy such as LiteLLM.

## Steps

Set `AGENT_MODEL` in `provider:model` format. Without any other setting, the model string goes straight to
[pydantic-ai](https://ai.pydantic.dev/models/), which reads the usual environment variable of the provider, such as
`OPENAI_API_KEY`.

For an OpenAI-compatible endpoint, set `AGENT_API_BASE` and `AGENT_API_KEY`:

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

The `azure:` prefix, or setting `AGENT_API_VERSION`, selects the Azure provider.

The agent calls tools, so choose a model that supports tool calling well. A small local model may pick the wrong
tools or arguments.

## Verify it works

Ask a few questions through [LibreChat](librechat.md) or the [demo clients](mcp-clients.md#verify-it-works). To compare
models on the same questions, use the golden set in [Benchmark prompt and model changes](evals.md).

## Reference

[Configuration: model](../reference/configuration.md#model)
