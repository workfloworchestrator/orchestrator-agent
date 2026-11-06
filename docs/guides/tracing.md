# How to trace the agent

This guide shows you how to send Pydantic AI's model and tool-call spans to [Langfuse](https://langfuse.com/) or
another OpenTelemetry backend. To learn what traces show and how they relate to evals, see
[Tracing and observability](../architecture/observability.md).

!!! warning "Traces contain your data"
    A trace holds the user's question, the prompts and everything the tools returned. If that data must stay in your
    infrastructure, [self-host Langfuse](https://langfuse.com/self-hosting) and point `LANGFUSE_HOST` at it.

## Steps

=== "Langfuse"

    1. In the agent checkout, install its optional integration:

        ```shell
        uv sync --locked --extra langfuse
        ```

        For the repository's Dockerfile, build with `--build-arg 'UV_EXTRA_ARGS=--extra langfuse'` to include the extra.

    2. Add these settings to the agent's `.env` or container environment:

        ```shell
        LANGFUSE_ENABLED=true
        LANGFUSE_PUBLIC_KEY=<public key>
        LANGFUSE_SECRET_KEY=<secret key>
        LANGFUSE_HOST=https://cloud.langfuse.com
        ```

    3. Restart the source process with the extra enabled and `.env` loaded:

        ```shell
        uv run --locked --extra langfuse --env-file .env \
          uvicorn orchestrator_agent.app:app --host 127.0.0.1 --port 8080
        ```

        For a container, recreate it with the image containing the extra and the updated environment.

=== "Another OpenTelemetry backend"

    !!! note
        The setup has only been tested and run with Langfuse. Other backends should work, but are untested.

    1. Create `traced_agent.py` in the agent checkout, alongside `pyproject.toml`:

        ```python
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from pydantic_ai import Agent

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        Agent.instrument_all()

        from orchestrator_agent.app import app  # noqa: E402
        ```

    2. Set `LANGFUSE_ENABLED=false` in `.env`. Launch the wrapper with its exporter dependencies and configuration
       passed into the process:

        ```shell
        OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
        OTEL_SERVICE_NAME=orchestrator-agent \
        uv run --with opentelemetry-sdk --with opentelemetry-exporter-otlp-proto-http --env-file .env \
          uvicorn traced_agent:app --host 127.0.0.1 --port 8080
        ```

## Verify it works

=== "Langfuse"

    1. Check the agent's startup log for `Langfuse tracing enabled`.
        - If it logs `Langfuse auth check failed`, check the keys and `LANGFUSE_HOST`.
        - If it logs that the `langfuse` extra is not installed, install it and restart.
    2. Ask the agent a question, for example "How many active subscriptions are there?".
    3. Open your project's traces view in Langfuse and look for the new model run. Allow time for batched spans to
       arrive.
    4. Open the trace. It should show the model calls and each tool call, with the arguments the agent passed and
       what Orchestrator Core returned.

=== "Another OpenTelemetry backend"

    1. Ask the agent a question, for example "How many active subscriptions are there?".
    2. In your backend, search for traces of the service `orchestrator-agent` (the `OTEL_SERVICE_NAME` you set).
    3. Open the trace and check the instrumented model calls and tools.

    If no trace arrives, check that `OTEL_EXPORTER_OTLP_ENDPOINT` is reachable from the agent and that you started
    `traced_agent:app`, not `orchestrator_agent.app`.

## Reference

- [Configuration: observability](../reference/configuration.md#observability)
- [Tracing and observability](../architecture/observability.md)
- [Langfuse documentation](https://langfuse.com/docs)
- [Langfuse's Pydantic AI integration](https://langfuse.com/integrations/frameworks/pydantic-ai)
