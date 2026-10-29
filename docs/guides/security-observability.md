# Security and observability

!!! info "Goal"
    Secure the agent with OIDC and trace what it does with [Langfuse](https://langfuse.com/).

!!! warning
    The agents profile of the example orchestrator is a demo stack and has no authentication. Do not expose it beyond
    your own machine.

## Steps

### Authenticate incoming requests

With `OAUTH2_ACTIVE=true`, which is the default, the agent authenticates requests with OIDC, using `oauth2_lib` like
Orchestrator Core. Set the details of your identity provider:

```shell
OAUTH2_ACTIVE=true
OIDC_BASE_URL=https://idp.example.com
OIDC_CONF_URL=https://idp.example.com/.well-known/openid-configuration
OAUTH2_RESOURCE_SERVER_ID=<client id>
OAUTH2_RESOURCE_SERVER_SECRET=<client secret>
```

`OAUTH2_ACTIVE=false` turns authentication off. Use it only for local development.

### Authenticate towards Orchestrator Core

The agent calls Orchestrator Core for its tools. `OAUTH2_OUTBOUND_ACTIVE` controls whether it authenticates those
calls with OAuth2 client credentials. When it is unset it follows `OAUTH2_ACTIVE`. Set `OAUTH2_TOKEN_URL` to the token
endpoint. Set it to `false` when Orchestrator Core accepts the calls without a token, independent of how clients
authenticate to the agent.

### Restrict where the agent listens

Bind published ports to a specific address and put the agent behind a reverse proxy that terminates TLS. The agent
and Orchestrator Core read the database, so keep `DATABASE_URI` credentials in a secret store and give the agent a
read-only database role where your deployment allows it.

### Trace with Langfuse

Install the `langfuse` extra and set:

```shell
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=<public key>
LANGFUSE_SECRET_KEY=<secret key>
LANGFUSE_HOST=https://cloud.langfuse.com
```

The agent then sends OpenTelemetry traces of every run: the model calls, and the tools called with their arguments.
If the extra is not installed or Langfuse cannot be configured, the agent starts without tracing. See the
[Langfuse documentation](https://langfuse.com/docs) for the dashboard.

### Trace with another OpenTelemetry backend

The traces are standard OpenTelemetry spans from pydantic-ai, so you can send them to any OTLP backend, such as Jaeger,
Grafana Tempo or Arize Phoenix. The agent only sets this up for Langfuse, so for another backend you do it yourself in
a small module that wraps the app. Install `opentelemetry-sdk` and `opentelemetry-exporter-otlp-proto-http`, then
create `traced_agent.py`:

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

Leave `LANGFUSE_ENABLED` off, point the exporter at your backend with the standard OpenTelemetry variables and start
this module instead of `orchestrator_agent.app`:

```shell
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_SERVICE_NAME=orchestrator-agent
uv run uvicorn traced_agent:app --port 8080
```

## Verify it works

- Call an endpoint without a token while `OAUTH2_ACTIVE=true`. The agent should refuse the request. `GET /health` is
  meant for monitoring.
- Ask a question and look for the trace in Langfuse.

## Reference

[Configuration: authentication and observability](../reference/configuration.md#authentication)
