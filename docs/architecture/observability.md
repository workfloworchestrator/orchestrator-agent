# Tracing and observability

An answer from the agent is the end of a chain: the model reads the question, picks tools, calls Orchestrator Core,
reads the results and maybe picks more tools. When the answer is wrong, slow or expensive, the chat window shows only
the last link. Tracing helps inspect the instrumented model and tool operations that led to it.

To set it up, see [How to trace the agent](../guides/tracing.md).

<a id="why-traces-and-not-logs"></a>

## What traces add to logs

Logs tell you that something happened. For an agent the interesting question is usually *why* it happened: why the
model picked this tool, with these arguments, after reading that result. Each step depends on the steps before it, so
a line per event loses the thing you need.

A trace keeps instrumented steps together. Pydantic AI emits spans for its agent runs, model calls and model-driven
tool calls, with prompts, responses, available usage data, tool arguments and results. This helps show timing and
the relationship between those operations; it does not instrument every operation in the service automatically.

That structure is what makes traces useful for an agent:

- **Wrong answers.** A wrong tool, filter or identifier can produce a plausible but incorrect answer. A trace helps
  locate the step that caused it. Without that evidence, a change to a prompt, a plugin or
  [the domain context](../guides/domain-context.md) is a guess.
- **Cost.** Token counts per run and per model show which questions are expensive and what another model would save.
- **Latency.** A run that loops through the same tool, or reads large results, shows up as a long trace, and the
  timings show whether the time went to the model or to a tool.
- **What the model did.** A trace records instrumented tool use. Workflow forms also call Core directly from Python,
  outside the model's tool-call loop, so model/tool spans alone are not a complete audit of form validation or
  workflow starts. Core's recorded identity depends on the token used, see
  [Authentication and authorization](auth.md#outbound-who-the-agent-acts-as).

## Traces contain your data

The same detail that makes a trace useful makes it sensitive. A trace holds the user's question, the prompts and
everything the tools returned, which can include customer and network data. A tracing backend therefore deserves the
same scrutiny as the model provider: where does the data go, who can read it and how long is it kept. A hosted
backend is quick to start with; a self-hosted one keeps the data inside your infrastructure, at the cost of running
it. [The database stores the conversations](index.md#the-database-stores-the-conversations) raises the same question
about the agent's own storage.

<a id="langfuse-and-why-switching-is-easy"></a>

## Langfuse and other OpenTelemetry backends

The agent has built-in support for [Langfuse](https://langfuse.com/). Langfuse is built for LLM applications: it
shows model calls as conversations, adds up token costs per model and lets you search through runs. It is available
as a hosted service and can be self-hosted.

The agent is not tied to it. The agent does not create the spans itself: pydantic-ai, the framework the agent is
built on, emits them as standard [OpenTelemetry](https://opentelemetry.io/) spans. The Langfuse support is only a
small function that configures their export to Langfuse. Another OpenTelemetry backend needs its own exporter
configuration, such as the wrapper in [How to trace the agent](../guides/tracing.md).

!!! info "Only tested with Langfuse"
    The setup has only been tested and run with Langfuse. Other OpenTelemetry backends, such as Jaeger, Grafana Tempo,
    Arize Phoenix or Pydantic Logfire, should work because the spans are standard, but they have not been tested with
    the agent. How a backend shows model and tool spans also differs per backend.

## Traces and evals

The [evals](evaluation.md) are built on the same spans. They run the agent against a seeded Orchestrator Core,
collect the OpenTelemetry spans of each run and compare the tool calls in them with the expected trajectory. A trace
in production and a run in the eval suite describe the same thing: which tools the agent called, in which order and
with which arguments.

That shared shape links the two. Tracing shows what the agent does with real questions; the evals check that a change
does not break what already worked. A bad answer found in a trace is the best material for a new eval case, because
the trace already contains the question and the route the agent took. Writing down the route it should have taken
turns a one-off failure into a check that runs on every later change. See
[How to add an evaluation case](../guides/add-eval-case.md).

## What is not there yet

- **Complete workflow auditing.** The built-in integration instruments Pydantic AI; it does not add spans around all
  direct Core calls made by the form-fill skill, conversation persistence or the HTTP request lifecycle.
- **One trace across services.** The agent does not take over a `traceparent` header from its caller and does not
  pass one on to Orchestrator Core. The chat UI, the agent and Orchestrator Core each record their part on their own,
  and you link them by time and user. A gateway in front of the agents is one place to tie them together, see
  [Gateways](https://workfloworchestrator.org/learn/orchestrator-agent/production/#gateways).
- **Eval results in the tracing backend.** Langfuse can store scores and datasets, but the evals keep their results
  in the repository instead.
