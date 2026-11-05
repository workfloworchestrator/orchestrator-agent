# Tracing and observability

An answer from the agent is the end of a chain: the model reads the question, picks tools, calls Orchestrator Core,
reads the results and maybe picks more tools. When the answer is wrong, slow or expensive, the chat window shows only
the last link. Tracing records the whole chain.

To set it up, see [How to trace the agent](../guides/tracing.md).

## Why traces and not logs

Logs tell you that something happened. For an agent the interesting question is usually *why* it happened: why the
model picked this tool, with these arguments, after reading that result. Each step depends on the steps before it, so
a line per event loses the thing you need.

A trace keeps the steps together. Each run of the agent becomes one trace, with a span for every model call (the
prompt, the answer, the tokens it cost) and every tool call (the tool, its arguments and what Orchestrator Core
returned), nested in the order they happened and timed.

That structure is what makes traces useful for an agent:

- **Wrong answers.** Most wrong answers come from a wrong tool, a wrong filter or a wrong identifier, not from bad
  wording. A trace shows the step where the run went wrong. Without it, a change to a prompt, a plugin or
  [the domain context](../guides/domain-context.md) is a guess.
- **Cost.** Token counts per run and per model show which questions are expensive and what another model would save.
- **Latency.** A run that loops through the same tool, or reads large results, shows up as a long trace, and the
  timings show whether the time went to the model or to a tool.
- **What was done.** A trace records which tools ran for a question, including workflows that were started. Who
  started them is recorded by Orchestrator Core, see [Authentication and authorization](auth.md#outbound-who-the-agent-acts-as).

## Traces contain your data

The same detail that makes a trace useful makes it sensitive. A trace holds the user's question, the prompts and
everything the tools returned, which can include customer and network data. A tracing backend therefore deserves the
same scrutiny as the model provider: where does the data go, who can read it and how long is it kept. A hosted
backend is quick to start with; a self-hosted one keeps the data inside your infrastructure, at the cost of running
it. [The database stores the conversations](index.md#the-database-stores-the-conversations) raises the same question
about the agent's own storage.

## Langfuse, and why switching is easy

The agent has built-in support for [Langfuse](https://langfuse.com/). Langfuse is built for LLM applications: it
shows model calls as conversations, adds up token costs per model and lets you search through runs. It is available
as a hosted service and can be self-hosted.

The agent is not tied to it. The agent does not create the spans itself: pydantic-ai, the framework the agent is
built on, emits them as standard [OpenTelemetry](https://opentelemetry.io/) spans. The Langfuse support is only a
small function that sends those spans to Langfuse. Any backend that accepts OpenTelemetry can receive the same spans,
without changes to the agent. General-purpose backends such as Jaeger or Grafana Tempo fit well if your organisation
already runs one for its other services; LLM-specific ones such as Langfuse or Arize Phoenix are better at showing
prompts, answers and costs.

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
[Benchmark prompt and model changes](../guides/evals.md).

## What is not there yet

- **One trace across services.** The agent does not take over a `traceparent` header from its caller and does not
  pass one on to Orchestrator Core. The chat UI, the agent and Orchestrator Core each record their part on their own,
  and you link them by time and user. A gateway in front of the agents is one place to tie them together, see
  [Gateways](production.md#gateways).
- **Eval results in the tracing backend.** Langfuse can store scores and datasets, but the evals keep their results
  in the repository instead.
