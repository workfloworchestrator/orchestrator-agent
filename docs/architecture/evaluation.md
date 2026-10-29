# How we evaluate the agent

The evals of the agent answer one question: when a user asks something, does the agent call the right tools, in the
right order, with the right arguments? How to run them is described in [Evals](../reference/evals.md) and
[Benchmark prompt and model changes](../guides/evals.md).

## What we measure

We measure the **trajectory** and not the wording of the answer. The agent runs in-process against a live
Orchestrator Core with a seeded database, and the tool calls are read from the OpenTelemetry spans of pydantic-ai.
This has a few consequences:

- A change of prompt, model or plugin shows up as a different set of tool calls, which is a much more stable signal
  than a judgement of free text.
- The benchmark measures the agent only. Transport, packaging and the protocol adapters are covered by the normal
  tests, and not by the evals.
- Some cases have exactly one right route, so they must match exactly, including negative examples that expect no
  tool call at all. Other cases allow a detour that is consistent with the rules, so they earn partial credit with an
  in-order match. Arguments are checked for the parts that the prompt influences, such as the entity type and
  identifiers that are passed through verbatim. A call budget keeps the agent from wandering.

## Why golden sets are hand-curated

Each case in the dataset is written by a person who knows the seed data and the rules of the prompt. A note with each
case says why its expected trajectory is the ideal one, and the cases are tagged with the failure mode they test for.
A generated set would reflect what the agent does today, so it could not tell when the agent does the wrong thing.

The committed baseline report records the result of a run that was reviewed. A change that moves scores must bless a
new baseline in the same change, so every change of the baseline is a diff that somebody reviewed.

## What is not covered yet

- **Prompt-rule compliance**, such as no search without a filter discovery, and export only when a query exists, as
  deterministic checks derived from the prompt.
- **Alternate golden routes.** A case has one expected route and a detour earns partial credit.
- **Significance testing.** Runs are compared against the baseline with a plain tolerance, until noise proves it
  insufficient.
- **Multi-turn sessions**, such as a follow-up on an earlier result.
- **Answer quality.** The final text is not scored.
