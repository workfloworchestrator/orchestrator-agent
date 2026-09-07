# agent-evals

Golden-set benchmark for the WFO search agent, built on
[pydantic-evals](https://ai.pydantic.dev/evals/) with its built-in span-based
evaluators. The agent runs **in-process** — assembled by the same
`build_agent(agent_settings.create_model())` the server adapters use — against
a live orchestrator-core MCP server, so the benchmark measures agent behavior
only: which tools are called, in which order, with which arguments. Transport,
packaging and the A2A adapter are integration-test territory (`tests/`), not
eval territory.

## Setup

Start the backing services (a bare orchestrator-core with the MCP server and a
seeded database — the agent itself is not a container here):

```bash
docker compose -f docker-compose.yml up -d --wait
```

Then, from this folder (dependencies managed with [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
cp .env.example .env   # fill in AGENT_API_KEY (gitignored); pin AGENT_MODEL — scores
                       # are only comparable on the same model. Real env vars win over .env.
uv run python run_eval.py                    # full golden set
uv run python run_eval.py --cases fuzzy-search-nodes   # subset (smoke)
uv run python run_eval.py --fail-under 0.7   # CI gate: exit non-zero below the floor
```

## The dataset

`wfo_search_dataset.yaml` is the hand-curated golden set in pydantic-evals'
native schema. Each case pairs a user question with built-in evaluators:

| Evaluator | What it asserts |
| --- | --- |
| `TrajectoryMatch(order='exact')` | The tool-call sequence, byte-for-byte (binary). Used where exactly one route is right — including negative examples with an empty expected trajectory. |
| `TrajectoryMatch(order='in_order')` | F1 over the longest common subsequence — partial credit where a rule-consistent detour exists (e.g. `get_valid_operators` before a filtered search). |
| `ArgumentCorrectness(match_mode='subset')` | The prompt-influenced arguments of one call: entity type, identifiers passed through verbatim. |
| `MaxToolCalls(8)` (dataset-wide) | The prompt's call-budget discipline. |

Cases reference the deterministic seed data in `docker/core/seed.py`;
`metadata.failure_modes` tags the task shape, `metadata.curation_note` says why
the expected trajectory is the ideal one.

The file is kept in pydantic-evals' canonical serialized form: the
`yaml-language-server` header points at the generated
`wfo_search_dataset_schema.json` for IDE validation. After hand-editing cases,
round-trip the file to restore canonical form and refresh the schema:

```bash
uv run python -c "from pydantic_evals import Dataset; \
  Dataset[str, str].from_file('wfo_search_dataset.yaml').to_file('wfo_search_dataset.yaml')"
``` Tool calls are read from the
OpenTelemetry spans pydantic-ai emits (`Agent.instrument_all()` +
an SDK tracer provider, both set up by `run_eval.py`).

## Deliberately not included (yet)

Kept out until needed, as custom evaluators when they come:

- **Prompt-rule compliance** (no bare search, discover-before-filter, export
  only with a query) — deterministic checks derived from this agent's prompt.
- **Alternate golden routes** — `in_order` covers today's cases; a
  best-of-routes evaluator is ~15 lines when a case truly needs it.
- **Paired-bootstrap significance testing** — the statistically rigorous
  comparator; the committed-baseline comparison below uses a plain tolerance
  instead until noise proves it insufficient.
- **Multi-turn sessions** — a case's task function can run several turns
  in-process when a scenario requires it.

## Baseline

**`baseline_report.json` is the committed baseline**: a full report of a
blessed run, serialized with pydantic-evals' own `EvaluationReportAdapter`.
Every benchmark run can compare itself against it per case:

```bash
uv run python run_eval.py --baseline baseline_report.json   # fail on per-case regression
uv run python run_eval.py --save-report baseline_report.json  # bless a new baseline
```

A score that drops more than `--tolerance` (default 0.2) below the baseline,
or an assertion that flips from passing to failing, fails the run and names
the case. Improvements never fail. Re-bless the baseline deliberately — in the
same MR as the change that moves the numbers, on the pinned model — so a
baseline change is always a reviewed diff.

## CI

`.github/workflows/eval.yml` runs the benchmark on demand
(`workflow_dispatch`), nightly, and on PRs labeled `eval`: compose up the
backing services, run the golden set in-process from the branch's working
tree, gate on the committed baseline plus the `--fail-under` floor. Requires
the `EVAL_AGENT_API_KEY` repository secret (and optionally an
`EVAL_AGENT_MODEL` variable, default `openai:gpt-4o`).
