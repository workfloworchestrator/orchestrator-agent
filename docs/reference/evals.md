# Evaluation benchmark

The `evals/` folder of the agent repository holds a golden-set benchmark built on
[pydantic-evals](https://ai.pydantic.dev/evals/). The agent runs in-process against a live Orchestrator Core MCP
server, so the benchmark measures agent behaviour only: which tools are called, in which order, with which
arguments, and what the last call returned. See [Understanding agent evaluations](../explanation/evaluation.md) for the reasoning and
[How to benchmark prompt and model changes](../guides/evals.md) for comparisons, or
[How to add an evaluation case](../guides/add-eval-case.md) to extend the dataset.

## Dataset

`wfo_search_dataset.yaml` is a hand-curated set of cases in the native pydantic-evals schema. Each case pairs a
user question with evaluators. Cases refer to the deterministic seed data in `docker/core/seed.py`.
`metadata.failure_modes` tags the shape of the task and `metadata.curation_note` explains why the expected
trajectory is the ideal one.

The file uses the canonical serialized form, and `wfo_search_dataset_schema.json` provides IDE validation.
[How to add an evaluation case](../guides/add-eval-case.md#3-validate-and-run-the-case) includes the serialization
command used after editing cases.

## Evaluators

| Evaluator | What it asserts |
| --- | --- |
| `TrajectoryMatch(order='exact')` | The tool-call sequence, exactly. Used where one route is right, including negative examples with an empty expected trajectory |
| `TrajectoryMatch(order='in_order')` | F1 over the longest common subsequence, so a rule-consistent detour earns partial credit |
| `ArgumentCorrectness(match_mode='subset')` | Expected top-level arguments of one call; expected nested values must match in full. The default occurrence is the first successful call |
| `ToolResultContains` | The outcome: the last successful call to a tool returned every expected string and none of the excluded ones. Defined in `evals/evaluators.py`; strings are matched against the result as compact JSON |
| `MaxToolCalls(8)` | Dataset-wide budget of locally executed tool-call attempts, including failed attempts by default |

The route evaluators judge how the agent got there; `ToolResultContains` judges whether it got the right data. They
can disagree: a model that takes a different route to the correct rows scores low on route and full on outcome,
and a model that crashes before an answer scores zero on all of them. Read the two together.

Tool calls are read from the OpenTelemetry spans that pydantic-ai emits.
`TrajectoryMatch` and `ArgumentCorrectness` exclude failed tool-call attempts by default; `MaxToolCalls` includes
them. A retry can therefore consume the call budget without appearing in the matched trajectory.

## Commands

Run from `evals/` after completing the [guide setup](../guides/evals.md#1-prepare-the-evaluation-environment).

| Command | Effect |
| --- | --- |
| `uv run python run_eval.py` | Run the full golden set |
| `uv run python run_eval.py --cases <name>` | Run the named case; accepts comma-separated names |
| `uv run python run_eval.py --models <entry>` | Run only these entries of `eval-models.json`; accepts comma-separated names |
| `uv run python run_eval.py --repeat 5` | Run every case five times; the summary shows per-case averages |
| `uv run python run_eval.py --fail-under 0.7` | Exit non-zero when the aggregate is below the floor |
| `uv run python run_eval.py --baseline baseline_report.json` | Fail when a case regresses against the committed baseline |
| `uv run python run_eval.py --save-report baseline_report.json` | Save a report; review it before accepting it as a baseline |
| `uv run python run_eval.py --summary-file <path>` | Append the result table and any gate failure to a file as Markdown |

## Reports

Reports are printed to the terminal; JSON is written only with `--save-report`. The terminal includes inputs and
durations but hides final outputs. Saved reports retain outputs and evaluator reasons, not full tool transcripts.

Each completed case contains:

| Field | Contents |
| --- | --- |
| `scores` | Numeric evaluator results; `TrajectoryMatch` ranges from 0 to 1, with 1 indicating a full match in its configured mode |
| `assertions` | Separate boolean checks, including selected arguments, the returned data (`ToolResultContains`) and the call budget |
| `reason` within an evaluator result | The evaluator's explanation, when provided |
| `output` | The final agent answer; not graded by the bundled cases, whose outcome check reads the tool result instead |

Execution failures are recorded separately in `failures`. A completed case can have failed assertions or low scores.
The [product-count example](../guides/add-eval-case.md#2-add-the-case-to-the-dataset) checks `aggregate` and `entity_type: PRODUCT`, but does not verify the returned
count or answer against the three seeded products. The benchmark records tool-call spans locally for grading; it does
not automatically publish its reports to Langfuse.

## Exit status

Without gates, failed assertions and low scores do not cause a non-zero exit. Case execution failures do. With
`--fail-under`, the runner averages the per-score averages together with the assertion pass rate (when present).
This aggregate is not a task-success percentage, and the floor does not require every assertion to pass.

The report metadata records the model entry, its model settings and the name of the variable that holds its endpoint,
never the URL itself. Only settings that
cannot carry a credential are recorded with their values (`temperature`, `top_p`, `max_tokens`, `thinking`, `seed`,
`timeout`, `parallel_tool_calls`); any other key, such as `extra_headers`, is listed by name only, because saved reports
can be committed as baselines. It does not record the Core image, the domain context or the git revision; note those
yourself next to a saved report.

## Settings

Real environment variables override `evals/.env`. The harness reads `WFO_CORE_MCP_URL`, `OAUTH2_ACTIVE` and
`DATASET_FILE`, plus the variables the entries of `eval-models.json` name.

Each entry of `eval-models.json` is one model:

| Field | Meaning |
|---|---|
| `model` | `provider:model`, or the bare model name on an OpenAI-compatible endpoint |
| `api_key` | Name of the variable that holds the key. The file is committed, so never the key itself |
| `api_base` | Optional. Name of the variable that holds the endpoint's URL; without it the provider's own URL is used |
| `settings` | Optional. Model settings sent with every request to this model, in the form of `AGENT_MODEL_SETTINGS` |

```json
{"model": "Qwen3.8-27B-FP8", "api_key": "LLMS_API_KEY", "api_base": "LLMS_API_BASE",
 "settings": {"temperature": 0.6, "top_p": 0.95}}
```

An entry with an unset variable shows `no key` in the summary and is skipped. An entry with `api_base` evaluates an
open-weight or self-hosted model behind an OpenAI-compatible endpoint, such as a LiteLLM proxy. Every model except an
`anthropic:` one is built by the agent's own settings, so what is measured is what the server would send.
Baselines are per model: with `--save-report baseline_report.json` the model is in the file name.

## Baseline

`baseline_report.json` is a full report of a blessed run. A score that drops more than `--tolerance` (default 0.2)
below the baseline, or an assertion that flips from passing to failing, fails the run and names the case.
The tolerance is an absolute difference, not a percentage. Improvements never fail. Only cases and evaluator names
present on both sides are compared: new cases are reported as skipped, and removed cases or checks do not fail the
comparison. A passing comparison can retain existing baseline failures. Review dataset coverage and individual results.

The comparator does not check model or configuration identity. The
[comparison guide](../guides/evals.md) covers reviewed baseline selection and replacement.

## CI

`.github/workflows/eval.yml` runs the benchmark on demand and on pull requests labelled `eval`. It reads
`eval-models.json` and starts one job per entry. Each job starts `evals/docker-compose.yml`, runs
`run_eval.py --models <entry> --repeat 5 --baseline baseline_report.json --fail-under 0.7`, and gates on the committed
baseline and the floor. Its result table and any gate failure are on the run's summary page. A job gets only the repository secrets its entry names and is skipped with a notice when one of
them is not set, so adding a model needs no change to the workflow. An entry is not compared to a baseline until one is
committed.
