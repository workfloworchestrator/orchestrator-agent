# Evaluation benchmark

The `evals/` folder of the agent repository holds a golden-set benchmark built on
[pydantic-evals](https://ai.pydantic.dev/evals/). The agent runs in-process against a live Orchestrator Core MCP
server, so the benchmark measures agent behaviour only: which tools are called, in which order and with which
arguments. See [Understanding agent evaluations](../architecture/evaluation.md) for the reasoning and
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
| `MaxToolCalls(8)` | Dataset-wide budget of locally executed tool-call attempts, including failed attempts by default |

Tool calls are read from the OpenTelemetry spans that pydantic-ai emits.
`TrajectoryMatch` and `ArgumentCorrectness` exclude failed tool-call attempts by default; `MaxToolCalls` includes
them. A retry can therefore consume the call budget without appearing in the matched trajectory.

## Commands

Run from `evals/` after completing the [guide setup](../guides/evals.md#1-prepare-the-evaluation-environment).

| Command | Effect |
| --- | --- |
| `uv run python run_eval.py` | Run the full golden set |
| `uv run python run_eval.py --cases <name>` | Run the named case; accepts comma-separated names |
| `uv run python run_eval.py --fail-under 0.7` | Exit non-zero when the aggregate is below the floor |
| `uv run python run_eval.py --baseline baseline_report.json` | Fail when a case regresses against the committed baseline |
| `uv run python run_eval.py --save-report baseline_report.json` | Save a report; review it before accepting it as a baseline |

## Reports

Reports are printed to the terminal; JSON is written only with `--save-report`. The terminal includes inputs and
durations but hides final outputs. Saved reports retain outputs and evaluator reasons, not full tool transcripts.

Each completed case contains:

| Field | Contents |
| --- | --- |
| `scores` | Numeric evaluator results; `TrajectoryMatch` ranges from 0 to 1, with 1 indicating a full match in its configured mode |
| `assertions` | Separate boolean checks, including selected arguments and the call budget |
| `reason` within an evaluator result | The evaluator's explanation, when provided |
| `output` | The final agent answer; not graded by the bundled cases |

Execution failures are recorded separately in `failures`. A completed case can have failed assertions or low scores.
The [product-count example](../guides/add-eval-case.md#2-add-the-case-to-the-dataset) checks `aggregate` and `entity_type: PRODUCT`, but does not verify the returned
count or answer against the three seeded products. The benchmark records tool-call spans locally for grading; it does
not automatically publish its reports to Langfuse.

## Exit status

Without gates, failed assertions and low scores do not cause a non-zero exit. Case execution failures do. With
`--fail-under`, the runner averages the per-score averages together with the assertion pass rate (when present).
This aggregate is not a task-success percentage, and the floor does not require every assertion to pass.

The runner does not automatically capture the model and all environment settings in report metadata.

## Settings

Real environment variables override `evals/.env`. The harness reads `AGENT_MODEL`, `AGENT_API_KEY`,
`WFO_CORE_MCP_URL`, `OAUTH2_ACTIVE` and `DATASET_FILE`. It publishes the first four to the agent environment.
Export other agent settings, such as `AGENT_API_BASE`, `AGENT_API_VERSION` and `AGENT_DOMAIN_CONTEXT`, in the shell;
the harness does not forward them from its `.env`.

The runner executes cases sequentially, with one attempt per case. It has no repeat flag.

## Baseline

`baseline_report.json` is a full report of a blessed run. A score that drops more than `--tolerance` (default 0.2)
below the baseline, or an assertion that flips from passing to failing, fails the run and names the case.
The tolerance is an absolute difference, not a percentage. Improvements never fail. Only cases and evaluator names
present on both sides are compared: new cases are reported as skipped, and removed cases or checks do not fail the
comparison. A passing comparison can retain existing baseline failures. Review dataset coverage and individual results.

The comparator does not check model or configuration identity. The
[comparison guide](../guides/evals.md) covers reviewed baseline selection and replacement.

## CI

`.github/workflows/eval.yml` runs the benchmark on demand, nightly and on pull requests labelled `eval`. It gates on
the committed baseline and the `--fail-under` floor. It needs the `EVAL_AGENT_API_KEY` repository secret and
optionally an `EVAL_AGENT_MODEL` variable (default `openai:gpt-4o`).
