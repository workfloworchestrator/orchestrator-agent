"""Run the WFO search agent golden-set benchmark with pydantic-evals.

The agent runs in-process — built by the same ``build_agent`` assembly the
server adapters use — against the live orchestrator-core MCP server from
docker-compose.yml, once per model in ../eval-models.json whose provider key
is set. The span-based evaluators read the tool calls from the OpenTelemetry
spans pydantic-ai emits: TrajectoryMatch and ArgumentCorrectness judge the
route (which tools, in what order, with which arguments), ToolResultContains
(evaluators.py) judges what the last call returned, so a different route to the
right data is visible as such.

A ``agent:<model>`` entry runs the model the way the server would be configured
for an OpenAI-compatible endpoint, from the agent's own environment:
AGENT_API_BASE and AGENT_API_KEY (both required, else the entry shows "no key")
and AGENT_MODEL_SETTINGS, e.g. for an open-weight model behind a LiteLLM proxy.

    uv run python run_eval.py [--models openai:gpt-4o-mini] [--cases name1,name2] [--repeat 5] [--fail-under 0.8]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from statistics import fmean
from typing import NamedTuple
from urllib.parse import urlparse

from evaluators import ToolResultContains
from settings import eval_settings

# The agent's AgentSettings reads the process environment at import time, so
# the resolved eval settings must be published before importing the agent.
eval_settings.apply_to_environment()

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# The span-based evaluators need a real (SDK) tracer provider; pydantic-evals
# attaches its in-memory exporter to it per case.
trace.set_tracer_provider(TracerProvider())

from pydantic_ai import Agent

Agent.instrument_all()

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Dataset
from pydantic_evals.reporting import EvaluationReport, EvaluationReportAdapter
from rich.align import Align
from rich.console import Console
from rich.table import Table

from orchestrator_agent.agent import WFOAgent, build_agent, new_deps
from orchestrator_agent.form_fill import build_form_fill_skill
from orchestrator_agent.settings import AgentSettings


def configured_agent(name: str) -> AgentSettings | None:
    """The agent settings behind an ``agent:<model>`` entry, or None when no endpoint and key are configured.

    The endpoint, key and model settings are the ones the agent itself reads from AGENT_API_BASE, AGENT_API_KEY and
    AGENT_MODEL_SETTINGS; only the model is replaced, so one environment can be evaluated against several models.
    """
    settings = AgentSettings(AGENT_MODEL=name.partition(":")[2])
    return settings if settings.AGENT_API_BASE and settings.AGENT_API_KEY else None


def model_settings_for(name: str) -> ModelSettings | None:
    """The model settings an entry runs with: the configured ones for ``agent:`` entries, none for the others."""
    settings = configured_agent(name) if name.startswith("agent:") else None
    return settings.create_model_settings() if settings else None


# Saved reports can be committed as baselines, so only settings that cannot carry a credential are written to them.
# AGENT_MODEL_SETTINGS also accepts extra_headers, extra_body and provider-specific keys, which can.
_REPORTABLE_SETTINGS = frozenset(
    {"temperature", "top_p", "max_tokens", "thinking", "seed", "timeout", "parallel_tool_calls"}
)


def run_metadata(name: str) -> dict[str, object]:
    """What to record in the report so a result can be tied to the configuration that produced it.

    Model settings outside the allowlist are recorded by name only, so a report shows that they were set without
    holding their values.
    """
    settings = configured_agent(name) if name.startswith("agent:") else None
    model_settings = dict(settings.create_model_settings() or {}) if settings else {}
    return {
        "model": name,
        "model_settings": {key: value for key, value in model_settings.items() if key in _REPORTABLE_SETTINGS},
        "unreported_model_settings": sorted(set(model_settings) - _REPORTABLE_SETTINGS),
        "api_host": urlparse(settings.AGENT_API_BASE).hostname if settings and settings.AGENT_API_BASE else None,
    }


def build_model(name: str) -> Model | None:
    """The pydantic-ai model for a ``provider:model`` name, or None when the settings hold no key for that provider."""
    provider, _, model_name = name.partition(":")
    if provider == "agent":
        configured = configured_agent(name)
        model = configured.create_model() if configured else None
        return model if isinstance(model, Model) else None  # a custom endpoint always yields a model, never a bare name
    if provider == "openai":
        if not eval_settings.openai_api_key:
            return None
        # Chat Completions, as the agent's own settings build it; the plain model string would take the
        # Responses API, which the locked openai SDK cannot read.
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=eval_settings.openai_api_key))
    if provider == "anthropic":
        if not eval_settings.anthropic_api_key:
            return None
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
        from pydantic_ai.providers.anthropic import AnthropicProvider

        # The instructions and tool definitions are the same on every call: cache them.
        settings = AnthropicModelSettings(anthropic_cache_instructions=True, anthropic_cache_tool_definitions=True)
        anthropic = AnthropicProvider(api_key=eval_settings.anthropic_api_key)
        return AnthropicModel(model_name, provider=anthropic, settings=settings)
    raise SystemExit(f"{name}: no provider {provider!r} in build_model")


def task_for(agent: WFOAgent) -> Callable[[str], Awaitable[str]]:
    """The task under evaluation: one user turn of this agent against the real MCP toolset."""

    async def run_wfo_agent(question: str) -> str:
        deps = new_deps(question)
        async with agent:
            result = await agent.run(question, deps=deps)
        return str(result.output)

    return run_wfo_agent


def for_model(path: Path, model: str) -> Path:
    """``baseline_report.json`` for ``openai:gpt-4o-mini`` is ``baseline_report.openai-gpt-4o-mini.json``."""
    return path.with_name(f"{path.stem}.{model.replace(':', '-').replace('/', '-')}{path.suffix}")


def crashes(report: EvaluationReport) -> Counter[str]:
    """Per case, how many of its runs failed to execute. Such a run counts as 0 on everything the case measures."""
    return Counter(failure.source_case_name or failure.name for failure in report.failures)


def case_means(report: EvaluationReport) -> dict[str, dict[str, float]]:
    """Per case, the average of every score and the pass rate of every assertion over the case's runs.

    A ``--repeat`` report holds one entry per run of a case. A crashed run counts as 0; a case
    whose every run crashed has no measured names and maps to an empty dict.
    """
    crashed = crashes(report)
    values: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in report.cases:
        for name, result in (run.scores | run.assertions).items():
            values[run.source_case_name or run.name][name].append(float(result.value))
    return {
        case_name: {
            name: sum(of_runs) / (len(of_runs) + crashed[case_name]) for name, of_runs in values[case_name].items()
        }
        for case_name in values.keys() | crashed.keys()
    }


class CaseScores(NamedTuple):
    route: float
    arguments: float | None
    outcome: float | None


def _pass_rate(results: list[float | None], crashed: int) -> float | None:
    """The mean of the runs that have this measure, crashed runs counting as 0; None when no run has it."""
    present = [value for value in results if value is not None]
    return sum(present) / (len(present) + crashed) if present else None


def case_scores(report: EvaluationReport) -> dict[str, CaseScores]:
    """Per case over its runs: the route score average, the argument-correctness pass rate and the outcome pass rate.

    A run that crashed counts as 0 on all three; a case without an argument or outcome assertion has no rate for it.
    """
    crashed = crashes(report)
    runs: dict[str, list[CaseScores]] = defaultdict(list)
    for case in report.cases:
        arguments, outcome = case.assertions.get("ArgumentCorrectness"), case.assertions.get("ToolResultContains")
        runs[case.source_case_name or case.name].append(
            CaseScores(
                float(case.scores["TrajectoryMatch"].value),
                None if arguments is None else float(arguments.value),
                None if outcome is None else float(outcome.value),
            )
        )
    return {
        name: CaseScores(
            sum(run.route for run in runs[name]) / (len(runs[name]) + crashed[name]),
            _pass_rate([run.arguments for run in runs[name]], crashed[name]),
            _pass_rate([run.outcome for run in runs[name]], crashed[name]),
        )
        for name in runs.keys() | crashed.keys()
    }


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def print_table(reports: dict[str, EvaluationReport | None], cases: list[str], repeat: int) -> None:
    """One table: a column per model holding three sub-columns, route match, argument correctness and outcome.

    The route is how the agent got there, the outcome is whether the last call returned the expected data. Rich
    tables cannot span a header over several columns, so each model's cell holds a three-column grid; every grid has
    the same rows (a header line, the cases, the averages), so the rows line up across the models.
    """
    scores = {model: case_scores(report) for model, report in reports.items() if report is not None}

    def column(model: str) -> Table:
        grid = Table.grid(padding=(0, 2))
        if model not in scores:
            grid.add_column()
            grid.add_row("no key")
            return grid
        grid.add_column(justify="right")
        grid.add_column(justify="right")
        grid.add_column(justify="right")
        grid.add_row("route", "arguments", "outcome")
        for case in cases:
            grid.add_row(*(_percent(value) for value in scores[model][case]))
        everything = list(scores[model].values())
        averages = (
            fmean(case.route for case in everything),
            *(
                fmean(values)
                if (values := [v for case in everything if (v := getattr(case, field)) is not None])
                else None
                for field in ("arguments", "outcome")
            ),
        )
        grid.add_row(*(_percent(average) for average in averages), style="bold")
        return grid

    names = Table.grid()
    names.add_column(no_wrap=True)
    names.add_row("")  # level with the route / arguments / outcome line
    for case in cases:
        names.add_row(case)
    names.add_row("Averages", style="bold")

    table = Table(title=f"Average over {repeat} run(s) per case")
    table.add_column("Case")
    for model in reports:
        table.add_column(model, justify="center")
    table.add_row(names, *(Align.center(column(model)) for model in reports))
    # Wide enough for every column, so a narrow terminal or a CI log wraps rather than truncates.
    Console(width=max(Console().width, 28 + 36 * len(reports))).print(table)


def compare_to_baseline(report: EvaluationReport, baseline: EvaluationReport, tolerance: float) -> list[str]:
    """Per-case regressions of this run against a stored baseline report.

    Compares every score average and assertion pass rate of every case present
    in both reports: either counts as regressed when it drops more than
    ``tolerance`` below the baseline (LLM runs are non-deterministic; the
    tolerance absorbs run-to-run noise). With one run per case a pass rate is 1
    or 0, so an assertion that flips from passing to failing is a regression.
    Improvements and cases only present on one side never fail the comparison.
    A crashed run counts as 0, so a case that starts crashing is a regression.
    """
    baseline_means = case_means(baseline)
    regressions: list[str] = []
    for case_name, means in case_means(report).items():
        base = baseline_means.get(case_name)
        if base is None:
            print(f"baseline: case {case_name!r} not in baseline, skipping")
            continue
        if not means:  # every run crashed
            means = dict.fromkeys(base, 0.0)
        for name, value in means.items():
            if name in base and value - base[name] < -tolerance:
                regressions.append(f"{case_name} :: {name} {base[name]:.2f} -> {value:.2f}")
    return regressions


def evaluate_model(
    model_name: str, dataset: Dataset[str, str], args: argparse.Namespace
) -> tuple[EvaluationReport | None, list[str]]:
    """Run the dataset against one model, print its report notes, and return the report with its gate failures.

    The report is None when the settings hold no key for the model's provider.
    """
    model = build_model(model_name)
    if model is None:
        return None, []
    agent = build_agent(model, form_fill=build_form_fill_skill(), model_settings=model_settings_for(model_name))
    report = dataset.evaluate_sync(
        task_for(agent),
        name=model_name,
        max_concurrency=args.concurrency,
        repeat=args.repeat,
        metadata=run_metadata(model_name),
    )
    failures: list[str] = []

    if args.save_report:
        path = for_model(args.save_report, model_name)
        path.write_bytes(EvaluationReportAdapter.dump_json(report, indent=2))
        print(f"{model_name}: report written to {path}")

    if report.failures:
        print(f"{model_name}: {len(report.failures)} run(s) failed to execute, counted as 0")

    if args.baseline:
        path = for_model(args.baseline, model_name)
        if not path.exists():
            print(f"{model_name}: no baseline at {path.name}, nothing to compare against")
        else:
            baseline = EvaluationReportAdapter.validate_json(path.read_bytes())
            regressions = compare_to_baseline(report, baseline, args.tolerance)
            print("\n".join(f"REGRESSION vs {path.name}: {line}" for line in regressions))
            if regressions:
                failures.append(f"{model_name}: {len(regressions)} regression(s) vs baseline {path.name}")
            else:
                print(f"{model_name}: no regressions vs baseline {path.name} (tolerance {args.tolerance})")

    if args.fail_under is not None:
        averages = report.averages()
        values = list(averages.scores.values())
        if averages.assertions is not None:
            values.append(averages.assertions)
        mean = sum(values) / len(values) if values else 0.0
        mean *= len(report.cases) / ((len(report.cases) + len(report.failures)) or 1)  # a crashed run is 0
        print(f"{model_name}: overall mean (score averages + assertion pass rate): {mean:.3f}")
        if mean < args.fail_under:
            failures.append(f"{model_name}: overall mean {mean:.3f} below --fail-under {args.fail_under}")
    return report, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        help="comma-separated models from eval-models.json to run (default: every one whose provider key is set)",
    )
    parser.add_argument("--cases", help="comma-separated case names to run (default: all)")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run every case this many times; scores and assertions then count as per-case averages",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="how many runs go at once; the agent shares one MCP session and core's tools keep no session state",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit non-zero when the mean over all assertions/scores drops below this",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="stored report (see --save-report) to compare against, one file per model; exit non-zero on per-case regression",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="score drop vs the baseline that still counts as run-to-run noise (default 0.2)",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="write each model's full report as JSON, the model in the file name (baseline_report.json blesses baselines)",
    )
    args = parser.parse_args()

    dataset = Dataset[str, str].from_file(eval_settings.dataset_file, custom_evaluator_types=[ToolResultContains])
    if args.cases:
        wanted = {name.strip() for name in args.cases.split(",")}
        dataset = Dataset[str, str](
            name=dataset.name,
            cases=[c for c in dataset.cases if c.name in wanted],
            evaluators=dataset.evaluators,
        )

    models: list[str] = json.loads(eval_settings.models_file.read_text())
    if args.models:
        selected = [name.strip() for name in args.models.split(",")]
        if unknown := [name for name in selected if name not in models]:
            sys.exit(f"not in {eval_settings.models_file.name}: {', '.join(unknown)}")
        models = selected

    outcomes = {model_name: evaluate_model(model_name, dataset, args) for model_name in models}
    reports = {model_name: report for model_name, (report, _) in outcomes.items()}
    failures = [failure for _, found in outcomes.values() for failure in found]

    print_table(reports, [case.name for case in dataset.cases], args.repeat)
    if not any(reports.values()):
        failures.append("no model ran: set a provider key (see .env.example)")
    if failures:
        sys.exit("; ".join(failures))


if __name__ == "__main__":
    main()
