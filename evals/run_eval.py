"""Run the WFO search agent golden-set benchmark with pydantic-evals.

The agent runs in-process — built by the same ``build_agent`` /
``agent_settings.create_model()`` assembly the server adapters use — against
the live orchestrator-core MCP server from docker-compose.yml. The built-in
span-based evaluators (TrajectoryMatch, ArgumentCorrectness, MaxToolCalls)
read the tool calls from the OpenTelemetry spans pydantic-ai emits, so this
measures agent behavior only: which tools, in what order, with which
arguments.

    uv run python run_eval.py [--cases name1,name2] [--fail-under 0.8]
"""

import argparse
import sys
from pathlib import Path

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

from pydantic_evals import Dataset
from pydantic_evals.reporting import EvaluationReportAdapter

from orchestrator_agent.agent import build_agent, new_deps
from orchestrator_agent.settings import agent_settings

_agent = None


def _get_agent():
    """Build the agent on first use so e.g. --help works without credentials."""
    global _agent
    if _agent is None:
        _agent = build_agent(agent_settings.create_model())
    return _agent


async def run_wfo_agent(question: str) -> str:
    """The task under evaluation: one user turn against the real MCP toolset."""
    agent = _get_agent()
    deps = new_deps(question)
    async with agent:
        result = await agent.run(question, deps=deps)
    return result.output


def compare_to_baseline(report, baseline, tolerance: float) -> list[str]:
    """Per-case regressions of this run against a stored baseline report.

    Compares every score and assertion of every case present in both reports:
    a score counts as regressed when it drops more than ``tolerance`` below the
    baseline (LLM runs are non-deterministic; the tolerance absorbs run-to-run
    noise), an assertion when it flips from passing to failing. Improvements
    and cases only present on one side never fail the comparison.
    """
    baseline_cases = {case.name: case for case in baseline.cases}
    regressions: list[str] = []
    for case in report.cases:
        base = baseline_cases.get(case.name)
        if base is None:
            print(f"baseline: case {case.name!r} not in baseline, skipping")
            continue
        for name, result in case.scores.items():
            base_result = base.scores.get(name)
            if base_result is None:
                continue
            delta = float(result.value) - float(base_result.value)
            if delta < -tolerance:
                regressions.append(f"{case.name} :: {name} {float(base_result.value):.2f} -> {float(result.value):.2f}")
        for name, result in case.assertions.items():
            base_result = base.assertions.get(name)
            if base_result is not None and base_result.value and not result.value:
                regressions.append(f"{case.name} :: {name} passed -> failed ({result.reason})")
    return regressions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", help="comma-separated case names to run (default: all)")
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
        help="stored report (see --save-report) to compare against; exit non-zero on per-case regression",
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
        help="write this run's full report as JSON (pass baseline_report.json to bless a new baseline)",
    )
    args = parser.parse_args()

    dataset = Dataset[str, str].from_file(eval_settings.dataset_file)
    if args.cases:
        wanted = {name.strip() for name in args.cases.split(",")}
        dataset = Dataset[str, str](
            name=dataset.name,
            cases=[c for c in dataset.cases if c.name in wanted],
            evaluators=dataset.evaluators,
        )

    report = dataset.evaluate_sync(run_wfo_agent, name="wfo-search", max_concurrency=1)
    report.print(include_input=True, include_output=False, include_durations=True)

    if args.save_report:
        args.save_report.write_bytes(EvaluationReportAdapter.dump_json(report, indent=2))
        print(f"report written to {args.save_report}")

    if report.failures:
        sys.exit(f"{len(report.failures)} case(s) failed to execute")

    failures: list[str] = []
    if args.baseline:
        baseline = EvaluationReportAdapter.validate_json(args.baseline.read_bytes())
        regressions = compare_to_baseline(report, baseline, args.tolerance)
        for line in regressions:
            print(f"REGRESSION vs {args.baseline.name}: {line}")
        if regressions:
            failures.append(f"{len(regressions)} regression(s) vs baseline {args.baseline.name}")
        else:
            print(f"no regressions vs baseline {args.baseline.name} (tolerance {args.tolerance})")

    if args.fail_under is not None:
        averages = report.averages()
        values = list(averages.scores.values())
        if averages.assertions is not None:
            values.append(averages.assertions)
        mean = sum(values) / len(values) if values else 0.0
        print(f"overall mean (score averages + assertion pass rate): {mean:.3f}")
        if mean < args.fail_under:
            failures.append(f"overall mean {mean:.3f} below --fail-under {args.fail_under}")

    if failures:
        sys.exit("; ".join(failures))


if __name__ == "__main__":
    main()
