"""Custom evaluators for the golden set.

``TrajectoryMatch`` and ``ArgumentCorrectness`` judge the route the agent took. ``ToolResultContains`` judges what the
route produced, so a run that reaches the right data by a different route is not marked down for it.
"""

import json
from dataclasses import dataclass, field

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

_TOOL_NAME = "gen_ai.tool.name"
# The instrumentation versions of pydantic-ai record a tool's return value under different attribute names
_RESULT_ATTRIBUTES = ("gen_ai.tool.call.result", "tool_response")


def _compact(recorded: object) -> str:
    """A recorded tool result as compact JSON text, whatever shape the instrumentation stored it in.

    The result may be stored as the JSON of the returned object or as the JSON of a string holding that JSON;
    both are unwrapped so the expected strings in the dataset can be written once, in compact form.
    """
    if not isinstance(recorded, str):
        return json.dumps(recorded, separators=(",", ":"), ensure_ascii=False)
    try:
        return _compact(json.loads(recorded))
    except ValueError:
        return recorded


@dataclass(repr=False)
class ToolResultContains(Evaluator[object, object, object]):
    """Pass when the last successful call to ``tool_name`` returned every string in ``contains`` and none in ``excludes``.

    The last call is the one whose result the person ends up seeing. Strings are matched against the result as compact
    JSON (no spaces after ``:`` or ``,``).
    """

    tool_name: str
    contains: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        calls = sorted(
            (
                node
                for node in ctx.span_tree
                if node.attributes.get(_TOOL_NAME) == self.tool_name and node.status != "error"
            ),
            key=lambda node: node.start_timestamp,
        )
        results = [
            _compact(node.attributes[name]) for node in calls for name in _RESULT_ATTRIBUTES if name in node.attributes
        ]
        if not results:
            return EvaluationReason(False, f"no successful {self.tool_name} call returned a recorded result")
        result = results[-1]
        missing = [text for text in self.contains if text not in result]
        unexpected = [text for text in self.excludes if text in result]
        if missing or unexpected:
            return EvaluationReason(False, f"last {self.tool_name} result: missing {missing}, unexpected {unexpected}")
        return EvaluationReason(True)
