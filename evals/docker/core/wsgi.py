"""Bare orchestrator-core app for the eval stack: two demo workflows, the REST
API, the MCP server (MCP_ENABLED=True in the environment) and the minimal
domain models that let the seeded subscriptions index and resolve."""

import demo_products  # noqa: F401  Side-effects: registers the demo domain models
import demo_workflows  # Side-effects: registers the demo workflows
from orchestrator.core import OrchestratorCore
from orchestrator.core.settings import AppSettings

app = OrchestratorCore(base_settings=AppSettings())
app.include_router(demo_workflows.router)  # where the port form's customer options are
