"""Bare orchestrator-core app for the eval stack: no workflows, just the REST
API, the MCP server (MCP_ENABLED=True in the environment) and the minimal
domain models that let the seeded subscriptions index and resolve."""

import demo_products  # noqa: F401  Side-effects: registers the demo domain models
from orchestrator.core import OrchestratorCore
from orchestrator.core.settings import AppSettings

app = OrchestratorCore(base_settings=AppSettings())
