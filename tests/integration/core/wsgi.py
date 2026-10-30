"""orchestrator-core with the widget demo: its customers resolver, its task, and the MCP server (MCP_ENABLED)."""

import demo
from orchestrator.core import OrchestratorCore
from orchestrator.core.settings import AppSettings

app = OrchestratorCore(base_settings=AppSettings())
app.include_router(demo.router)  # the route the demo form's fields point at
