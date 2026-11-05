"""orchestrator-core CLI for the widget stack (migrations)."""

import demo  # noqa: F401  registers the task
from orchestrator.core import app_settings
from orchestrator.core.cli.main import app as core_cli
from orchestrator.core.db import init_database

if __name__ == "__main__":
    init_database(app_settings)
    core_cli()
