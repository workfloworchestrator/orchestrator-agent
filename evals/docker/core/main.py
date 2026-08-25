"""orchestrator-core CLI entrypoint for the eval stack (migrations, indexing)."""

import demo_products  # noqa: F401  Side-effects: registers the demo domain models
from orchestrator.core import app_settings
from orchestrator.core.cli.main import app as core_cli
from orchestrator.core.db import init_database

if __name__ == "__main__":
    init_database(app_settings)
    core_cli()
