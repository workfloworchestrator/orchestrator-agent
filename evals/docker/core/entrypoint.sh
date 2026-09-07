#!/bin/bash
# Eval-stack orchestrator-core entrypoint: migrate, seed, index, serve.
set -eu

cd /home/orchestrator
export PATH="/home/orchestrator/.venv/bin:$PATH"

# The image ships orchestrator-core without the [mcp] extra; install it for the
# exact version the image carries so the MCP server can mount.
if ! python -c "import fastmcp" 2>/dev/null; then
    core_version=$(python -c "from importlib.metadata import version; print(version('orchestrator-core'))")
    echo "= install orchestrator-core[mcp]==${core_version}"
    uv pip install --python /home/orchestrator/.venv "orchestrator-core[mcp]==${core_version}"
fi

if [ ! -f alembic.ini ]; then
    echo "= db init (bare deployment: scaffold migrations + alembic.ini)"
    python main.py db init
fi

echo "= db upgrade heads"
python main.py db upgrade heads

echo "= seed demo data"
python seed.py

echo "= index subscriptions"
python main.py index subscriptions

echo "= start orchestrator-core"
exec uvicorn --host 0.0.0.0 --port 8080 wsgi:app
