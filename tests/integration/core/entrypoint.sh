#!/bin/bash
# Widget integration stack: migrate, seed, serve.
set -eu

cd /home/orchestrator
export PATH="/home/orchestrator/.venv/bin:$PATH"

# The image ships orchestrator-core without the [mcp] extra; install it for the version the image carries.
if ! python -c "import fastmcp" 2>/dev/null; then
    core_version=$(python -c "from importlib.metadata import version; print(version('orchestrator-core'))")
    uv pip install --python /home/orchestrator/.venv "orchestrator-core[mcp]==${core_version}"
fi

if [ ! -f alembic.ini ]; then
    python main.py db init
fi
python main.py db upgrade heads
python seed.py
exec uvicorn --host 0.0.0.0 --port 8080 wsgi:app
