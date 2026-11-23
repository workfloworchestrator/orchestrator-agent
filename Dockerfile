FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG UV_EXTRA_ARGS=""

COPY pyproject.toml uv.lock .
COPY src/ src/
RUN uv sync --frozen --no-dev ${UV_EXTRA_ARGS}

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

CMD ["uvicorn", "orchestrator_agent.app:app", "--host", "0.0.0.0", "--port", "8080"]
