FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock .
COPY src/ src/
RUN uv pip install --system --no-cache .

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "orchestrator_agent.app:app", "--host", "0.0.0.0", "--port", "8080"]
