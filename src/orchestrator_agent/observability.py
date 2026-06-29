# Copyright 2019-2025 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional Langfuse OpenTelemetry tracing for the orchestrator agent.

Langfuse and pydantic-ai's instrumentation are imported lazily so this module
stays importable without the optional ``langfuse`` extra installed. Every failure
mode (extra missing, missing/invalid credentials, auth failure) degrades to
"no tracing" and never prevents the application from starting.
"""

from __future__ import annotations

import structlog

from orchestrator_agent.settings import agent_settings

logger = structlog.get_logger(__name__)


def configure_langfuse() -> object | None:
    """Enable Langfuse OTel tracing if configured.

    Returns the Langfuse client when tracing is active, or ``None`` when disabled
    or unavailable. Credentials and host are read by the langfuse SDK from the
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST environment variables.
    """
    if not agent_settings.LANGFUSE_ENABLED:
        return None

    try:
        from langfuse import get_client
        from pydantic_ai import Agent
    except ImportError:
        logger.error("LANGFUSE_ENABLED is set but the 'langfuse' extra is not installed; tracing disabled")
        return None

    client = get_client()
    if not client.auth_check():
        logger.warning("Langfuse auth check failed; tracing disabled. Check LANGFUSE_* environment variables")
        return None

    Agent.instrument_all()
    logger.info("Langfuse tracing enabled")
    return client


def shutdown_langfuse(client: object) -> None:
    """Flush any pending spans to Langfuse on shutdown."""
    client.flush()  # type: ignore[attr-defined]
