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

"""Incoming request authentication for the standalone agent.

Ports the ``AgentAuthMiddleware`` pattern from orchestrator-core and adds a
concrete ``OIDCAuth`` subclass that resolves user info via the standard OIDC
userinfo endpoint.
"""

from http import HTTPStatus

import structlog
from fastapi import HTTPException
from httpx import AsyncClient
from oauth2_lib.fastapi import AuthManager, OIDCAuth, OIDCConfig, OIDCUserModel
from oauth2_lib.settings import oauth2lib_settings
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = structlog.get_logger(__name__)

# Paths that should never require authentication.
PUBLIC_PATHS = frozenset({"/health", "/health/", "/.well-known/agent-card.json"})


class UserinfoOIDCAuth(OIDCAuth):
    """OIDCAuth that resolves user identity via the OIDC userinfo endpoint.

    The base ``OIDCAuth`` in oauth2_lib leaves ``userinfo()`` abstract.  This
    subclass calls the provider's ``/userinfo`` endpoint with the bearer token,
    which is the standard OIDC flow supported by any OIDC provider.

    Also overrides ``check_openid_config`` to tolerate providers (e.g. AWS
    Cognito) whose discovery document omits fields that ``OIDCConfig`` requires.
    """

    async def check_openid_config(self, async_client: AsyncClient) -> None:
        """Load OIDC discovery, filling in defaults for missing optional fields."""
        if self.openid_config is not None:
            return

        response = await async_client.get(self.openid_config_url)
        if response.status_code != HTTPStatus.OK:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Could not load openid config from {self.openid_config_url}",
            )

        data = response.json()
        for key, default in {
            "response_modes_supported": [],
            "grant_types_supported": [],
            "claims_supported": [],
            "claims_parameter_supported": False,
            "request_parameter_supported": False,
            "code_challenge_methods_supported": [],
        }.items():
            data.setdefault(key, default)

        self.openid_config = OIDCConfig(**data)

    async def userinfo(self, async_request: AsyncClient, token: str) -> OIDCUserModel:
        if self.openid_config is None:
            raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="OIDC config not loaded")

        response = await async_request.get(
            self.openid_config.userinfo_endpoint,
            headers={"Authorization": f"Bearer {token}"},
        )
        if response.status_code != HTTPStatus.OK:
            logger.warning("Userinfo request failed", status=response.status_code)
            raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="Token validation failed")

        return self.user_model_cls(response.json())


def create_auth_manager() -> AuthManager:
    """Create an AuthManager with the userinfo-based OIDC authentication."""
    auth_manager = AuthManager()
    auth_manager.authentication = UserinfoOIDCAuth(
        openid_url=oauth2lib_settings.OIDC_BASE_URL,
        openid_config_url=oauth2lib_settings.OIDC_CONF_URL,
        resource_server_id=oauth2lib_settings.OAUTH2_RESOURCE_SERVER_ID,
        resource_server_secret=oauth2lib_settings.OAUTH2_RESOURCE_SERVER_SECRET,
        oidc_user_model_cls=OIDCUserModel,
    )
    return auth_manager


class AuthMiddleware:
    """ASGI middleware that enforces authentication on incoming requests.

    Based on ``AgentAuthMiddleware`` from orchestrator-core.  Differences:

    * Applied globally (A2A routes are added directly, not mounted).
    * Skips authentication for paths in ``PUBLIC_PATHS``.
    """

    def __init__(self, app: ASGIApp, auth_manager: AuthManager) -> None:
        self.app = app
        self.auth_manager = auth_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        if not oauth2lib_settings.OAUTH2_ACTIVE:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        auth_header = headers.get("authorization", "")
        token = auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else None

        request = Request(scope, receive)
        try:
            user = await self.auth_manager.authentication.authenticate(request, token)
            if user:
                await self.auth_manager.authorization.authorize(request, user)
            else:
                raise HTTPException(status_code=401, detail="Unauthorized")
        except HTTPException as exc:
            response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
