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

import httpx
import structlog
from oauth2_lib.settings import oauth2lib_settings

logger = structlog.get_logger(__name__)


class OAuthTokenManager:
    """Manages OAuth2 client credentials tokens with caching."""

    def __init__(self) -> None:
        self._token: str | None = None

    @property
    def auth_enabled(self) -> bool:
        return oauth2lib_settings.OAUTH2_ACTIVE

    async def get_token(self) -> str | None:
        if not self.auth_enabled:
            return None
        if self._token is not None:
            return self._token
        return await self._fetch_token()

    async def _fetch_token(self) -> str:
        if not oauth2lib_settings.OAUTH2_TOKEN_URL:
            raise RuntimeError("OAuth2 settings are incomplete; ensure OAUTH2_TOKEN_URL is set")

        logger.debug("Fetching OAuth2 token", endpoint=oauth2lib_settings.OAUTH2_TOKEN_URL)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                oauth2lib_settings.OAUTH2_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": oauth2lib_settings.OAUTH2_RESOURCE_SERVER_ID,
                    "client_secret": oauth2lib_settings.OAUTH2_RESOURCE_SERVER_SECRET,
                },
                timeout=30,
            )
        response.raise_for_status()

        token = response.json()["access_token"]
        self._token = token
        logger.debug("OAuth2 token acquired")
        return token

    async def refresh_token(self) -> str:
        self._token = None
        return await self._fetch_token()

    def get_auth_headers(self) -> dict[str, str]:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}


token_manager = OAuthTokenManager()
