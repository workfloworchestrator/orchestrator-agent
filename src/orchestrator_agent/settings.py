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

from pydantic import Field
from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    """Settings for the standalone orchestrator agent."""

    DATABASE_URI: str = Field(description="PostgreSQL connection URI for WFO database")
    BASE_URL: str = Field(default="http://localhost:8000", description="Public URL of this agent service")


agent_settings = AgentSettings()  # type: ignore[call-arg]
