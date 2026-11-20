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

from pydantic_ai.models.test import TestModel

from orchestrator_agent.agent import build_agent


def test_build_agent_carries_no_model_settings_by_default():
    assert build_agent(TestModel()).model_settings is None


def test_build_agent_applies_the_model_settings():
    agent = build_agent(TestModel(), model_settings={"temperature": 0.3, "thinking": "medium"})
    assert agent.model_settings == {"temperature": 0.3, "thinking": "medium"}
