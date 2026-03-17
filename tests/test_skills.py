"""Tests for skills registry."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URI", "postgresql://test:test@localhost:5432/test")

from orchestrator_agent.memory import MemoryScope
from orchestrator_agent.skills import SKILLS, Skill
from orchestrator_agent.state import SearchState, TaskAction


class TestSkillsRegistry:
    def test_all_actions_have_skills(self):
        for action in TaskAction:
            assert action in SKILLS, f"Missing skill for {action}"

    def test_skill_structure(self):
        for action, skill in SKILLS.items():
            assert isinstance(skill, Skill)
            assert skill.action == action
            assert skill.name
            assert skill.description
            assert isinstance(skill.tags, list)
            assert isinstance(skill.memory_scope, MemoryScope)

    def test_skill_prompts_callable(self):
        state = SearchState(user_input="test")
        for skill in SKILLS.values():
            prompt = skill.get_prompt(state)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_search_skill_has_toolsets(self):
        assert len(SKILLS[TaskAction.SEARCH].toolsets) > 0

    def test_text_response_has_no_toolsets(self):
        assert SKILLS[TaskAction.TEXT_RESPONSE].toolsets == []
