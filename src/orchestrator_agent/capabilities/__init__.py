from orchestrator_agent.capabilities.behavior import PluginCapability, build_plugin_capability
from orchestrator_agent.capabilities.hooks import build_capabilities, needs_form_fill
from orchestrator_agent.capabilities.loader import load_plugin_specs, load_system_prompt
from orchestrator_agent.capabilities.spec import PluginSpec, skills_from_specs

__all__ = [
    "PluginCapability",
    "PluginSpec",
    "build_capabilities",
    "build_plugin_capability",
    "load_system_prompt",
    "load_plugin_specs",
    "needs_form_fill",
    "skills_from_specs",
]
