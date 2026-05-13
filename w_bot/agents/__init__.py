"""W-bot CLI Agent package."""

# Core components
from .core.config import Settings, load_settings
from .core.runtime import AgentRuntime

# Intent classification
from .intent import IntentClassifier, IntentResult, IntentType, ToolRecommendation, heuristic_classify

# Memory
from .memory import LongTermMemoryStore

# Multimodal
from .multimodal import MultimodalNormalizer

# Providers
from .providers import resolve_provider_capabilities

# Skills
from .skills import SkillsLoader

# Tools
from .tools.runtime import build_tools

__all__ = [
    # Core
    "AgentRuntime",
    "Settings",
    "load_settings",
    # Intent
    "IntentClassifier",
    "IntentType",
    "IntentResult",
    "ToolRecommendation",
    "heuristic_classify",
    # Memory
    "LongTermMemoryStore",
    # Skills
    "SkillsLoader",
    # Tools
    "build_tools",
    # Providers
    "resolve_provider_capabilities",
    # Multimodal
    "MultimodalNormalizer",
]
