"""W-bot CLI Agent package."""

# Core components
from .core.agent import WBotGraph
from .core.config import Settings, load_settings

# Intent classification
from .intent import IntentClassifier, IntentType, IntentResult, ToolRecommendation, heuristic_classify

# Memory
from .memory import LongTermMemoryStore

# Skills
from .skills import SkillsLoader

# Tools
from .tools.runtime import build_tools

# Providers
from .providers import resolve_provider_capabilities

# Multimodal
from .multimodal import MultimodalNormalizer

__all__ = [
    # Core
    "WBotGraph",
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
