"""Two-stage hybrid intent classifier for W-bot.

This module implements the main IntentClassifier class that combines
heuristic (fast path) and LLM-based (accurate) intent classification.
"""

from typing import Any

from langchain_core.messages import AnyMessage

from ..core.config import IntentClassifierSettings
from ..core.logging_config import get_logger
from .intent import IntentResult
from .intent_heuristic import _should_expose_run_skill, heuristic_classify
from .intent_llm import _llm_classify_async, llm_classify_sync

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for available tools."""

    def __init__(self, tools: list[Any] | None = None):
        self._tools_by_name: dict[str, Any] = {}
        if tools:
            for tool in tools:
                name = str(getattr(tool, "name", "")).strip()
                if name:
                    self._tools_by_name[name] = tool

    def has(self, tool_name: str) -> bool:
        """Check if a tool exists in the registry."""
        return tool_name in self._tools_by_name

    def register(self, tool: Any) -> None:
        """Register a tool."""
        name = str(getattr(tool, "name", "")).strip()
        if name:
            self._tools_by_name[name] = tool

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Get all tool names."""
        return tuple(sorted(self._tools_by_name.keys()))


class IntentClassifier:
    """Two-stage hybrid intent classifier.

    Stage 1: Heuristic classification (millisecond-level)
    Stage 2: LLM classification (optional, 200-500ms)
    """

    def __init__(
        self,
        llm: Any | None,
        settings: IntentClassifierSettings,
        tools_registry: ToolRegistry | None = None,
    ):
        """Initialize the intent classifier.

        Args:
            llm: LLM instance for Stage 2 classification (must support ainvoke)
            settings: Intent classifier configuration
            tools_registry: Registry of available tools for tool exposure control
        """
        self._llm = llm
        self._settings = settings
        self._registry = tools_registry or ToolRegistry()

    @property
    def settings(self) -> IntentClassifierSettings:
        """Get classifier settings."""
        return self._settings

    async def classify(
        self,
        text: str,
        history: list[AnyMessage] | None = None,
    ) -> IntentResult:
        """Execute two-stage intent classification.

        Args:
            text: User input text
            history: Optional message history for context

        Returns:
            IntentResult with classified intent and tool recommendations
        """
        if not self._settings.enabled:
            return IntentResult.default_unknown()

        # === Stage 1: Heuristic pre-classification ===
        heuristic_result = heuristic_classify(text, history)

        if not heuristic_result.requires_llm:
            return heuristic_result

        # === Stage 2: LLM classification ===
        if not self._settings.use_llm or self._llm is None:
            return heuristic_result  # Fall back to heuristic result

        llm_result = await _llm_classify_async(
            text=text,
            llm=self._llm,
            history=history,
            timeout=self._settings.llm_timeout_seconds,
        )

        # === Merge results ===
        return self._merge_results(heuristic_result, llm_result)

    def classify_sync(
        self,
        text: str,
        history: list[AnyMessage] | None = None,
    ) -> IntentResult:
        """Synchronous version of classify (fallback when async not available).

        Args:
            text: User input text
            history: Optional message history for context

        Returns:
            IntentResult with classified intent and tool recommendations
        """
        if not self._settings.enabled:
            return IntentResult.default_unknown()

        # === Stage 1: Heuristic pre-classification ===
        heuristic_result = heuristic_classify(text, history)

        if not heuristic_result.requires_llm:
            return heuristic_result

        # === Stage 2: LLM classification (sync) ===
        if not self._settings.use_llm or self._llm is None:
            return heuristic_result

        llm_result = llm_classify_sync(
            text=text,
            llm=self._llm,
            history=history,
        )

        return self._merge_results(heuristic_result, llm_result)

    def _merge_results(
        self,
        heuristic: IntentResult,
        llm: IntentResult,
    ) -> IntentResult:
        """Merge heuristic and LLM results.

        Strategy:
        - If LLM confidence is high, prefer LLM result
        - If LLM confidence is low, consider heuristic result
        - If both are uncertain, prefer LLM

        Args:
            heuristic: Result from Stage 1 (heuristic)
            llm: Result from Stage 2 (LLM)

        Returns:
            Merged IntentResult
        """
        # Use LLM result if confidence is above threshold
        if llm.primary_intent.confidence >= self._settings.confidence_threshold_llm:
            return llm

        # Use heuristic result if confidence is above threshold
        if heuristic.primary_intent.confidence >= self._settings.confidence_threshold_heuristic:
            return heuristic

        # Both uncertain, prefer LLM but keep heuristic's secondary intents
        return IntentResult(
            primary_intent=llm.primary_intent,
            secondary_intents=heuristic.secondary_intents,
            should_enable_tools=llm.should_enable_tools or heuristic.should_enable_tools,
            recommended_tools=llm.recommended_tools or heuristic.recommended_tools,
            requires_llm=False,
            metadata={
                "source": "merged",
                "llm_confidence": llm.primary_intent.confidence,
                "heuristic_confidence": heuristic.primary_intent.confidence,
            },
        )

    def select_tools_for_intent(self, result: IntentResult) -> tuple[str, ...]:
        """Select tools to expose based on intent result.

        Args:
            result: Intent classification result

        Returns:
            Tuple of tool names to expose
        """
        if not self._settings.enable_tool_exposure_control:
            return self._registry.tool_names

        # If tools should not be enabled, return empty tuple
        if not result.should_enable_tools:
            return ()

        if not result.recommended_tools:
            # No recommendations but tools needed, return all tools
            return self._registry.tool_names

        # Sort by exposure level
        sorted_tools = sorted(
            result.recommended_tools,
            key=lambda t: t.exposure_level,
            reverse=True,
        )

        selected = []
        for tool in sorted_tools:
            if tool.exposure_level < 0.3:
                continue
            if self._registry.has(tool.tool_name):
                selected.append(tool.tool_name)
            if len(selected) >= self._settings.max_tools_per_intent:
                break

        # If no tools selected but intent requires tools, return all
        if not selected and result.should_enable_tools:
            return self._registry.tool_names

        return tuple(selected)


# === Backward compatibility helpers ===

def should_enable_tools_for_text(text: str) -> bool:
    """Check if tools should be enabled for the given text.

    This is a backward-compatible wrapper around heuristic_classify.

    Args:
        text: User input text

    Returns:
        True if tools should be enabled
    """
    result = heuristic_classify(text, history=None)
    return result.should_enable_tools


def should_expose_run_skill(text: str) -> bool:
    """Check if run_skill tool should be exposed.

    This is a backward-compatible wrapper.

    Args:
        text: User input text

    Returns:
        True if run_skill should be exposed
    """
    return _should_expose_run_skill(text)
