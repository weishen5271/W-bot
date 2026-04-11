"""Intent type definitions and data structures for W-bot intent classification."""

from dataclasses import dataclass, field
from enum import Enum, auto


class IntentType(Enum):
    """Intent type enumeration covering all possible user intents."""

    # === Social / Casual ===
    CASUAL_CHAT = auto()  # Casual greeting (你好, bye, 谢谢, etc.)
    CAPABILITY_QUESTION = auto()  # Asking about bot capabilities (能读取文件吗?)

    # === File Operations ===
    FILE_READ = auto()  # Read file/directory
    FILE_EDIT = auto()  # Modify/edit file
    FILE_CREATE = auto()  # Create new file

    # === Execution Operations ===
    COMMAND_EXEC = auto()  # Execute command/terminal
    WEB_SEARCH = auto()  # Internet search
    WEB_FETCH = auto()  # Fetch webpage content

    # === Delegation / Concurrency ===
    SPAWN_SUBAGENT = auto()  # Spawn sub-agent
    RUN_SKILL = auto()  # Run skill
    PARALLEL_TASK = auto()  # Parallel task

    # === Scheduling ===
    CRON_SCHEDULE = auto()  # Scheduled task
    MESSAGE_SEND = auto()  # Send message

    # === Project Analysis ===
    PROJECT_INSPECTION = auto()  # Project code understanding

    # === General ===
    GENERAL_TASK = auto()  # General task that doesn't fit other categories
    NO_TOOL_NEEDED = auto()  # No tool needed (e.g., pure Q&A)
    UNKNOWN = auto()  # Unknown intent


@dataclass(frozen=True)
class IntentDecision:
    """Single intent decision result."""

    intent: IntentType
    confidence: float  # 0.0 ~ 1.0
    reasoning: str = ""


@dataclass(frozen=True)
class ToolRecommendation:
    """Tool recommendation result."""

    tool_name: str
    exposure_level: float = 1.0  # 1.0=must expose, 0.5=optional, 0=do not expose
    reason: str = ""


@dataclass(frozen=True)
class IntentResult:
    """Two-stage intent classification final result."""

    primary_intent: IntentDecision
    secondary_intents: tuple[IntentDecision, ...] = field(default_factory=tuple)
    should_enable_tools: bool = True
    recommended_tools: tuple[ToolRecommendation, ...] = field(default_factory=tuple)
    requires_llm: bool = False
    metadata: dict = field(default_factory=dict)

    @classmethod
    def default_unknown(cls) -> "IntentResult":
        """Return a default 'unknown' result with conservative defaults."""
        return cls(
            primary_intent=IntentDecision(IntentType.UNKNOWN, confidence=0.0),
            should_enable_tools=True,
            requires_llm=False,
            metadata={"source": "default_unknown"},
        )

    @classmethod
    def casual_chat(cls) -> "IntentResult":
        """Return a casual chat result (no tools needed)."""
        return cls(
            primary_intent=IntentDecision(IntentType.CASUAL_CHAT, confidence=0.95),
            should_enable_tools=False,
            requires_llm=False,
            metadata={"source": "heuristic"},
        )

    @classmethod
    def capability_question(cls) -> "IntentResult":
        """Return a capability question result (no tools needed)."""
        return cls(
            primary_intent=IntentDecision(IntentType.CAPABILITY_QUESTION, confidence=0.90),
            should_enable_tools=False,
            requires_llm=False,
            metadata={"source": "heuristic"},
        )
