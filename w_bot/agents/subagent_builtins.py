from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BuiltinSubagentDefinition:
    key: str
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    max_turns: int = 12
    read_only: bool = False


BUILTIN_SUBAGENTS: dict[str, BuiltinSubagentDefinition] = {
    "worker": BuiltinSubagentDefinition(
        key="worker",
        name="Worker",
        description="General-purpose execution agent for independent tasks.",
        system_prompt=(
            "You are a focused child agent.\n"
            "Complete the delegated task independently, use tools when needed, "
            "and return a concise summary with concrete results."
        ),
        max_turns=12,
    ),
    "explore": BuiltinSubagentDefinition(
        key="explore",
        name="Explore",
        description="Read-only research agent for code and docs exploration.",
        system_prompt=(
            "You are a read-only research agent.\n"
            "Investigate the delegated task, gather evidence from files and commands, "
            "and return structured findings without modifying files."
        ),
        allowed_tools=["read_file", "list_dir", "web_search", "web_fetch"],
        disallowed_tools=["write_file", "edit_file", "save_memory", "spawn"],
        max_turns=10,
        read_only=True,
    ),
    "plan": BuiltinSubagentDefinition(
        key="plan",
        name="Plan",
        description="Planning agent for implementation steps and tradeoffs.",
        system_prompt=(
            "You are a planning agent.\n"
            "Analyze the task, inspect only what is necessary, and produce an actionable plan "
            "with risks, affected areas, and recommended next steps."
        ),
        allowed_tools=["read_file", "list_dir"],
        disallowed_tools=["write_file", "edit_file", "save_memory", "spawn"],
        max_turns=8,
        read_only=True,
    ),
    "verify": BuiltinSubagentDefinition(
        key="verify",
        name="Verify",
        description="Verification agent for validation and checks.",
        system_prompt=(
            "You are a verification agent.\n"
            "Validate the delegated work objectively, run safe checks when useful, "
            "and report pass/fail findings with evidence."
        ),
        allowed_tools=["read_file", "list_dir", "exec"],
        disallowed_tools=["write_file", "edit_file", "save_memory", "spawn"],
        max_turns=10,
        read_only=True,
    ),
}
