from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document

from .logging_config import get_logger

logger = get_logger(__name__)

HEADER = "# Long-term Memory"
DESCRIPTION = "This file stores important information that should persist across sessions."
FOOTER = (
    "*This file is automatically updated by nanobot when important information "
    "should be remembered.*"
)

SECTION_ORDER = [
    "User Profile",
    "Feedback",
    "Project Knowledge",
    "Reference",
]

SECTION_PLACEHOLDERS = {
    "User Profile": "(Stable facts about the user, habits, and identity-relevant preferences)",
    "Feedback": "(Explicit corrections, likes/dislikes, and response-quality feedback)",
    "Project Knowledge": "(Current project decisions, constraints, conventions, and goals)",
    "Reference": "(Reusable factual references, commands, links, and environment details)",
}

SECTION_ALIASES = {
    "User Information": "User Profile",
    "Preferences": "Feedback",
    "Project Context": "Project Knowledge",
    "Important Notes": "Reference",
}

MEMORY_TYPE_TO_SECTION = {
    "user": "User Profile",
    "profile": "User Profile",
    "user_info": "User Profile",
    "identity": "User Profile",
    "feedback": "Feedback",
    "preference": "Feedback",
    "preferences": "Feedback",
    "correction": "Feedback",
    "project": "Project Knowledge",
    "context": "Project Knowledge",
    "task": "Project Knowledge",
    "decision": "Project Knowledge",
    "constraint": "Project Knowledge",
    "reference": "Reference",
    "fact": "Reference",
    "environment": "Reference",
    "experience": "Reference",
}

SECTION_TO_MEMORY_TYPE = {
    "User Profile": "user",
    "Feedback": "feedback",
    "Project Knowledge": "project",
    "Reference": "reference",
}

SECTION_SCORE_BONUS = {
    "User Profile": 2,
    "Feedback": 3,
    "Project Knowledge": 4,
    "Reference": 1,
}

MAX_ITEMS_PER_SECTION = 120


@dataclass(frozen=True)
class MemoryEntry:
    timestamp: str
    section: str
    memory_type: str
    text: str
    priority: int = 2

    @property
    def line(self) -> str:
        return f"{self.timestamp} [{self.memory_type}|p{self.priority}] {self.text}"

    def to_document(self, *, user_id: str, source: str = "MEMORY.MD") -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "user_id": user_id,
                "section": self.section,
                "memory_type": self.memory_type,
                "priority": self.priority,
                "timestamp": self.timestamp,
                "source": source,
                "display": self.line,
            },
        )


class LongTermMemoryStore:
    def __init__(self, memory_file_path: str = "MEMORY.MD") -> None:
        self._memory_file_path = memory_file_path
        self._lock = threading.Lock()
        logger.info("Initializing local long-term memory store: file=%s", memory_file_path)
        self._ensure_file_exists()

    def retrieve(self, user_id: str, query: str, k: int = 4) -> list[Document]:
        logger.debug("Retrieving long-term memories from file: user_id=%s, k=%s", user_id, k)
        with self._lock:
            sections = self._read_sections()

        candidates: list[tuple[int, MemoryEntry]] = []
        for section in SECTION_ORDER:
            for entry in sections.get(section, []):
                score = _score_entry(entry, query)
                if score > 0:
                    candidates.append((score, entry))

        if not candidates:
            return []

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1].priority,
                item[1].timestamp,
            ),
            reverse=True,
        )
        return [entry.to_document(user_id=user_id) for _, entry in candidates[:k]]

    def retrieve_recent(self, user_id: str, k: int = 4) -> list[Document]:
        logger.debug("Retrieving recent long-term memories: user_id=%s, k=%s", user_id, k)
        with self._lock:
            sections = self._read_sections()

        flattened: list[MemoryEntry] = []
        for section in SECTION_ORDER:
            flattened.extend(sections.get(section, []))
        if not flattened:
            return []

        flattened.sort(key=lambda entry: entry.timestamp, reverse=True)
        picked: list[MemoryEntry] = []
        covered_sections: set[str] = set()
        for entry in flattened:
            if len(picked) >= k:
                break
            if entry.section not in covered_sections:
                picked.append(entry)
                covered_sections.add(entry.section)
        if len(picked) < k:
            seen = {item.line for item in picked}
            for entry in flattened:
                if len(picked) >= k:
                    break
                if entry.line in seen:
                    continue
                picked.append(entry)
                seen.add(entry.line)
        picked.sort(key=lambda entry: entry.timestamp)
        return [entry.to_document(user_id=user_id) for entry in picked]

    def save(
        self,
        user_id: str,
        text: str,
        memory_type: str = "experience",
        priority: int = 2,
    ) -> str:
        clean_text = " ".join((text or "").strip().split())
        if not clean_text:
            logger.warning("Skip saving memory: empty text")
            return ""

        section = _map_section(memory_type)
        normalized_type = _normalize_memory_type(memory_type, section=section)
        safe_priority = min(3, max(0, int(priority)))
        timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        entry = MemoryEntry(
            timestamp=timestamp,
            section=section,
            memory_type=normalized_type,
            text=clean_text,
            priority=safe_priority,
        )
        logger.info(
            "Saving local long-term memory: user_id=%s, section=%s, type=%s, priority=%s, text_len=%s",
            user_id,
            section,
            normalized_type,
            safe_priority,
            len(clean_text),
        )

        with self._lock:
            sections = self._read_sections()
            existing_normalized = {_normalize(item.text) for item in sections[section]}
            if _normalize(clean_text) in existing_normalized:
                logger.debug("Duplicate memory skipped in section=%s", section)
                return _stable_id(clean_text)

            sections[section].append(entry)
            self._compress_sections(sections, max_items_per_section=MAX_ITEMS_PER_SECTION)
            self._write_sections(sections)

        return _stable_id(entry.line)

    def render_context(self, docs: list[Document]) -> str:
        grouped: dict[str, list[str]] = {section: [] for section in SECTION_ORDER}
        for doc in docs:
            metadata = doc.metadata or {}
            section = str(metadata.get("section") or "Reference")
            display = str(metadata.get("display") or doc.page_content).strip()
            if display:
                grouped.setdefault(section, []).append(display)

        blocks: list[str] = []
        for section in SECTION_ORDER:
            items = grouped.get(section) or []
            if not items:
                continue
            blocks.append(f"[{section}]\n" + "\n".join(f"- {item}" for item in items))
        return "\n\n".join(blocks)

    def _ensure_file_exists(self) -> None:
        memory_file = self._memory_file_path
        folder = os.path.dirname(os.path.abspath(memory_file))
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        if os.path.exists(memory_file):
            return

        logger.info("Creating memory file from template: %s", memory_file)
        with open(memory_file, "w", encoding="utf-8") as f:
            f.write(_render_template({name: [] for name in SECTION_ORDER}))

    def _read_sections(self) -> dict[str, list[MemoryEntry]]:
        self._ensure_file_exists()
        with open(self._memory_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        sections: dict[str, list[MemoryEntry]] = {name: [] for name in SECTION_ORDER}
        current_section = ""
        for raw in lines:
            line = raw.rstrip("\n")
            if line.startswith("## "):
                name = _canonical_section(line[3:].strip())
                current_section = name if name in sections else ""
                continue
            if current_section and line.startswith("- "):
                item = line[2:].strip()
                if item and item not in SECTION_PLACEHOLDERS.values():
                    sections[current_section].append(_parse_memory_entry(item, current_section))
        return sections

    def _write_sections(self, sections: dict[str, list[MemoryEntry]]) -> None:
        with open(self._memory_file_path, "w", encoding="utf-8") as f:
            f.write(_render_template(sections))

    @staticmethod
    def _compress_sections(
        sections: dict[str, list[MemoryEntry]],
        max_items_per_section: int,
    ) -> None:
        for section in SECTION_ORDER:
            items = sections.get(section, [])
            if len(items) <= max_items_per_section:
                continue
            items.sort(key=lambda item: (item.priority, item.timestamp), reverse=True)
            kept = items[:max_items_per_section]
            kept.sort(key=lambda item: item.timestamp)
            sections[section] = kept


def _render_template(sections: dict[str, list[MemoryEntry]]) -> str:
    lines: list[str] = [HEADER, "", DESCRIPTION, ""]
    for section in SECTION_ORDER:
        lines.append(f"## {section}")
        lines.append("")
        items = sections.get(section, [])
        if items:
            lines.extend(f"- {item.line}" for item in items)
        else:
            lines.append(SECTION_PLACEHOLDERS[section])
        lines.append("")
    lines.extend(["---", "", FOOTER, ""])
    return "\n".join(lines)


def _parse_memory_entry(raw: str, section: str) -> MemoryEntry:
    pattern = re.compile(
        r"^(?P<timestamp>\S+)\s+\[(?P<memory_type>[a-zA-Z0-9_]+)(?:\|p(?P<priority>[0-3]))?\]\s+(?P<text>.+)$"
    )
    match = pattern.match(raw)
    if match:
        timestamp = match.group("timestamp")
        memory_type = _normalize_memory_type(match.group("memory_type"), section=section)
        priority = int(match.group("priority") or _default_priority_for_type(memory_type))
        text = match.group("text").strip()
        return MemoryEntry(
            timestamp=timestamp,
            section=section,
            memory_type=memory_type,
            text=text,
            priority=priority,
        )

    legacy_pattern = re.compile(r"^(?P<timestamp>\S+)\s+\[(?P<memory_type>[^\]]+)\]\s+(?P<text>.+)$")
    legacy_match = legacy_pattern.match(raw)
    if legacy_match:
        memory_type = _normalize_memory_type(legacy_match.group("memory_type"), section=section)
        return MemoryEntry(
            timestamp=legacy_match.group("timestamp"),
            section=section,
            memory_type=memory_type,
            text=legacy_match.group("text").strip(),
            priority=_default_priority_for_type(memory_type),
        )

    return MemoryEntry(
        timestamp=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        section=section,
        memory_type=SECTION_TO_MEMORY_TYPE.get(section, "reference"),
        text=raw.strip(),
        priority=_default_priority_for_type(SECTION_TO_MEMORY_TYPE.get(section, "reference")),
    )


def _map_section(memory_type: str) -> str:
    key = (memory_type or "").strip().lower()
    return MEMORY_TYPE_TO_SECTION.get(key, "Reference")


def _canonical_section(section: str) -> str:
    normalized = SECTION_ALIASES.get(section, section)
    return normalized if normalized in SECTION_ORDER else "Reference"


def _normalize_memory_type(memory_type: str, *, section: str) -> str:
    key = (memory_type or "").strip().lower()
    if key in MEMORY_TYPE_TO_SECTION:
        return key
    return SECTION_TO_MEMORY_TYPE.get(section, "reference")


def _default_priority_for_type(memory_type: str) -> int:
    if memory_type in {"feedback", "correction", "constraint", "decision"}:
        return 3
    if memory_type in {"project", "user", "profile", "preference"}:
        return 2
    return 1


def _score_entry(entry: MemoryEntry, query: str) -> int:
    score = _score_text(entry.text, query)
    if score <= 0:
        return 0
    score += entry.priority * 3
    score += SECTION_SCORE_BONUS.get(entry.section, 0)
    query_l = (query or "").lower()
    if entry.memory_type in query_l or entry.section.lower() in query_l:
        score += 4
    return score


def _score_text(text: str, query: str) -> int:
    query = (query or "").strip()
    if not query:
        return 0
    if query in text:
        return max(3, len(query))

    text_l = text.lower()
    tokens = _tokenize(query)
    score = 0
    for token in tokens:
        if token and token in text_l:
            score += len(token)
    return score


def _tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower())
    return [token for token in raw_tokens if len(token) >= 2]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
