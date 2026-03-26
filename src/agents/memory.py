from __future__ import annotations

import hashlib
import os
import re
import threading
from datetime import datetime, timezone

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
    "User Information",
    "Preferences",
    "Project Context",
    "Important Notes",
]

SECTION_PLACEHOLDERS = {
    "User Information": "(Important facts about the user)",
    "Preferences": "(User preferences learned over time)",
    "Project Context": "(Information about ongoing projects)",
    "Important Notes": "(Things to remember)",
}


class LongTermMemoryStore:
    def __init__(self, memory_file_path: str = "MEMORY.MD") -> None:
        self._memory_file_path = memory_file_path
        self._lock = threading.Lock()
        logger.info("Initializing local long-term memory store: file=%s", memory_file_path)
        self._ensure_file_exists()

    def retrieve(self, user_id: str, query: str, k: int = 4) -> list[Document]:
        logger.debug(
            "Retrieving long-term memories from file: user_id=%s, k=%s", user_id, k
        )
        with self._lock:
            sections = self._read_sections()

        candidates: list[tuple[int, str, str]] = []
        for section, items in sections.items():
            for item in items:
                score = _score_text(item, query)
                if score > 0:
                    candidates.append((score, section, item))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)
        docs: list[Document] = []
        for _, section, item in candidates[:k]:
            docs.append(
                Document(
                    page_content=item,
                    metadata={"user_id": user_id, "section": section, "source": "MEMORY.MD"},
                )
            )
        return docs

    def retrieve_recent(self, user_id: str, k: int = 4) -> list[Document]:
        logger.debug("Retrieving recent long-term memories: user_id=%s, k=%s", user_id, k)
        with self._lock:
            sections = self._read_sections()

        candidates: list[tuple[str, str]] = []
        for section in SECTION_ORDER:
            for item in sections.get(section, []):
                candidates.append((section, item))

        if not candidates:
            return []

        docs: list[Document] = []
        for section, item in candidates[-k:]:
            docs.append(
                Document(
                    page_content=item,
                    metadata={"user_id": user_id, "section": section, "source": "MEMORY.MD"},
                )
            )
        return docs

    def save(self, user_id: str, text: str, memory_type: str = "experience") -> str:
        clean_text = " ".join((text or "").strip().split())
        if not clean_text:
            logger.warning("Skip saving memory: empty text")
            return ""

        section = _map_section(memory_type)
        timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        entry = f"{timestamp} [{memory_type}] {clean_text}"
        logger.info(
            "Saving local long-term memory: user_id=%s, section=%s, text_len=%s",
            user_id,
            section,
            len(clean_text),
        )

        with self._lock:
            sections = self._read_sections()
            existing_normalized = {_normalize(item) for item in sections[section]}
            if _normalize(clean_text) in existing_normalized:
                logger.debug("Duplicate memory skipped in section=%s", section)
                return _stable_id(clean_text)

            sections[section].append(entry)
            self._compress_sections(sections, max_items_per_section=80)
            self._write_sections(sections)

        return _stable_id(entry)

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

    def _read_sections(self) -> dict[str, list[str]]:
        self._ensure_file_exists()
        with open(self._memory_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        sections: dict[str, list[str]] = {name: [] for name in SECTION_ORDER}
        current_section = ""
        for raw in lines:
            line = raw.rstrip("\n")
            if line.startswith("## "):
                name = line[3:].strip()
                current_section = name if name in sections else ""
                continue
            if current_section and line.startswith("- "):
                item = line[2:].strip()
                if item:
                    sections[current_section].append(item)
        return sections

    def _write_sections(self, sections: dict[str, list[str]]) -> None:
        with open(self._memory_file_path, "w", encoding="utf-8") as f:
            f.write(_render_template(sections))

    @staticmethod
    def _compress_sections(
        sections: dict[str, list[str]],
        max_items_per_section: int,
    ) -> None:
        for section in SECTION_ORDER:
            items = sections.get(section, [])
            if len(items) <= max_items_per_section:
                continue
            sections[section] = items[-max_items_per_section:]


def _render_template(sections: dict[str, list[str]]) -> str:
    lines: list[str] = [HEADER, "", DESCRIPTION, ""]
    for section in SECTION_ORDER:
        lines.append(f"## {section}")
        lines.append("")
        items = sections.get(section, [])
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append(SECTION_PLACEHOLDERS[section])
        lines.append("")
    lines.extend(["---", "", FOOTER, ""])
    return "\n".join(lines)


def _map_section(memory_type: str) -> str:
    key = (memory_type or "").strip().lower()
    if key in {"preference", "preferences"}:
        return "Preferences"
    if key in {"user", "profile", "user_info"}:
        return "User Information"
    if key in {"project", "context", "task"}:
        return "Project Context"
    return "Important Notes"


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
    return " ".join(text.strip().lower().split())


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
