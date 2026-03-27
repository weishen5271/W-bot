from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InboundMedia:
    id: str
    path: str
    mime: str
    kind: str
    size_bytes: int
    sha256: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "mime": self.mime,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class InboundMessage:
    content: str
    media: list[InboundMedia] = field(default_factory=list)

    def to_human_content(self) -> dict[str, Any]:
        return {
            "text": self.content,
            "media": [item.to_dict() for item in self.media],
        }
