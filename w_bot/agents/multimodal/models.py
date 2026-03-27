from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MediaKind = Literal["image", "audio", "video", "document", "other"]


@dataclass(frozen=True)
class MediaItem:
    id: str
    path: str
    mime: str
    kind: MediaKind
    size_bytes: int
    sha256: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactRef:
    media_id: str
    kind: str
    value: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityDecision:
    media_id: str
    use_native: bool
    reason: str


@dataclass
class NormalizedUserContent:
    blocks: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    decisions: list[CapabilityDecision] = field(default_factory=list)
