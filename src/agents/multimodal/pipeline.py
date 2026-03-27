from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path

from .models import MediaItem

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
_DOC_EXTS = {
    ".txt",
    ".md",
    ".json",
    ".csv",
    ".log",
    ".yaml",
    ".yml",
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_mime(path: str, hinted_mime: str = "") -> str:
    if hinted_mime:
        return hinted_mime

    guessed, _ = mimetypes.guess_type(path)
    if guessed:
        return guessed

    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_EXTS:
        return "image/png"
    if suffix in _AUDIO_EXTS:
        return "audio/mpeg"
    if suffix in _VIDEO_EXTS:
        return "video/mp4"
    if suffix in _DOC_EXTS:
        return "text/plain"
    return "application/octet-stream"


def classify_kind(*, path: str, mime: str, hinted_kind: str = "") -> str:
    if hinted_kind in {"image", "audio", "video", "document", "other"}:
        return hinted_kind

    lowered = mime.lower()
    if lowered.startswith("image/"):
        return "image"
    if lowered.startswith("audio/"):
        return "audio"
    if lowered.startswith("video/"):
        return "video"

    suffix = Path(path).suffix.lower()
    if lowered.startswith("text/") or suffix in _DOC_EXTS:
        return "document"
    if suffix in _IMAGE_EXTS:
        return "image"
    if suffix in _AUDIO_EXTS:
        return "audio"
    if suffix in _VIDEO_EXTS:
        return "video"
    return "other"


def to_data_url(*, path: str, mime: str) -> str:
    raw = Path(path).read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def read_text_excerpt(*, path: str, max_chars: int) -> str:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "\n...[truncated]"


def to_media_item(payload: dict[str, object]) -> MediaItem | None:
    path = str(payload.get("path") or "").strip()
    if not path:
        return None

    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None

    mime = detect_mime(path, str(payload.get("mime") or "").strip())
    kind = classify_kind(path=path, mime=mime, hinted_kind=str(payload.get("kind") or "").strip())
    size_bytes = file_path.stat().st_size
    sha256 = str(payload.get("sha256") or "").strip() or file_sha256(path)

    raw_meta = payload.get("meta")
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    return MediaItem(
        id=str(payload.get("id") or file_path.stem),
        path=str(file_path),
        mime=mime,
        kind=kind,  # type: ignore[arg-type]
        size_bytes=size_bytes,
        sha256=sha256,
        meta=meta,
    )
