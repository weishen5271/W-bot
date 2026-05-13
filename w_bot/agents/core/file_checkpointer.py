from __future__ import annotations

from pathlib import Path


def resolve_short_term_memory_path(configured_path: str) -> str:
    """Resolve the legacy short-term memory path to the current SQLite location."""

    target = Path(configured_path).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if target.suffix.lower() == ".pkl":
        target = target.with_suffix(".sqlite")
    elif not target.suffix:
        target = target.with_suffix(".sqlite")
    return str(target)


__all__ = ["resolve_short_term_memory_path"]
