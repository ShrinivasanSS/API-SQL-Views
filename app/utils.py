"""Utility helpers shared across the application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def safe_resolve(base: Path, relative: str) -> Path:
    """Resolve a relative path within a base directory and guard against escape."""

    candidate = (base / relative).resolve()
    base_resolved = base.resolve()
    if not str(candidate).startswith(str(base_resolved)):
        raise ValueError("Attempted to access a path outside the allowed directory")
    return candidate


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def preview_json(path: Path, *, limit: int = 280) -> str:
    text = path.read_text()
    return text[:limit] + ("…" if len(text) > limit else "")


__all__ = ["safe_resolve", "load_json", "preview_json"]
