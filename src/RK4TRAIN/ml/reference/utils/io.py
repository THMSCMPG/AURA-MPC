"""Atomic JSON I/O helpers using :mod:`pathlib`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: Path) -> None:
    """Atomically write ``obj`` as UTF-8 JSON to ``path``.

    The payload is first written to a sibling ``<name>.tmp`` file and then
    :func:`os.replace`-renamed into place, so readers never observe a
    half-written file.

    Args:
        obj: JSON-serialisable object.
        path: Destination path. Parent directories are created as needed.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, dest)


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file.

    Args:
        path: Source path.

    Returns:
        The decoded JSON object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    src = Path(path)
    return json.loads(src.read_text(encoding="utf-8"))
