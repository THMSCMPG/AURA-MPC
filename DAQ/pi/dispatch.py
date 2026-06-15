"""pi/dispatch.py – Forward SensorPacket JSON to downstream consumers.

Three output modes are supported (selected at construction time):

    ``stdout`` – one JSON object per line to ``sys.stdout`` (or a caller-
                 supplied text stream).  This is the canonical transport
                 for piping into
                 ``python -m scripts.predict --mode live`` on the
                 PINN-AURA-MFP orchestrator.

    ``socket`` – newline-delimited JSON over a TCP connection.  Uses a
                 persistent socket when possible and silently reconnects
                 on errors so transient network blips don't lose data.

    ``file``   – newline-delimited JSON to a rotating file for offline
                 replay.  Rotation is size-based (``max_bytes``); old
                 files are renamed ``<path>.1``, ``<path>.2``, …
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any, Optional, TextIO

log = logging.getLogger(__name__)


class Dispatcher:
    """Send SensorPacket JSON to the downstream orchestrator."""

    VALID_MODES = ("stdout", "socket", "file")

    def __init__(
        self,
        mode: str = "stdout",
        *,
        host: str = "127.0.0.1",
        port: int = 9000,
        path: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        sink: Optional[TextIO] = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown dispatch mode: {mode!r} (valid: {self.VALID_MODES})"
            )
        self._mode = mode
        self._host = host
        self._port = port
        self._path = Path(path) if path else None
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._sink = sink if sink is not None else sys.stdout
        self._sock: Optional[socket.socket] = None
        self._file_handle: Optional[TextIO] = None

    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    def send(self, packet: dict[str, Any]) -> None:
        """Serialise *packet* and write it according to the configured mode.

        Blocking (simple, synchronous).  Callers that need non-blocking
        behaviour can run the dispatcher in a background thread.
        """
        line = json.dumps(packet, separators=(",", ":")) + "\n"
        if self._mode == "stdout":
            self._send_stream(line)
        elif self._mode == "socket":
            self._send_socket(line)
        else:
            self._send_file(line)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            finally:
                self._file_handle = None

    def __enter__(self) -> "Dispatcher":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ─── Backends ────────────────────────────────────────────────────
    def _send_stream(self, line: str) -> None:
        try:
            self._sink.write(line)
            self._sink.flush()
        except OSError as exc:
            log.error("stdout write failed: %s", exc)

    def _send_socket(self, line: str) -> None:
        data = line.encode("utf-8")
        for attempt in (1, 2):
            if self._sock is None:
                try:
                    self._sock = socket.create_connection(
                        (self._host, self._port), timeout=5
                    )
                except OSError as exc:
                    log.error("socket connect %s:%d failed: %s",
                              self._host, self._port, exc)
                    return
            try:
                self._sock.sendall(data)
                return
            except OSError as exc:
                log.warning("socket send failed (attempt %d): %s", attempt, exc)
                try:
                    self._sock.close()
                finally:
                    self._sock = None
        log.error("giving up on packet after 2 socket-send attempts")

    def _send_file(self, line: str) -> None:
        if self._path is None:
            raise RuntimeError("file-mode dispatcher requires a path")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._file_handle is None:
            self._file_handle = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        self._file_handle.write(line)
        self._file_handle.flush()
        # Rotate if needed.
        try:
            size = os.fstat(self._file_handle.fileno()).st_size
        except OSError:
            return
        if size >= self._max_bytes:
            self._rotate()

    def _rotate(self) -> None:
        if self._file_handle is None or self._path is None:
            return
        self._file_handle.close()
        self._file_handle = None
        for i in range(self._backup_count, 0, -1):
            src = self._path.with_suffix(self._path.suffix + f".{i}")
            dst = self._path.with_suffix(self._path.suffix + f".{i + 1}")
            if src.exists():
                if i == self._backup_count:
                    src.unlink()
                else:
                    src.rename(dst)
        rotated = self._path.with_suffix(self._path.suffix + ".1")
        self._path.rename(rotated)
        self._file_handle = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
