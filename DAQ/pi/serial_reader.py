"""pi/serial_reader.py – Read framed sensor packets from a byte stream.

Supports two input sources, chosen at construction time:

    * ``SerialReader(port="/dev/serial0", baud=115200)`` – live UART link
      to the Pico.  Uses ``pyserial``.
    * ``SerialReader.from_file(path)`` – deterministic replay from a
      pre-recorded binary file.  Requires no hardware and no
      ``pyserial`` install.

Each yielded item is the ``dict`` produced by
:func:`pico.protocol.packet.unpack` – or ``None`` if the frame was
corrupted (CRC/length/sync/version mismatch) so the daemon can bump its
fault counter without dropping the iteration.

A single bit-flip in the middle of a frame will be caught by either the
COBS layer (inconsistent overhead byte) or the CRC check; either way the
reader resyncs at the next ``0x00`` delimiter, so a corrupted byte
**cannot desync the reader**.
"""

from __future__ import annotations

import logging
from typing import BinaryIO, Iterable, Iterator, Optional

from pico.protocol import framing, packet

log = logging.getLogger(__name__)

_FRAME_MAX_BYTES = 512    # safety: a runaway non-zero stream gets discarded


class SerialReader:
    """Iterate sensor-frame dicts from a byte stream (UART or file)."""

    def __init__(
        self,
        port: str = "/dev/serial0",
        baud: int = 115_200,
        timeout: float = 1.0,
        stream: Optional[BinaryIO] = None,
    ) -> None:
        self._port = port
        self._baud = baud
        self._timeout = timeout
        self._stream = stream        # if non-None, file-replay mode
        self._serial = None
        self._buf = bytearray()
        self._crc_errors = 0
        self._frames_ok = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "SerialReader":
        """Open *path* and read frames from it (no UART, no pyserial)."""
        stream = open(path, "rb")                       # noqa: SIM115 (kept by __exit__)
        return cls(stream=stream)

    @classmethod
    def from_stream(cls, stream: BinaryIO) -> "SerialReader":
        """Read frames from any pre-opened binary stream (e.g. BytesIO)."""
        return cls(stream=stream)

    @classmethod
    def from_stdin(cls) -> "SerialReader":
        """Read framed bytes from ``sys.stdin.buffer`` (pipe mode).

        Intended for the end-to-end test where the Pico firmware runs in
        a mock subprocess piping its UART bytes into the daemon's stdin.
        """
        import sys as _sys
        stream = _sys.stdin.buffer
        return cls(stream=stream)

    # ------------------------------------------------------------------
    def open(self) -> None:
        if self._stream is not None:
            return
        import serial  # pyserial; imported lazily so file-replay mode has no hard dep
        self._serial = serial.Serial(self._port, self._baud, timeout=self._timeout)
        log.info("Serial port %s opened at %d baud", self._port, self._baud)

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None
        if self._stream is not None:
            try:
                self._stream.close()
            finally:
                self._stream = None

    def __enter__(self) -> "SerialReader":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    @property
    def crc_errors(self) -> int:
        return self._crc_errors

    @property
    def frames_ok(self) -> int:
        return self._frames_ok

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Optional[dict]]:
        """Yield a decoded frame dict per valid packet; ``None`` on corruption.

        For stream mode iteration ends at EOF.  For UART mode iteration
        is unbounded (the daemon stops it with SIGINT/SIGTERM).
        """
        if self._stream is not None:
            yield from self._iter_stream()
            return
        if self._serial is None:
            raise RuntimeError("SerialReader is not open; call open() first")
        yield from self._iter_serial()

    # ------------------------------------------------------------------
    def _iter_stream(self) -> Iterator[Optional[dict]]:
        while True:
            chunk = self._stream.read(4096)  # type: ignore[union-attr]
            if not chunk:
                # Drain anything still in the buffer.
                yield from self._drain()
                return
            self._buf.extend(chunk)
            yield from self._drain()

    def _iter_serial(self) -> Iterator[Optional[dict]]:
        while True:
            chunk = self._serial.read(64)  # type: ignore[union-attr]
            if not chunk:
                continue
            self._buf.extend(chunk)
            yield from self._drain()

    # ------------------------------------------------------------------
    def _drain(self) -> Iterator[Optional[dict]]:
        """Extract and decode every complete 0x00-delimited frame."""
        while True:
            # skip leading 0x00 bytes (inter-frame padding)
            while self._buf and self._buf[0] == 0x00:
                del self._buf[0]
            if not self._buf:
                return
            end = self._buf.find(0x00)
            if end == -1:
                if len(self._buf) > _FRAME_MAX_BYTES:
                    log.warning("oversized run of non-zero bytes – discarding %d bytes",
                                len(self._buf))
                    self._buf.clear()
                return
            cobs_bytes = bytes(self._buf[:end])
            del self._buf[:end + 1]          # consume trailing 0x00 too
            if not cobs_bytes:
                continue
            yield self._decode_frame(cobs_bytes)

    def _decode_frame(self, cobs_bytes: bytes) -> Optional[dict]:
        try:
            raw = framing.decode(cobs_bytes)
            result = packet.unpack(raw)
        except (ValueError, packet.FrameError) as exc:
            self._crc_errors += 1
            log.warning("bad frame: %s", exc)
            return None
        self._frames_ok += 1
        return result


# ──────────────────────────────────────────────────────────────────────
def frames_from_file(path: str) -> Iterable[Optional[dict]]:
    """Convenience helper: yield every frame in *path* then close the file."""
    with SerialReader.from_file(path) as reader:
        yield from reader
