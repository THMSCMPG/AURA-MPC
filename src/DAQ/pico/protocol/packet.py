"""pico/protocol/packet.py – Binary sensor frame pack / unpack (v1).

Frame layout (before COBS wrapping)
-----------------------------------

    Offset  Size  Field
    ------  ----  ------------------------------------------------
    0       1     SYNC1      = 0xAA
    1       1     SYNC2      = 0x55
    2       1     VERSION    = 0x01
    3       2     LENGTH     – uint16 big-endian, payload length (always 33 for v1)
    5       33    PAYLOAD    – see :data:`PAYLOAD_FMT`
    38      2     CRC16      – CRC-16/CCITT-FALSE over VERSION+LENGTH+PAYLOAD, big-endian

Total: 40 bytes per frame.

Payload struct (version 1)
--------------------------

    Offset  Size  Field                     Type      Fault sentinel
    ------  ----  ------------------------  --------  ---------------
    0       8     timestamp_ms_since_epoch  uint64    –
    8       2     pyranometer_raw_counts    uint16    0
    10      16    thermocouple_raw[4]       int32[4]  INT32_MIN (-2147483648)
    26      2     anemometer_speed_x100     uint16    0   (m/s × 100)
    28      2     anemometer_dir_deg        uint16    0   (0-359)
    30      2     fault_flags               uint16    bitmask – see :data:`FAULT_*`
    32      1     reserved                  uint8     0

Fault flag bitmask (``fault_flags``)
------------------------------------

    bit 0  (0x0001)  pyranometer
    bit 1  (0x0002)  thermocouple 0
    bit 2  (0x0004)  thermocouple 1
    bit 3  (0x0008)  thermocouple 2
    bit 4  (0x0010)  thermocouple 3
    bit 5  (0x0020)  anemometer
    bit 6  (0x0040)  RTC
    bit 15 (0x8000)  persistent-fault indicator (≥10 consecutive errors)

This module is pure Python and contains no ``machine`` imports, so it can
be imported from both MicroPython (Pico) and CPython (Pi 3B+ daemon and
the dev-machine test suite).
"""

import struct

from .checksum import crc16

# ── Frame constants ────────────────────────────────────────────────────────
SYNC1   = 0xAA
SYNC2   = 0x55
VERSION = 0x01

HEADER_FMT  = ">BBBH"            # SYNC1, SYNC2, VERSION, LENGTH
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 5 bytes
CRC_SIZE    = 2
OVERHEAD    = HEADER_SIZE + CRC_SIZE         # 7 bytes

# Payload: >Q H 4i H H H B  →  8 + 2 + 16 + 2 + 2 + 2 + 1 = 33
PAYLOAD_FMT  = ">QH4iHHHB"
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FMT)  # 33
assert PAYLOAD_SIZE == 33, "payload spec mismatch"

FRAME_SIZE = OVERHEAD + PAYLOAD_SIZE          # 40

# ── Fault sentinels / bitmask ──────────────────────────────────────────────
PYRANOMETER_FAULT_SENTINEL = 0
THERMOCOUPLE_FAULT_SENTINEL = -(1 << 31)     # INT32_MIN  = -2147483648
ANEMOMETER_SPEED_FAULT_SENTINEL = 0
ANEMOMETER_DIR_FAULT_SENTINEL = 0

FAULT_PYRANOMETER  = 0x0001
FAULT_TC0          = 0x0002
FAULT_TC1          = 0x0004
FAULT_TC2          = 0x0008
FAULT_TC3          = 0x0010
FAULT_ANEMOMETER   = 0x0020
FAULT_RTC          = 0x0040
FAULT_PERSISTENT   = 0x8000

FAULT_TC_BITS = (FAULT_TC0, FAULT_TC1, FAULT_TC2, FAULT_TC3)


class FrameError(ValueError):
    """Raised for malformed frames (sync mismatch, bad CRC, length, version)."""


def pack(
    timestamp_ms: int,
    pyranometer_raw: int,
    thermocouple_raw,          # sequence of 4 ints
    anemometer_speed_x100: int,
    anemometer_dir_deg: int,
    fault_flags: int = 0,
    reserved: int = 0,
) -> bytes:
    """Pack one aggregated sensor sample into a 40-byte wire frame.

    Parameters
    ----------
    timestamp_ms : int
        Milliseconds since UNIX epoch (uint64).
    pyranometer_raw : int
        Raw ADC counts (0-65535). Must be 0 if sensor is faulted.
    thermocouple_raw : sequence of 4 int
        Raw linearised counts for TC0..TC3. Use INT32_MIN on fault.
    anemometer_speed_x100 : int
        Wind speed in m/s × 100 (uint16).
    anemometer_dir_deg : int
        Wind direction 0–359 (uint16).
    fault_flags : int
        Bitmask (uint16). See module-level ``FAULT_*`` constants.
    reserved : int
        Reserved byte (default 0).

    Returns
    -------
    bytes
        Exactly 40 bytes: header + payload + CRC.
    """
    tc = tuple(thermocouple_raw)
    if len(tc) != 4:
        raise ValueError(f"thermocouple_raw must have 4 elements, got {len(tc)}")

    payload = struct.pack(
        PAYLOAD_FMT,
        timestamp_ms & 0xFFFFFFFFFFFFFFFF,
        pyranometer_raw & 0xFFFF,
        tc[0], tc[1], tc[2], tc[3],
        anemometer_speed_x100 & 0xFFFF,
        anemometer_dir_deg & 0xFFFF,
        fault_flags & 0xFFFF,
        reserved & 0xFF,
    )

    header = struct.pack(HEADER_FMT, SYNC1, SYNC2, VERSION, PAYLOAD_SIZE)
    crc = crc16(header[2:] + payload)             # over VERSION+LENGTH+PAYLOAD
    return header + payload + struct.pack(">H", crc)


def unpack(frame: bytes) -> dict:
    """Unpack a 40-byte frame produced by :func:`pack`.

    Parameters
    ----------
    frame : bytes
        Raw frame bytes (no COBS, no 0x00 delimiters).

    Returns
    -------
    dict
        Keys: ``timestamp_ms``, ``pyranometer_raw``, ``thermocouple_raw`` (list of 4),
        ``anemometer_speed_x100``, ``anemometer_dir_deg``, ``fault_flags``,
        ``reserved``, ``version``.

    Raises
    ------
    FrameError
        If sync bytes, version, length, or CRC are invalid.
    """
    if len(frame) < OVERHEAD:
        raise FrameError(f"Frame too short: {len(frame)} bytes")

    sync1, sync2, version, length = struct.unpack_from(HEADER_FMT, frame, 0)
    if sync1 != SYNC1 or sync2 != SYNC2:
        raise FrameError(f"Bad sync: {sync1:#04x} {sync2:#04x}")
    if version != VERSION:
        raise FrameError(f"Unsupported version: {version:#04x}")

    expected_len = HEADER_SIZE + length + CRC_SIZE
    if len(frame) != expected_len:
        raise FrameError(
            f"Length mismatch: header says payload={length} "
            f"(frame should be {expected_len} B), got {len(frame)} B"
        )
    if length != PAYLOAD_SIZE:
        raise FrameError(f"Unexpected payload length for v1: {length}")

    payload = frame[HEADER_SIZE:HEADER_SIZE + length]
    recv_crc = struct.unpack_from(">H", frame, HEADER_SIZE + length)[0]
    calc_crc = crc16(frame[2:HEADER_SIZE + length])   # VERSION+LENGTH+PAYLOAD
    if recv_crc != calc_crc:
        raise FrameError(f"CRC mismatch: {recv_crc:#06x} != {calc_crc:#06x}")

    (
        ts_ms, pyr, tc0, tc1, tc2, tc3, spd, direction, flags, reserved,
    ) = struct.unpack(PAYLOAD_FMT, payload)

    return {
        "version":               version,
        "timestamp_ms":          ts_ms,
        "pyranometer_raw":       pyr,
        "thermocouple_raw":      [tc0, tc1, tc2, tc3],
        "anemometer_speed_x100": spd,
        "anemometer_dir_deg":    direction,
        "fault_flags":           flags,
        "reserved":              reserved,
    }
