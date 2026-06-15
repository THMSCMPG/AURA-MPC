"""pico/protocol/checksum.py – CRC-16/CCITT-FALSE checksum.

Polynomial : 0x1021 (x^16 + x^12 + x^5 + 1)
Init value : 0xFFFF
Input/output reflection: False

This is the same variant used by Xmodem and is sometimes called CRC-CCITT.
"""


def crc16(data: bytes | bytearray, init: int = 0xFFFF) -> int:
    """Compute CRC-16/CCITT-FALSE over *data*.

    Parameters
    ----------
    data : bytes | bytearray
        Input bytes.
    init : int
        Initial CRC value (default 0xFFFF).

    Returns
    -------
    int
        16-bit CRC value (0–65535).

    Examples
    --------
    >>> crc16(b"123456789")
    0x29B1
    """
    crc = init & 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
        crc &= 0xFFFF
    return crc


def crc16_bytes(data: bytes | bytearray) -> bytes:
    """Return CRC-16 as a 2-byte big-endian value."""
    value = crc16(data)
    return bytes([value >> 8, value & 0xFF])
