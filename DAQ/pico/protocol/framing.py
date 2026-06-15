"""pico/protocol/framing.py – COBS (Consistent Overhead Byte Stuffing) framing.

COBS eliminates all 0x00 bytes from the encoded payload, allowing 0x00 to
be used as an unambiguous packet delimiter on a byte stream.

Reference: Cheshire & Baker (1999) "Consistent Overhead Byte Stuffing"
           IEEE/ACM Transactions on Networking 7(2):159–172.

Frame format on the wire
------------------------
    0x00  COBS_ENCODED(LEN:1 | SEQ:2 | TYPE:1 | PAYLOAD | CRC16:2)  0x00

The leading 0x00 is optional (skip any 0x00 bytes until the first non-zero
byte to synchronise); the trailing 0x00 is the definitive end-of-frame.

Public API
----------
    encode(data)  → bytes   COBS-encode *data* (no 0x00 in output, no delimiters)
    decode(data)  → bytes   COBS-decode *data* (raises ValueError on bad frame)
    wrap(data)    → bytes   encode + add leading/trailing 0x00 delimiters
    unwrap(frame) → bytes   strip delimiters + decode
"""


def encode(data: bytes | bytearray) -> bytes:
    """COBS-encode *data*.

    Parameters
    ----------
    data : bytes | bytearray
        Raw payload (may contain 0x00 bytes).

    Returns
    -------
    bytes
        Encoded bytes with no 0x00 bytes.  Length is at most
        ``len(data) + len(data) // 254 + 1``.
    """
    output = bytearray()
    data = bytes(data)
    idx = 0
    while idx <= len(data):
        code_pos = len(output)
        output.append(0)          # placeholder for the overhead byte
        code = 1
        while idx < len(data) and data[idx] != 0x00:
            output.append(data[idx])
            idx += 1
            code += 1
            if code == 0xFF:
                output[code_pos] = code
                code_pos = len(output)
                output.append(0)
                code = 1
        output[code_pos] = code
        idx += 1                  # skip over the 0x00 (or the sentinel)
    return bytes(output)


def decode(data: bytes | bytearray) -> bytes:
    """COBS-decode *data*.

    Parameters
    ----------
    data : bytes | bytearray
        COBS-encoded bytes (no delimiters).

    Returns
    -------
    bytes
        Decoded payload (may contain 0x00 bytes).

    Raises
    ------
    ValueError
        If the encoded data is malformed (e.g. a zero byte inside the
        payload or an unexpected end of data).
    """
    data = bytes(data)
    if not data:
        return b""
    output = bytearray()
    idx = 0
    while idx < len(data):
        code = data[idx]
        if code == 0:
            raise ValueError("COBS decode: unexpected 0x00 byte inside frame")
        idx += 1
        for _ in range(code - 1):
            if idx >= len(data):
                raise ValueError("COBS decode: truncated frame")
            output.append(data[idx])
            idx += 1
        if code < 0xFF and idx < len(data):
            output.append(0x00)
    return bytes(output)


def wrap(data: bytes | bytearray) -> bytes:
    """COBS-encode *data* and surround with 0x00 frame delimiters."""
    return b"\x00" + encode(data) + b"\x00"


def unwrap(frame: bytes | bytearray) -> bytes:
    """Strip leading/trailing 0x00 delimiters and COBS-decode.

    Parameters
    ----------
    frame : bytes | bytearray
        Raw bytes as received from the UART, including the 0x00 delimiters.

    Raises
    ------
    ValueError
        If the frame is empty or malformed.
    """
    frame = bytes(frame).strip(b"\x00")
    if not frame:
        raise ValueError("COBS unwrap: empty frame")
    return decode(frame)
