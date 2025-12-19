"""
holo.tnc.frame

Lightweight framing for TNC-style byte streams.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional


PREAMBLE_BYTE = 0x55
SYNC_WORD = b"\x2d\xd4"
FRAME_HDR = struct.Struct(">HBB")  # length, flags, reserved
CRC_STRUCT = struct.Struct(">H")


def crc16_ccitt(data: bytes, *, init: int = 0xFFFF, poly: int = 0x1021) -> int:
    crc = int(init) & 0xFFFF
    for b in data:
        crc ^= (int(b) & 0xFF) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def build_frame(
    payload: bytes,
    *,
    flags: int = 0,
    preamble_len: int = 16,
    sync: bytes = SYNC_WORD,
) -> bytes:
    payload = bytes(payload)
    hdr = FRAME_HDR.pack(int(len(payload)), int(flags), 0)
    crc = crc16_ccitt(hdr + payload)
    preamble = bytes([PREAMBLE_BYTE]) * int(preamble_len)
    return preamble + sync + hdr + payload + CRC_STRUCT.pack(int(crc))


@dataclass
class FrameDecoder:
    """
    Incremental frame decoder for byte streams.
    """

    max_payload: int = 16384
    sync: bytes = SYNC_WORD

    def __post_init__(self) -> None:
        self._buf = bytearray()

    def push(self, data: bytes) -> List[bytes]:
        if data:
            self._buf.extend(data)
        out: List[bytes] = []
        sync = self.sync
        sync_len = len(sync)
        min_len = sync_len + FRAME_HDR.size + CRC_STRUCT.size

        while True:
            idx = self._buf.find(sync)
            if idx < 0:
                if len(self._buf) > sync_len:
                    self._buf = self._buf[-(sync_len - 1):]
                break
            if idx > 0:
                del self._buf[:idx]
            if len(self._buf) < min_len:
                break

            off = sync_len
            try:
                length, flags, _res = FRAME_HDR.unpack_from(self._buf, off)
            except struct.error:
                break
            length = int(length)
            if length < 0 or length > int(self.max_payload):
                del self._buf[:1]
                continue

            total = sync_len + FRAME_HDR.size + length + CRC_STRUCT.size
            if len(self._buf) < total:
                break

            hdr = self._buf[off: off + FRAME_HDR.size]
            payload = self._buf[off + FRAME_HDR.size: off + FRAME_HDR.size + length]
            crc_expected = CRC_STRUCT.unpack_from(self._buf, off + FRAME_HDR.size + length)[0]
            crc_actual = crc16_ccitt(hdr + payload)
            if int(crc_expected) != int(crc_actual):
                del self._buf[:1]
                continue

            out.append(bytes(payload))
            del self._buf[:total]

        return out
