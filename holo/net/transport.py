"""
holo.net.transport

UDP framing, segmentation, and reassembly for holographic chunks.

The codec treats chunks as opaque bytes. This module slices those bytes
into datagrams with lightweight headers so they can travel over lossy,
reordered UDP links and be reassembled on the other side.
"""

from __future__ import annotations

import math
import socket
import struct
import hmac
import hashlib
import time
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:  # pragma: no cover
    AESGCM = None

MAGIC = b"HODT"  # HOlographix Datagram Transport (chunk data)
CONTROL_MAGIC = b"HOCT"  # HOlographix Control Transport

CTRL_TYPE_INV = 1
CTRL_TYPE_WANT = 2

# Header layout:
#   magic[4],
#   content_id[16],
#   chunk_id[u32],
#   frag_idx[u16],
#   frag_total[u16],
#   chunk_len[u32]
_HDR_STRUCT = struct.Struct(">4s16sIHHI")
_HDR_SIZE = _HDR_STRUCT.size

_CTRL_HEADER = struct.Struct(">4sB16sH")  # magic, ctrl_type, content_id, count
MAX_CTRL_CHUNKS = 64

__all__ = [
    "MAGIC",
    "CONTROL_MAGIC",
    "CTRL_TYPE_INV",
    "CTRL_TYPE_WANT",
    "ChunkAssembler",
    "iter_chunk_datagrams",
    "send_chunk",
    "encode_inventory_datagram",
    "encode_want_datagram",
    "parse_control_datagram",
]


def iter_chunk_datagrams(
    content_id: bytes,
    chunk_id: int,
    chunk_bytes: bytes,
    *,
    max_payload: int = 1200,
    auth_key: Optional[bytes] = None,
    enc_key: Optional[bytes] = None,
    key_id: int = 0,
) -> Iterable[bytes]:
    """
    Yield datagrams for one holographic chunk.

    Parameters
    ----------
    content_id : bytes
        16-byte identifier from holo.net.arch.
    chunk_id : int
        Logical chunk identifier.
    chunk_bytes : bytes
        Raw chunk payload produced by the codec.
    max_payload : int, optional
        Maximum UDP payload size in bytes (including header). Defaults to 1200,
        which is conservative for most links.
    auth_key : bytes, optional
        If provided, append an HMAC-SHA256 over (header+payload) for integrity/auth.
    """
    if len(content_id) != 16:
        raise ValueError("content_id must be 16 bytes (blake2s default)")

    mac_len = 32 if auth_key else 0
    enc_overhead = 1 + 12 + 16 if enc_key else 0  # key_id byte + nonce + tag
    overhead = _HDR_SIZE + mac_len + enc_overhead
    if max_payload <= overhead:
        raise ValueError("max_payload too small to hold header")

    payload_size = max_payload - overhead
    frag_total = int(math.ceil(len(chunk_bytes) / float(payload_size)))
    frag_total = max(1, frag_total)

    for frag_idx in range(frag_total):
        start = frag_idx * payload_size
        end = min(len(chunk_bytes), start + payload_size)
        payload = chunk_bytes[start:end]
        header = _HDR_STRUCT.pack(
            MAGIC,
            bytes(content_id),
            int(chunk_id),
            int(frag_idx),
            int(frag_total),
            int(len(chunk_bytes)),
        )
        datagram = header + payload
        if enc_key:
            if AESGCM is None:
                raise RuntimeError("cryptography not available for encryption")
            key_byte = struct.pack("B", key_id & 0xFF)
            nonce = hashlib.blake2s(header + struct.pack(">H", frag_idx), digest_size=12).digest()
            aad = header + key_byte
            cipher = AESGCM(enc_key)
            ct = cipher.encrypt(nonce, payload, aad)
            datagram = header + key_byte + nonce + ct
        if auth_key:
            tag = hmac.new(auth_key, datagram, hashlib.sha256).digest()
            datagram += tag
        yield datagram


def send_chunk(
    sock: socket.socket,
    addr: Tuple[str, int],
    content_id: bytes,
    chunk_id: int,
    chunk_bytes: bytes,
    *,
    max_payload: int = 1200,
    auth_key: Optional[bytes] = None,
    enc_key: Optional[bytes] = None,
    key_id: int = 0,
) -> None:
    """
    Send one holographic chunk as a series of UDP datagrams.
    """
    for datagram in iter_chunk_datagrams(
        content_id,
        chunk_id,
        chunk_bytes,
        max_payload=max_payload,
        auth_key=auth_key,
        enc_key=enc_key,
        key_id=key_id,
    ):
        sock.sendto(datagram, addr)


def encode_inventory_datagram(content_id: bytes, chunk_ids: Iterable[int]) -> bytes:
    """
    Build a control-plane INVENTORY datagram listing chunk_ids we have for a content_id.
    """
    if len(content_id) != 16:
        raise ValueError("content_id must be 16 bytes")
    uniq = sorted({int(c) for c in chunk_ids})
    chunk_list = uniq[:MAX_CTRL_CHUNKS]
    header = _CTRL_HEADER.pack(CONTROL_MAGIC, CTRL_TYPE_INV, bytes(content_id), len(chunk_list))
    body = b"".join(struct.pack(">H", cid) for cid in chunk_list)
    return header + body


def encode_want_datagram(content_id: bytes, chunk_ids: Iterable[int]) -> bytes:
    """
    Build a control-plane WANT datagram requesting specific chunk_ids.
    """
    if len(content_id) != 16:
        raise ValueError("content_id must be 16 bytes")
    uniq = sorted({int(c) for c in chunk_ids})
    chunk_list = uniq[:MAX_CTRL_CHUNKS]
    header = _CTRL_HEADER.pack(CONTROL_MAGIC, CTRL_TYPE_WANT, bytes(content_id), len(chunk_list))
    body = b"".join(struct.pack(">H", cid) for cid in chunk_list)
    return header + body


def parse_control_datagram(data: bytes) -> Optional[Tuple[int, bytes, List[int]]]:
    """
    Parse a control-plane datagram. Returns (ctrl_type, content_id, chunk_ids) or None.
    """
    if len(data) < _CTRL_HEADER.size:
        return None
    magic, ctrl_type, cid, count = _CTRL_HEADER.unpack_from(data, 0)
    if magic != CONTROL_MAGIC:
        return None
    if ctrl_type not in (CTRL_TYPE_INV, CTRL_TYPE_WANT):
        return None
    needed = _CTRL_HEADER.size + count * 2
    if len(data) < needed:
        return None
    chunk_ids = list(struct.unpack_from(f">{count}H", data, _CTRL_HEADER.size)) if count else []
    return int(ctrl_type), bytes(cid), chunk_ids


class ChunkAssembler:
    """
    Reassemble UDP datagrams back into complete chunk bytes.

    Usage
    -----
    assembler = ChunkAssembler()
    data, addr = sock.recvfrom(4096)
    result = assembler.push_datagram(data)
    if result is not None:
        content_id, chunk_id, chunk_bytes = result
    """

    def __init__(
        self,
        *,
        auth_key: Optional[bytes] = None,
        enc_keys: Optional[Dict[int, bytes]] = None,
        max_partial_age: float = 5.0,
    ) -> None:
        self._partials: Dict[Tuple[bytes, int], Dict[str, object]] = {}
        self.auth_key = auth_key
        self.enc_keys = enc_keys or {}
        self.max_partial_age = float(max_partial_age) if max_partial_age is not None else None
        self.counters = {
            "datagrams": 0,
            "invalid": 0,
            "mac_fail": 0,
            "chunks_completed": 0,
            "expired": 0,
        }

    def _expire_partials(self, now: float) -> None:
        if self.max_partial_age is None:
            return
        to_drop = [k for k, v in self._partials.items() if now - v["t0"] > self.max_partial_age]  # type: ignore[index]
        if not to_drop:
            return
        for k in to_drop:
            self._partials.pop(k, None)
            self.counters["expired"] += 1

    def push_datagram(self, data: bytes) -> Optional[Tuple[bytes, int, bytes]]:
        """
        Ingest one datagram. Returns (content_id, chunk_id, chunk_bytes) when
        a full chunk is ready, or None otherwise.
        """
        self.counters["datagrams"] += 1
        now = time.time()
        self._expire_partials(now)

        mac_len = 32 if self.auth_key else 0
        if len(data) < _HDR_SIZE + mac_len:
            self.counters["invalid"] += 1
            return None

        msg = data
        tag = b""
        if mac_len:
            msg, tag = data[:-mac_len], data[-mac_len:]
            if not hmac.compare_digest(tag, hmac.new(self.auth_key, msg, hashlib.sha256).digest()):
                self.counters["mac_fail"] += 1
                return None

        magic, cid, chunk_id, frag_idx, frag_total, chunk_len = _HDR_STRUCT.unpack_from(msg, 0)
        if magic != MAGIC:
            self.counters["invalid"] += 1
            return None
        if frag_total <= 0 or frag_idx >= frag_total:
            self.counters["invalid"] += 1
            return None

        # Encrypted datagrams carry: header + key_id(1) + nonce(12) + ct+tag
        payload = msg[_HDR_SIZE:]
        if len(payload) >= 1 + 12 and (cid, chunk_id) not in self._partials:
            # Heuristic: try decrypt only if a key_id byte is present
            key_id = payload[0]
            if key_id in self.enc_keys:
                if AESGCM is None:
                    self.counters["invalid"] += 1
                    return None
                nonce = payload[1:13]
                ct = payload[13:]
                aad = msg[:_HDR_SIZE] + bytes([key_id])
                try:
                    cipher = AESGCM(self.enc_keys[key_id])
                    payload = cipher.decrypt(nonce, ct, aad)
                except Exception:
                    self.counters["invalid"] += 1
                    return None

        key = (cid, int(chunk_id))
        state = self._partials.get(key)
        if state is None:
            state = {
                "frag_total": int(frag_total),
                "chunk_len": int(chunk_len),
                "frags": {},
                "t0": now,
            }
            self._partials[key] = state
        else:
            if state["frag_total"] != int(frag_total) or state["chunk_len"] != int(chunk_len):
                self.counters["invalid"] += 1
                return None

        frags: Dict[int, bytes] = state["frags"]  # type: ignore[assignment]
        frags[int(frag_idx)] = bytes(payload)

        if len(frags) != state["frag_total"]:
            return None

        ordered: List[bytes] = []
        for idx in range(state["frag_total"]):
            part = frags.get(idx)
            if part is None:
                return None
            ordered.append(part)

        chunk_bytes = b"".join(ordered)
        if len(chunk_bytes) != state["chunk_len"]:
            chunk_bytes = chunk_bytes[: state["chunk_len"]]

        del self._partials[key]
        self.counters["chunks_completed"] += 1
        return cid, int(chunk_id), chunk_bytes
