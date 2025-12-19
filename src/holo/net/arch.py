"""
holo.net.arch

Helpers for normalising holo:// URIs into binary content identifiers.
"""

from __future__ import annotations

import hashlib

__all__ = [
    "content_id_bytes_from_uri",
    "content_id_hex_from_uri",
    "content_id_bytes_from_stream_frame",
]


def _norm_holo_uri(uri: str) -> str:
    """
    Normalise a holo:// style URI.

    The current implementation trims whitespace only, leaving room for
    future URI rules without breaking compatibility.
    """
    return uri.strip()


def content_id_bytes_from_uri(uri: str, *, digest_size: int = 16) -> bytes:
    """
    Map a holo:// URI to a deterministic binary content identifier.

    The identifier is a BLAKE2s digest so that the mapping is collision
    resistant and stable across processes and machines.
    """
    u = _norm_holo_uri(uri).encode("utf-8", errors="strict")
    return hashlib.blake2s(u, digest_size=int(digest_size)).digest()


def content_id_hex_from_uri(uri: str, *, digest_size: int = 16) -> str:
    """
    Hex-encoded wrapper around `content_id_bytes_from_uri`.
    """
    return content_id_bytes_from_uri(uri, digest_size=digest_size).hex()


def content_id_bytes_from_stream_frame(stream_id: str, frame_idx: int, *, digest_size: int = 16) -> bytes:
    """
    Deterministic content_id for a stream frame, using stream_id and frame index.
    """
    base = f"{stream_id}::{int(frame_idx)}"
    return content_id_bytes_from_uri(base, digest_size=digest_size)
