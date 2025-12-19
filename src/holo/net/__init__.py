"""
holo.net

Networking helpers for Holographix.

This namespace keeps transport framing, content identifiers, and mesh
policy separate from codec math so the two layers can evolve independently.
"""

from __future__ import annotations

from .arch import content_id_bytes_from_uri, content_id_hex_from_uri
from .transport import ChunkAssembler, iter_chunk_datagrams, send_chunk
from .mesh import MeshNode

__all__ = [
    "content_id_bytes_from_uri",
    "content_id_hex_from_uri",
    "ChunkAssembler",
    "iter_chunk_datagrams",
    "send_chunk",
    "MeshNode",
]
