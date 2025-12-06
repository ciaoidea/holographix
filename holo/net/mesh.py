"""
holo.net.mesh

Minimal mesh helper that gossips holographic chunks over UDP.

This is intentionally simple: it keeps peer management and replication
policy small so it can be swapped or extended without touching codec math
or transport framing.
"""

from __future__ import annotations

import glob
import socket
from typing import Iterable, List, Optional, Tuple

from .arch import content_id_bytes_from_uri
from .transport import ChunkAssembler, send_chunk
from ..cortex.store import CortexStore

__all__ = ["MeshNode"]


class MeshNode:
    """
    Small helper to send and receive holographic chunks with peers.
    """

    def __init__(
        self,
        sock: socket.socket,
        store: CortexStore,
        *,
        peers: Optional[Iterable[Tuple[str, int]]] = None,
        max_payload: int = 1200,
    ) -> None:
        self.sock = sock
        self.store = store
        self.peers: List[Tuple[str, int]] = list(peers) if peers else []
        self.max_payload = int(max_payload)
        self._assembler = ChunkAssembler()

    def add_peer(self, addr: Tuple[str, int]) -> None:
        if addr not in self.peers:
            self.peers.append(addr)

    def remove_peer(self, addr: Tuple[str, int]) -> None:
        if addr in self.peers:
            self.peers.remove(addr)

    def broadcast_chunk_dir(self, content_uri: str, chunk_dir: str) -> None:
        """
        Send every chunk in chunk_dir to all known peers.
        """
        content_id = content_id_bytes_from_uri(content_uri)
        chunk_paths = sorted(glob.glob(f"{chunk_dir}/chunk_*.holo"))
        for path in chunk_paths:
            with open(path, "rb") as f:
                chunk_bytes = f.read()
            chunk_id = self._chunk_id_from_name(path)
            for peer in self.peers:
                send_chunk(
                    self.sock,
                    peer,
                    content_id,
                    chunk_id,
                    chunk_bytes,
                    max_payload=self.max_payload,
                )

    def recv_once(self) -> Optional[Tuple[bytes, int, str]]:
        """
        Receive at most one datagram and attempt reassembly.

        Returns (content_id, chunk_id, stored_path) when a full chunk is
        reconstructed and persisted, or None if nothing was completed.
        """
        try:
            data, _ = self.sock.recvfrom(65536)
        except BlockingIOError:
            return None

        assembled = self._assembler.push_datagram(data)
        if assembled is None:
            return None

        content_id, chunk_id, chunk_bytes = assembled
        stored = self.store.store_chunk_bytes(content_id, chunk_id, chunk_bytes)
        return content_id, chunk_id, stored

    @staticmethod
    def _chunk_id_from_name(path: str) -> int:
        base = path.rsplit("/", 1)[-1]
        if base.startswith("chunk_") and base.endswith(".holo"):
            try:
                return int(base[6:-5])
            except ValueError:
                pass
        return 0
