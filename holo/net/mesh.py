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
import struct
from typing import Dict, Iterable, List, Optional, Tuple

from .arch import content_id_bytes_from_uri
from .transport import (
    ChunkAssembler,
    iter_chunk_datagrams,
    encode_inventory_datagram,
    encode_want_datagram,
    parse_control_datagram,
    CONTROL_MAGIC,
    CTRL_TYPE_INV,
    CTRL_TYPE_WANT,
    MAX_CTRL_CHUNKS,
)
from ..cortex.store import CortexStore

__all__ = ["MeshNode"]

_MCAST_TTL = struct.pack("b", 1)


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
        mcast_groups: Optional[Iterable[Tuple[str, int]]] = None,
        max_payload: int = 1200,
        auth_key: Optional[bytes] = None,
        enc_key: Optional[bytes] = None,
        enc_key_id: int = 0,
        enc_keys: Optional[Dict[int, bytes]] = None,
    ) -> None:
        self.sock = sock
        self.store = store
        self.peers: List[Tuple[str, int]] = list(peers) if peers else []
        self.mcast_groups: List[Tuple[str, int]] = list(mcast_groups) if mcast_groups else []
        self.max_payload = int(max_payload)
        self.auth_key = auth_key
        self.enc_key = enc_key
        self.enc_key_id = int(enc_key_id)
        rx_keys = enc_keys or {}
        if enc_key is not None:
            rx_keys[self.enc_key_id] = enc_key
        self._assembler = ChunkAssembler(auth_key=auth_key, enc_keys=rx_keys)
        self.auth_key = auth_key
        self.counters = {
            "sent_chunks": 0,
            "sent_datagrams": 0,
            "stored_chunks": 0,
            "ctrl_inv_sent": 0,
            "ctrl_want_sent": 0,
            "ctrl_inv_rcvd": 0,
            "ctrl_want_rcvd": 0,
        }
        self.peer_inventory: Dict[Tuple[str, int], Dict[bytes, set[int]]] = {}
        if self.mcast_groups:
            self._join_mcast_groups()
        self.stats = {
            "chunks_seen": 0,
            "datagrams_seen": 0,
            "duplicates": 0,
            "per_content": {},
        }

    def _join_mcast_groups(self) -> None:
        """
        Join all configured multicast groups on the socket (IPv4).
        """
        try:
            for group, port in self.mcast_groups:
                mreq = socket.inet_aton(group) + socket.inet_aton("0.0.0.0")
                self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, _MCAST_TTL)
                self.sock.bind(("", port))
        except OSError:
            pass

    def add_peer(self, addr: Tuple[str, int]) -> None:
        if addr not in self.peers:
            self.peers.append(addr)

    def remove_peer(self, addr: Tuple[str, int]) -> None:
        if addr in self.peers:
            self.peers.remove(addr)

    def add_mcast_group(self, group: Tuple[str, int]) -> None:
        if group not in self.mcast_groups:
            self.mcast_groups.append(group)
            self._join_mcast_groups()

    def broadcast_chunk_dir(self, content_uri: str, chunk_dir: str, *, repeats: int = 1) -> None:
        """
        Send every chunk in chunk_dir to all known peers.
        """
        content_id = content_id_bytes_from_uri(content_uri)
        chunk_paths = sorted(glob.glob(f"{chunk_dir}/chunk_*.holo"))
        repeats = max(1, int(repeats))
        for path in chunk_paths:
            with open(path, "rb") as f:
                chunk_bytes = f.read()
            chunk_id = self._chunk_id_from_name(path)
            for _ in range(repeats):
                for peer in self.peers:
                    for datagram in iter_chunk_datagrams(
                        content_id,
                        chunk_id,
                        chunk_bytes,
                        max_payload=self.max_payload,
                        auth_key=self.auth_key,
                        enc_key=self.enc_key,
                        key_id=self.enc_key_id,
                    ):
                        self._sendto(datagram, peer)
                        self.counters["sent_datagrams"] += 1
                self.counters["sent_chunks"] += 1

    def recv_once(self) -> Optional[Tuple[bytes, int, str]]:
        """
        Receive at most one datagram and attempt reassembly.

        Returns (content_id, chunk_id, stored_path) when a full chunk is
        reconstructed and persisted, or None if nothing was completed.
        """
        try:
            data, addr = self.sock.recvfrom(65536)
        except BlockingIOError:
            return None

        # Control-plane datagrams (inventory/want)
        if data[:4] == CONTROL_MAGIC:
            parsed = parse_control_datagram(data)
            if not parsed:
                return None
            ctrl_type, cid, chunk_ids = parsed
            if ctrl_type == CTRL_TYPE_INV:
                self._handle_inventory(addr, cid, chunk_ids)
                self.counters["ctrl_inv_rcvd"] += 1
            elif ctrl_type == CTRL_TYPE_WANT:
                self._handle_want(addr, cid, chunk_ids)
                self.counters["ctrl_want_rcvd"] += 1
            return None

        self.stats["datagrams_seen"] += 1
        assembled = self._assembler.push_datagram(data)
        if assembled is None:
            return None

        content_id, chunk_id, chunk_bytes = assembled
        stored = self.store.store_chunk_bytes(content_id, chunk_id, chunk_bytes)
        self.counters["stored_chunks"] += 1
        self.stats["chunks_seen"] += 1
        per = self.stats["per_content"].setdefault(content_id, {"seen": set(), "complete_at": None})
        before = len(per["seen"])
        per["seen"].add(int(chunk_id))
        after = len(per["seen"])
        if before == after:
            self.stats["duplicates"] += 1
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

    # ---------------- Control-plane helpers (INV/WANT) ----------------

    def _local_chunk_ids(self, content_id: bytes) -> set[int]:
        paths = self.store.iter_chunks(content_id)
        ids = set()
        for p in paths:
            cid = self._chunk_id_from_name(p)
            ids.add(int(cid))
        return ids

    def _handle_inventory(self, peer: Tuple[str, int], content_id: bytes, chunk_ids: Iterable[int]) -> None:
        inv = self.peer_inventory.setdefault(peer, {})
        inv[content_id] = set(int(c) for c in chunk_ids)

        local = self._local_chunk_ids(content_id)
        missing = [c for c in chunk_ids if c not in local]
        if not missing:
            return
        # Request only a limited number to avoid flooding
        want_ids = missing[:MAX_CTRL_CHUNKS]
        dg = encode_want_datagram(content_id, want_ids)
        self._sendto(dg, peer)
        self.counters["ctrl_want_sent"] += 1

    def _handle_want(self, peer: Tuple[str, int], content_id: bytes, chunk_ids: Iterable[int]) -> None:
        local = self._local_chunk_ids(content_id)
        to_send = [c for c in chunk_ids if c in local]
        if not to_send:
            return
        for chunk_id in to_send:
            path = self.store.content_dir(content_id) + f"/chunk_{int(chunk_id):04d}.holo"
            try:
                with open(path, "rb") as f:
                    chunk_bytes = f.read()
            except OSError:
                continue
            for datagram in iter_chunk_datagrams(
                content_id,
                chunk_id,
                chunk_bytes,
                max_payload=self.max_payload,
                auth_key=self.auth_key,
                enc_key=self.enc_key,
                key_id=self.enc_key_id,
            ):
                self._sendto(datagram, peer)
                self.counters["sent_datagrams"] += 1
            self.counters["sent_chunks"] += 1

    def send_inventory(self, content_id: bytes, peers: Optional[Iterable[Tuple[str, int]]] = None) -> None:
        chunk_ids = self._local_chunk_ids(content_id)
        if not chunk_ids:
            return
        dg = encode_inventory_datagram(content_id, chunk_ids)
        for peer in peers or self.peers:
            self._sendto(dg, peer)
            self.counters["ctrl_inv_sent"] += 1

    def _sendto(self, data: bytes, addr: Tuple[str, int]) -> None:
        """
        Send a datagram to a unicast or multicast address.
        """
        try:
            self.sock.sendto(data, addr)
        except OSError:
            pass
